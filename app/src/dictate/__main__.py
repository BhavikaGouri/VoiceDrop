"""Entry point: a push-to-talk app. Hold Right Alt, speak, and the text appears at your cursor.

What starts, and on which thread
--------------------------------
  main thread ........ the startup window (the user guide with an OK button); Tk needs this thread
  startup thread ..... loads the speech model, then starts everything below, then marks "Ready"
  hotkey threads ..... the Right Alt hook, its timer, and a dispatcher that runs the tasks below
  audio callback ..... PortAudio's own thread, filling the in-memory recording buffer
  transcriber thread . runs the speech model, so the hotkey and the window never wait for it
  inserter thread .... pastes the text at the cursor (clipboard, Ctrl+V, clipboard restored)

Because the model runs on its own thread, the app keeps answering the hotkey and the window
while a transcription is in progress. The model is loaded once at startup and stays in memory.

How one dictation is handled
----------------------------
  tap Right Alt ............ open_stream()  heads-up: the microphone stream starts and audio is
                                            buffered from this instant, before you speak
  press again and hold ..... start_mic()    the dictation is confirmed; keep capturing while held
  release .................. stop_mic()     stop the stream and queue the audio for transcription
  anything else ............ discard()      the first tap was a mistake: drop the buffer

Audio never touches the disk: it lives in an in-memory buffer and goes straight to the model.

The first syllable is not lost: the buffer begins at the first tap, so the recording contains it.
Before transcription only the leading and trailing silence is cut off, keeping 0.4 s of padding,
which tests showed never cuts the start of speech.

The first tap counts as a mistake when Right Alt is used with any other key (Alt+Tab,
Ctrl+Alt+..., Alt+F4), when no second press follows within 200 ms, when the first press is
held longer than 250 ms, or when the second press is released in under 300 ms. A dictation
is cut off at 5 minutes.

When a transcription finishes, the text is printed to the console (a line starting ">>>") and
pasted at the cursor.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

import numpy as np

from dictate.gesture import DiscardReason
from dictate.gui import StartupStatus, show_user_guide
from dictate.hotkey import Handlers, HotkeyListener
from dictate.inserter import TextInserter
from dictate.recorder import Recorder, RecorderError, Recording
from dictate.transcriber import Transcriber, Transcript, TranscriptionWorker

log = logging.getLogger("dictate")


class DictationTasks:
    """The work triggered by the hotkey.

    Every method runs on the hotkey's dispatcher thread, one call at a time and in the order
    the user did things. They must return quickly: start work and return, never wait for it.
    """

    def __init__(self, worker: TranscriptionWorker) -> None:
        self.worker = worker
        # The recorder tells us when a dictation hits the 5 minute limit, so it can be ended
        # even though the user is still holding Right Alt.
        self.recorder = Recorder(on_limit=self.on_max_length)

    def open_stream(self) -> None:
        # Task: start the microphone stream and begin filling the in-memory buffer.
        # This runs on the first tap, before we know it is a real dictation, so the buffer has
        # a head start on the speech. If the tap turns out to be a mistake, discard() drops it.
        try:
            self.recorder.open()
        except RecorderError as exc:
            # No microphone, or it is busy. start_mic() and stop_mic() then find nothing open
            # and do nothing. (How to tell the user is still an open decision, B11.)
            log.error("cannot open the microphone: %s", exc)
            return
        log.info("OPEN STREAM: microphone is buffering")

    def start_mic(self) -> None:
        # Task: confirm the dictation. The audio buffered since the first tap stays as the
        # start of the recording, and capture continues for as long as Right Alt stays held.
        # Also the place to show the "listening" overlay / tray state, once they exist.
        if self.recorder.confirm():
            log.info("START MIC: dictation confirmed, recording while Right Alt is held")

    def stop_mic(self) -> None:
        # Task: stop the stream and take everything that was captured, from the first tap on.
        recording = self.recorder.stop()
        if recording is not None:
            self._hand_over(recording, why="released")

    def on_max_length(self) -> None:
        # Task: the 5 minute limit was reached while Right Alt is still held. End the dictation
        # now, as if it had been released. The real release that follows finds the recorder
        # already stopped and does nothing.
        recording = self.recorder.stop()
        if recording is not None:
            self._hand_over(recording, why="5 minute limit reached")

    def discard(self, reason: DiscardReason) -> None:
        # Task: throw away whatever open_stream() buffered and stop the stream. Nothing is
        # transcribed or pasted. `reason` says why the first tap turned out to be a mistake.
        self.recorder.discard()
        log.info("DISCARD (%s): buffer dropped, stream stopped", reason.value)

    def _hand_over(self, recording: Recording, *, why: str) -> None:
        # Task: queue the whole recording for transcription and return at once; the speech
        # model takes seconds and must never run on this thread. The worker keeps a queue, so a
        # dictation started while an earlier one is still being transcribed is not lost: it is
        # transcribed next (decision A7).
        peak = float(np.max(np.abs(recording.samples))) if len(recording.samples) else 0.0
        log.info(
            "STOP MIC (%s): %.2f s of audio, peak level %.2f%s. Queued for transcription.",
            why,
            recording.seconds,
            peak,
            ", cut at the limit" if recording.truncated else "",
        )
        self.worker.submit(recording)


class Inserter(Protocol):
    def insert(self, text: str) -> None: ...


def deliver(transcript: Transcript, inserter: Inserter) -> None:
    """What happens to a finished transcription: print it, then paste it at the cursor.

    Runs on the transcriber thread. Printing comes first so the text is always visible in the
    console, even if the paste fails or the cursor is somewhere that cannot take text.
    """
    if transcript.skipped:
        print(f"\n>>> ({transcript.skipped})\n", flush=True)
        return
    log.info(
        "TRANSCRIPT: %.1f s recorded, %.1f s after trimming, %.1f s to transcribe",
        transcript.audio_seconds,
        transcript.sent_seconds,
        transcript.took_seconds,
    )
    print(f"\n>>> {transcript.text or '(empty)'}\n", flush=True)
    inserter.insert(transcript.text)  # queued; the inserter thread does the pasting


class DictationApp:
    """Owns every part of the app, and the order they start and stop in."""

    def __init__(
        self,
        transcriber: Transcriber | None = None,
        inserter: TextInserter | None = None,
    ) -> None:
        self.status = StartupStatus()
        self.transcriber = transcriber or Transcriber()
        self.inserter = inserter or TextInserter(
            on_notice=lambda m: print(f">>> ({m})", flush=True)
        )
        self.worker = TranscriptionWorker(
            self.transcriber, on_transcript=lambda t: deliver(t, self.inserter)
        )
        self.tasks = DictationTasks(self.worker)
        self.listener: HotkeyListener | None = None
        self._stopped = False

    def start_in_background(self) -> threading.Thread:
        """Start everything on a background thread, so the window can open straight away."""
        thread = threading.Thread(target=self._start, name="startup", daemon=True)
        thread.start()
        return thread

    def _start(self) -> None:
        try:
            # Step 1: load the speech model. This is the slow part (seconds; a minute or two the
            # very first time, which downloads it). It stays in memory until the app exits, so
            # every dictation after this one starts transcribing at once.
            self.status.set_loading(
                "Loading the speech model (the first start downloads it, about 630 MB)..."
            )
            log.info(
                "Loading %s (the first run downloads the model)...", self.transcriber.model_name
            )
            self.transcriber.load()

            # Step 2: start the threads that do the work after a dictation.
            self.worker.start()
            self.inserter.start()

            # Step 3: create the microphone stream now (about half a second) so the first tap
            # only has to start it. If the microphone is missing we carry on; the first tap
            # will try again.
            try:
                self.tasks.recorder.prepare()
            except RecorderError as exc:
                log.warning("microphone not ready: %s", exc)

            # Step 4: start listening for Right Alt. From here on the app is live.
            self.listener = HotkeyListener(
                Handlers(
                    on_stream_open=self.tasks.open_stream,
                    on_mic_start=self.tasks.start_mic,
                    on_mic_stop=self.tasks.stop_mic,
                    on_discard=self.tasks.discard,
                )
            )
            self.listener.start()
            self.status.set_ready()
            log.info("Ready. Tap Right Alt, then press it again within 200 ms and hold.")
        except Exception as exc:
            log.exception("startup failed")
            self.status.set_failed(f"The app could not start: {exc}")

    @property
    def failed(self) -> bool:
        return self.status.snapshot()[0] == "failed"

    def stop(self) -> None:
        """Stop everything, in the reverse order it started. Safe to call twice."""
        if self._stopped:
            return
        self._stopped = True
        if self.listener is not None:
            self.listener.stop()  # no new dictations
        self.tasks.recorder.close()  # release the microphone
        self.worker.stop()  # finish any transcription already queued ...
        self.inserter.stop()  # ... and paste it


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s.%(msecs)03d  %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("dictate.recorder").setLevel(logging.DEBUG)  # shows stream start time

    app = DictationApp()
    app.start_in_background()
    try:
        # The window opens at once and says "Loading" until the background thread finishes.
        # This blocks until the user presses OK; the app keeps running behind it.
        if not show_user_guide(app.status):
            log.warning("Running without the guide window (it could not open).")
        log.info("Guide closed. The app keeps running in the background. Ctrl+C here to quit.")
        while not app.failed:
            time.sleep(0.5)  # all the work happens on the app's own threads
        log.error("Stopping: the app could not start (see above).")
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()


if __name__ == "__main__":
    main()
