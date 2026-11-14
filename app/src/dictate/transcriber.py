"""Speech to text: runs a speech model (NVIDIA Parakeet TDT 0.6B v2) on a finished recording.

Steps for one recording
-----------------------
  1. Find speech ...... a speech detector (Silero, bundled with faster-whisper) marks the stretches
                        of the recording that contain speech. If there are none, the model is not
                        run at all.
  2. Trim ............. only the silence BEFORE the first speech and AFTER the last speech is
                        removed. Everything in between is left exactly as recorded, pauses and
                        noises included. 0.4 s of padding is kept on the outer side of the first
                        and last speech. That padding is what protects the first syllable: tested
                        on 32 clips with soft word starts ("Who", "Thank", "Fine"), trimming with
                        no padding cut about 40 ms off the start of speech (up to 134 ms) and lost
                        a first word; with 200 ms or more of padding nothing was cut. 400 ms is
                        double that margin.
  3. Transcribe ....... the engine (see engines.py) turns the trimmed speech into text.

The model runs entirely on the CPU, on this machine. See engines.py for the model itself.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dictate.engines import Engine, ParakeetEngine
from dictate.recorder import SAMPLE_RATE, Recording

log = logging.getLogger(__name__)

MIN_SPEECH_SECONDS = 0.2  # less speech than this in a recording counts as no speech
SPEECH_PAD_SECONDS = 0.4  # kept before the first speech and after the last (protects the start)
MIN_SILENCE_SECONDS = 0.5  # a pause at least this long ends a speech stretch

# A stretch of speech: (first sample, last sample) within the recording.
Span = tuple[int, int]


@dataclass(frozen=True)
class Transcript:
    text: str  # empty if nothing was transcribed
    audio_seconds: float  # the whole recording
    took_seconds: float
    skipped: str | None = None  # why the model was not run, e.g. "no speech detected"
    sent_seconds: float = 0.0  # what reached the model after silence and noise were removed


def find_speech(samples: np.ndarray) -> list[Span]:
    """The stretches of the recording that contain speech, according to the Silero detector.

    The spans are the detector's raw ones, with no padding: padding is added in trim_to_speech.
    Key clicks, coughs and noise bursts are not speech, so they do not appear here.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        min_speech_duration_ms=100,
        min_silence_duration_ms=int(MIN_SILENCE_SECONDS * 1000),
        speech_pad_ms=0,
    )
    return [(c["start"], c["end"]) for c in get_speech_timestamps(samples, options)]


def trim_to_speech(
    samples: np.ndarray, spans: list[Span], pad_seconds: float = SPEECH_PAD_SECONDS
) -> np.ndarray:
    """Cut off the leading and trailing silence only; keep everything in between untouched.

    The result runs from ``pad_seconds`` before the first speech to ``pad_seconds`` after the
    last speech. Pauses and noises between pieces of speech are NOT removed. The padding is
    clamped to the recording, so speech at the very start of the recording keeps all the audio
    before it.
    """
    if not spans:
        return samples[:0]
    pad = int(pad_seconds * SAMPLE_RATE)
    start = max(0, spans[0][0] - pad)
    end = min(len(samples), spans[-1][1] + pad)
    return samples[start:end]


class Transcriber:
    def __init__(
        self,
        engine: Engine | None = None,
        *,
        speech_finder: Callable[[np.ndarray], list[Span]] = find_speech,
    ) -> None:
        self._engine: Engine = engine if engine is not None else ParakeetEngine()
        self._find_speech = speech_finder  # swapped out in tests
        self._loaded = False

    @property
    def model_name(self) -> str:
        return self._engine.name

    def load(self) -> None:
        """Load the model (downloaded on the very first run). Call at startup, not on the hotkey
        path: it takes seconds."""
        started = time.perf_counter()
        self._engine.load()
        # The speech detector loads on first use (a large library import plus its own small
        # model, a few seconds in all). Run it once now so the first dictation is not slow.
        self._find_speech(np.zeros(SAMPLE_RATE, dtype=np.float32))
        self._loaded = True
        log.info("model %s ready in %.1f s", self.model_name, time.perf_counter() - started)

    def transcribe(self, recording: Recording) -> Transcript:
        """Turn a recording into text. Slow (seconds): call from a worker thread."""
        if not self._loaded:
            raise RuntimeError("call load() before transcribe()")
        started = time.perf_counter()
        samples = recording.samples
        audio_seconds = recording.seconds

        # Step 1: where is the speech? If there is hardly any, do not run the model at all.
        spans = self._find_speech(samples)
        heard = sum(end - start for start, end in spans) / SAMPLE_RATE
        if heard < MIN_SPEECH_SECONDS:
            return Transcript(
                "", audio_seconds, time.perf_counter() - started, "no speech detected"
            )

        # Step 2: cut off the leading and trailing silence (keeping 0.4 s of padding); the audio
        # in between is left exactly as recorded.
        speech = trim_to_speech(samples, spans)

        # Step 3: transcribe what is left.
        text = self._engine.transcribe(speech)
        return Transcript(
            text,
            audio_seconds,
            time.perf_counter() - started,
            sent_seconds=len(speech) / SAMPLE_RATE,
        )


class TranscriptionWorker:
    """Transcribes recordings one at a time on a background thread.

    The hotkey must never wait for Whisper, so recordings are handed over with submit(), which
    returns at once. Results come back through ``on_transcript`` on the worker thread.
    """

    def __init__(self, transcriber: Transcriber, on_transcript: Callable[[Transcript], None]):
        self._transcriber = transcriber
        self._on_transcript = on_transcript
        self._queue: queue.SimpleQueue[Recording | None] = queue.SimpleQueue()
        self._pending = 0  # queued plus the one being transcribed
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="transcriber", daemon=True)

    @property
    def busy(self) -> bool:
        """True while any recording is queued or being transcribed."""
        with self._lock:
            return self._pending > 0

    def start(self) -> None:
        self._thread.start()

    def submit(self, recording: Recording) -> None:
        with self._lock:
            self._pending += 1
        self._queue.put(recording)

    def stop(self, timeout: float = 10.0) -> None:
        """Finish queued work, then end the thread. A running transcription cannot be cut short."""
        if not self._thread.is_alive():  # never started (startup failed) or already stopped
            return
        self._queue.put(None)
        self._thread.join(timeout)

    def _run(self) -> None:
        while (recording := self._queue.get()) is not None:
            try:
                self._on_transcript(self._transcriber.transcribe(recording))
            except Exception:
                log.exception("transcription failed")
            finally:
                with self._lock:
                    self._pending -= 1
