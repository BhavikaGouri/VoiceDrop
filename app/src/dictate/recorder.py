"""Microphone recorder: a WASAPI input stream that starts buffering on the first tap.

Life of one recording, and which hotkey task drives each step
-------------------------------------------------------------
  prepare()  app start .......... create the stream once, without starting it
  open()     first tap .......... start the stream and begin buffering (heads-up)
  confirm()  second press held .. the buffer so far becomes the start of the recording
  stop()     release ............ stop the stream and return everything captured
  discard()  mistaken first tap . stop the stream and throw the buffer away
  close()    app exit ........... release the stream for good

Why the first tap only *starts* a stream that already exists: creating a WASAPI stream took
about half a second on the test laptop, while starting an existing one takes a few milliseconds.
Speech begins right after the second press, so creating the stream on the first tap would miss
the first syllable. Creating it at startup and just starting it on the tap does not.

Audio format: 16 kHz, mono, float32, which is exactly what Whisper expects, so nothing needs
resampling later. WASAPI shared mode only accepts that rate with ``auto_convert`` on, which
makes Windows do the sample-rate and channel conversion for us.
"""

from __future__ import annotations

import ctypes
import enum
import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000  # Hz, what Whisper expects
MAX_SECONDS = 300  # longest single dictation: 5 minutes (decision A8)


class RecorderError(RuntimeError):
    """The microphone could not be opened (no device, WASAPI missing, device busy)."""


@dataclass(frozen=True)
class Recording:
    samples: np.ndarray  # float32, mono, SAMPLE_RATE Hz, values in -1.0..1.0
    sample_rate: int
    truncated: bool  # True if MAX_SECONDS was reached and the rest was dropped

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.sample_rate


class _State(enum.Enum):
    IDLE = "idle"  # stream not running
    BUFFERING = "buffering"  # first tap seen: stream running, not yet a confirmed dictation
    RECORDING = "recording"  # second press held: a confirmed dictation


_com = threading.local()


def _init_com_on_this_thread() -> None:
    """Initialise COM on the calling thread, once.

    WASAPI is built on COM. A thread that has not initialised it gets an "Unanticipated host
    error" from PortAudio when it creates or starts a stream. The hotkey tasks run on their
    own threads, not the main one, so each of them needs this before touching the stream.
    """
    if getattr(_com, "ready", False) or not sys.platform.startswith("win"):
        return
    try:
        ctypes.OleDLL("ole32").CoInitializeEx(None, 0)  # 0 = multithreaded apartment
    except OSError:
        # RPC_E_CHANGED_MODE: the thread already has COM in another mode. That is fine.
        log.debug("COM already initialised on this thread in another mode")
    _com.ready = True


# A stream factory turns an audio callback into a stream object with start/stop/close.
# The real one opens WASAPI; tests pass a fake so they need no microphone.
StreamFactory = Callable[[Callable[..., None]], Any]


def open_wasapi_stream(callback: Callable[..., None]) -> Any:
    """Create (but do not start) a 16 kHz mono input stream on the default WASAPI microphone."""
    # Step 1: find the WASAPI host API. Windows also offers MME, DirectSound and WDM-KS, which
    # are older and slower; we want WASAPI specifically.
    wasapi = next((a for a in sd.query_hostapis() if "WASAPI" in a["name"]), None)
    if wasapi is None:
        raise RecorderError("WASAPI is not available on this system")

    # Step 2: use WASAPI's own default input device (the microphone Windows has selected).
    device = wasapi["default_input_device"]
    if device < 0:
        raise RecorderError("no microphone found")

    # Step 3: open the stream in shared mode, so other apps can keep using the microphone.
    # auto_convert lets Windows convert from the device's own format to 16 kHz mono for us.
    try:
        return sd.InputStream(
            device=device,
            channels=1,
            samplerate=SAMPLE_RATE,
            dtype="float32",
            latency="low",  # small buffers: audio reaches our callback with minimal delay
            callback=callback,
            extra_settings=sd.WasapiSettings(auto_convert=True),
        )
    except sd.PortAudioError as exc:
        raise RecorderError(f"could not open the microphone: {exc}") from exc


class Recorder:
    """Buffers microphone audio in memory. Methods are called from the hotkey dispatcher thread."""

    def __init__(
        self,
        *,
        max_seconds: float = MAX_SECONDS,
        on_limit: Callable[[], None] | None = None,
        stream_factory: StreamFactory = open_wasapi_stream,
        keep_stream: bool = True,
    ) -> None:
        self._max_frames = int(max_seconds * SAMPLE_RATE)
        self._on_limit = on_limit  # called once, on its own thread, if max_seconds is reached
        self._stream_factory = stream_factory
        # True: the stream stays created between dictations, so every first tap starts fast.
        # False: it is closed after each dictation and created again on the next first tap
        # (about half a second slower). Use False if a stream that exists but is stopped turns
        # out to show Windows' "microphone in use" indicator.
        self._keep_stream = keep_stream

        # _state_lock serialises prepare/open/confirm/stop/discard/close, which may come from the
        # dispatcher thread or from the max-length thread. _buffer_lock only protects the audio
        # buffer, and is the only lock the audio callback takes, so the callback can never wait
        # on a stream start or stop.
        self._state_lock = threading.Lock()
        self._buffer_lock = threading.Lock()

        self._state = _State.IDLE
        self._stream: Any = None
        self._chunks: list[np.ndarray] = []
        self._frames = 0
        self._collecting = False  # the audio callback keeps audio only while this is True
        self._confirmed = False  # second press seen (the max-length rule only applies then)
        self._limit_fired = False

    # -- app lifetime --------------------------------------------------------------------

    def prepare(self) -> None:
        """App start: create the stream now so the first tap only has to start it."""
        with self._state_lock:
            _init_com_on_this_thread()
            self._ensure_stream()

    def close(self) -> None:
        """App exit: stop whatever is running and release the stream for good."""
        with self._state_lock:
            _init_com_on_this_thread()
            self._shutdown_stream(force_close=True)

    # -- hotkey tasks --------------------------------------------------------------------

    def open(self) -> None:
        """First tap: start the stream and begin buffering. Raises RecorderError on failure."""
        with self._state_lock:
            _init_com_on_this_thread()

            # Step 1: a leftover recording means an earlier gesture was never closed. Drop it
            # so the new gesture starts from a clean buffer.
            if self._state is not _State.IDLE:
                log.warning("open() while %s: dropping the previous recording", self._state.value)
                self._shutdown_stream()

            # Step 2: start from an empty buffer, ready to collect.
            with self._buffer_lock:
                self._chunks = []
                self._frames = 0
                self._confirmed = False
                self._limit_fired = False
                self._collecting = True

            # Step 3: make sure a stream exists (a no-op after prepare()), then start it. From
            # here on the audio callback fills the buffer on PortAudio's own thread.
            try:
                self._ensure_stream()
                started = time.perf_counter()
                self._stream.start()
            except Exception as exc:
                # A stream that failed may be tied to a microphone that has gone away, so throw
                # it out; the next first tap builds a fresh one on the current default device.
                self._shutdown_stream(force_close=True)
                if isinstance(exc, RecorderError):
                    raise
                raise RecorderError(f"could not start the microphone: {exc}") from exc
            self._state = _State.BUFFERING
            log.debug("stream started in %.0f ms", (time.perf_counter() - started) * 1000)

    def confirm(self) -> bool:
        """Second press held: keep what was buffered and carry on. False if nothing was open."""
        with self._state_lock:
            # Step 4: promote the buffer to a real recording. Nothing is copied or restarted:
            # the audio from the first tap onwards simply becomes the start of the dictation.
            if self._state is not _State.BUFFERING:
                log.warning("confirm() while %s: ignored", self._state.value)
                return False
            self._state = _State.RECORDING
            with self._buffer_lock:
                self._confirmed = True
            return True

    def stop(self) -> Recording | None:
        """Release after a real hold: stop the stream and return the audio, or None."""
        with self._state_lock:
            _init_com_on_this_thread()
            if self._state is _State.IDLE:
                # Normal after the max length was reached: stop already ran, and the key
                # release that follows arrives here with nothing left to do.
                log.debug("stop() with nothing recording")
                return None
            if self._state is _State.BUFFERING:
                log.warning("stop() before confirm(): treating it as a discard")
                self._shutdown_stream()
                return None

            # Step 5: stop collecting first so late callbacks are ignored, then stop the stream.
            # stop() on the stream waits for any callback already running to finish. The buffer
            # is kept (keep_audio) because we are about to read it.
            self._shutdown_stream(keep_audio=True)

            # Step 6: join the chunks into one array. The stream is stopped, so nothing can be
            # added while we read it.
            with self._buffer_lock:
                samples = (
                    np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)
                )
                truncated = self._limit_fired
                self._chunks = []
                self._frames = 0
            return Recording(samples=samples, sample_rate=SAMPLE_RATE, truncated=truncated)

    def discard(self) -> None:
        """Mistaken first tap: stop the stream and drop everything captured."""
        with self._state_lock:
            _init_com_on_this_thread()
            if self._state is _State.IDLE:
                return
            # Step 7: stopping the stream also clears the buffer. Nothing is kept.
            self._shutdown_stream()

    # -- internals -----------------------------------------------------------------------

    def _ensure_stream(self) -> None:
        """Create the stream if there is none. Creating is slow (about half a second)."""
        if self._stream is None:
            created = time.perf_counter()
            self._stream = self._stream_factory(self._on_audio)
            log.debug("stream created in %.0f ms", (time.perf_counter() - created) * 1000)

    def _shutdown_stream(self, *, keep_audio: bool = False, force_close: bool = False) -> None:
        """Stop collecting and stop the stream; go back to IDLE.

        The buffer is cleared too, unless ``keep_audio`` is set (stop() reads it first). The
        stream object is released only with ``force_close`` or when ``keep_stream`` is off.
        """
        with self._buffer_lock:
            self._collecting = False
        stream = self._stream
        if stream is not None:
            release = force_close or not self._keep_stream
            try:
                stream.stop()
            except Exception:
                log.exception("stopping the microphone stream failed")
                release = True  # a stream that cannot stop is not worth keeping
            if release:
                self._stream = None
                try:
                    stream.close()
                except Exception:
                    log.exception("closing the microphone stream failed")
        if not keep_audio:
            with self._buffer_lock:
                self._chunks = []
                self._frames = 0
        self._state = _State.IDLE

    def _on_audio(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Audio callback, run by PortAudio for every block of microphone data.

        It must be quick and must not raise. Anything slow here causes audio glitches.
        """
        # Step 8: report glitches (for example input overflow) but keep going.
        if status:
            log.warning("audio status: %s", status)

        fire_limit = False
        with self._buffer_lock:
            if not self._collecting:
                return

            # Step 9: enforce the maximum length. Frames past the limit are dropped.
            room = self._max_frames - self._frames
            if room <= 0:
                return

            # Step 10: keep a copy. PortAudio reuses `indata` after this call returns, so
            # holding on to it would overwrite earlier audio with later audio.
            chunk = np.array(indata[:room, 0], dtype=np.float32, copy=True)
            self._chunks.append(chunk)
            self._frames += len(chunk)

            # Step 11: the limit counts only for a confirmed dictation. Tell the owner once.
            if self._frames >= self._max_frames and self._confirmed and not self._limit_fired:
                self._limit_fired = True
                fire_limit = True

        if fire_limit and self._on_limit is not None:
            # A separate thread: the owner will stop the recorder, which waits for this very
            # callback to finish, so calling it from here would deadlock.
            threading.Thread(target=self._on_limit, name="recorder-limit", daemon=True).start()
