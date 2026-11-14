"""Speech recognition engines. Each one turns a stretch of speech into text, locally.

Parakeet is the engine the app uses. Whisper small.en is kept alongside it so the two can be
compared and swapped; nothing in the app calls it unless asked to.

Both run entirely on this machine's CPU. The one time anything touches the internet is the first
download of the model files. After that the model is loaded straight from the downloaded folder
and nothing is requested from Hugging Face, not even a "is there a newer version" check.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from dictate.recorder import SAMPLE_RATE

# The Hugging Face cache warns about symlinks on Windows machines without Developer Mode. The
# cache works fine without them, so keep the console quiet. Must be set before the first import
# of huggingface_hub, which happens when a model loads.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

log = logging.getLogger(__name__)

# NVIDIA Parakeet TDT 0.6B v2: English only, 8-bit ONNX export (licence CC-BY-4.0, NVIDIA).
PARAKEET_REPO = "istupakov/parakeet-tdt-0.6b-v2-onnx"
PARAKEET_FILES = (
    "config.json",
    "vocab.txt",
    "encoder-model.int8.onnx",  # 637 MB, nearly all of the download
    "decoder_joint-model.int8.onnx",
)

# The ONNX runtime handles roughly 20 to 30 s of audio per call, so longer dictations are cut into
# chunks of at most this many seconds. 20 s keeps well inside that limit.
MAX_CHUNK_SECONDS = 20.0
CUT_SEARCH_SECONDS = 4.0  # where to look for a quiet spot to cut: the last 4 s before the limit


class Engine(Protocol):
    """What the transcriber needs from a speech model."""

    name: str

    def load(self) -> None:
        """Load the model (downloading it on the very first run) and warm it up."""

    def transcribe(self, samples: np.ndarray) -> str:
        """Text for speech: float32, mono, 16 kHz. May be called from a worker thread."""


def chunk_audio(samples: np.ndarray, max_seconds: float = MAX_CHUNK_SECONDS) -> list[np.ndarray]:
    """Split long audio into pieces of at most ``max_seconds``, cutting at quiet moments.

    The cut goes at the quietest 50 ms in the last few seconds before the limit, which in speech
    is a pause between words (or, after silence has been trimmed, the gap between two stretches
    of speech), so a word is not cut in half. Audio within the limit comes back as one piece.
    """
    limit = int(max_seconds * SAMPLE_RATE)
    window = int(0.05 * SAMPLE_RATE)
    search = int(CUT_SEARCH_SECONDS * SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    start = 0
    while len(samples) - start > limit:
        lo, hi = start + limit - search, start + limit
        frames = (hi - lo) // window
        energy = (samples[lo : lo + frames * window].reshape(frames, window) ** 2).mean(axis=1)
        cut = lo + int(np.argmin(energy)) * window + window // 2
        chunks.append(samples[start:cut])
        start = cut
    chunks.append(samples[start:])
    return chunks


def find_or_download_parakeet(
    download: Callable[..., str] | None = None,
) -> Path:
    """Folder holding the Parakeet files. Offline if they are already downloaded.

    ``download`` is huggingface_hub.snapshot_download; tests pass a fake.
    """
    if download is None:
        from huggingface_hub import snapshot_download

        download = snapshot_download

    def folder(**options: Any) -> Path:
        return Path(
            download(PARAKEET_REPO, allow_patterns=list(PARAKEET_FILES), **options)  # type: ignore[arg-type]
        )

    # Step 1: look on disk only. local_files_only means no network request of any kind.
    try:
        path = folder(local_files_only=True)
        if all((path / name).exists() for name in PARAKEET_FILES):
            return path
    except Exception:  # not in the cache yet (huggingface_hub raises its own error types)
        pass

    # Step 2: first run only. Download the files (about 630 MB), once.
    log.info("downloading the Parakeet model (about 630 MB) from Hugging Face, once ...")
    return folder()


class ParakeetEngine:
    """NVIDIA Parakeet TDT 0.6B v2, 8-bit, through onnx-asr and onnxruntime, on the CPU."""

    name = "parakeet-tdt-0.6b-v2"

    def __init__(self, *, threads: int | None = None, max_chunk_seconds: float = MAX_CHUNK_SECONDS):
        # Measured twice on a 12-thread laptop (6.4 s clip). The machine ran at two different
        # speeds, but the pattern held: 12 threads was clearly the slowest, and 4 threads was within
        # about 10% of the best both times (busy: 2 threads 4.0 s, 4 threads 2.3 s, 6 threads
        # 2.1 s, 8 threads 2.3 s, 12 threads 3.6 s; idle: 0.77, 0.87, 1.00, 1.04, 1.53 s). Four
        # also leaves cores free for the microphone and the rest of the app.
        self._threads = threads or min(4, os.cpu_count() or 4)
        self._max_chunk_seconds = max_chunk_seconds
        self._model: Any = None

    def load(self) -> None:
        import onnx_asr  # imported late: it pulls in onnxruntime, which is slow to import
        import onnxruntime

        started = time.perf_counter()
        model_dir = find_or_download_parakeet()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = self._threads
        self._model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v2", model_dir, quantization="int8", sess_options=options
        )
        # The first call after loading is much slower than the rest (about 5 s against 1.6 s), so
        # make it now, on silence, while nobody is waiting.
        self._model.recognize(np.zeros(SAMPLE_RATE, dtype=np.float32), sample_rate=SAMPLE_RATE)
        log.info("%s loaded in %.1f s", self.name, time.perf_counter() - started)

    def transcribe(self, samples: np.ndarray) -> str:
        parts = (
            self._model.recognize(chunk, sample_rate=SAMPLE_RATE)
            for chunk in chunk_audio(samples, self._max_chunk_seconds)
        )
        return " ".join(part.strip() for part in parts if part.strip())


class WhisperEngine:
    """Whisper small.en through faster-whisper, 8-bit, on the CPU (the previous engine)."""

    name = "whisper-small.en"

    def __init__(self, *, threads: int | None = None, beam_size: int = 5):
        self._threads = threads or min(8, os.cpu_count() or 4)
        self._beam_size = beam_size
        self._model: Any = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        started = time.perf_counter()
        for local_only in (True, False):  # use the downloaded copy if there is one
            try:
                self._model = WhisperModel(
                    "small.en",
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=self._threads,
                    local_files_only=local_only,
                )
                break
            except Exception:
                if not local_only:
                    raise
        segments, _ = self._model.transcribe(
            np.zeros(SAMPLE_RATE, dtype=np.float32), language="en", beam_size=self._beam_size
        )
        list(segments)  # warm-up: decoding happens while the generator is consumed
        log.info("%s loaded in %.1f s", self.name, time.perf_counter() - started)

    def transcribe(self, samples: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            samples,
            language="en",
            beam_size=self._beam_size,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
