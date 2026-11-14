from pathlib import Path

import numpy as np
import pytest

from dictate import engines
from dictate.engines import PARAKEET_FILES, ParakeetEngine, chunk_audio, find_or_download_parakeet
from dictate.recorder import SAMPLE_RATE

# -- cutting long audio into chunks -------------------------------------------------------


def test_short_audio_is_one_chunk():
    samples = np.ones(10 * SAMPLE_RATE, dtype=np.float32)
    chunks = chunk_audio(samples, max_seconds=20)
    assert len(chunks) == 1
    assert len(chunks[0]) == len(samples)


def test_chunks_are_at_most_the_limit_and_nothing_is_lost():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(95 * SAMPLE_RATE).astype(np.float32)
    chunks = chunk_audio(samples, max_seconds=20)
    assert all(len(c) <= 20 * SAMPLE_RATE for c in chunks)
    assert len(chunks) >= 5
    assert np.array_equal(np.concatenate(chunks), samples)  # every sample is in exactly one chunk


def test_the_cut_goes_at_the_quietest_moment_not_mid_word():
    rng = np.random.default_rng(1)
    samples = (rng.standard_normal(30 * SAMPLE_RATE) * 0.5).astype(np.float32)  # loud "speech"
    pause_at = int(17.3 * SAMPLE_RATE)  # inside the last 4 s before the 20 s limit
    samples[pause_at : pause_at + int(0.3 * SAMPLE_RATE)] = 0.0  # a 300 ms pause
    first = chunk_audio(samples, max_seconds=20)[0]
    assert pause_at <= len(first) <= pause_at + int(0.3 * SAMPLE_RATE)


def test_every_chunk_makes_progress_even_for_pure_silence():
    chunks = chunk_audio(np.zeros(70 * SAMPLE_RATE, dtype=np.float32), max_seconds=20)
    assert all(len(c) > 0 for c in chunks)
    assert sum(len(c) for c in chunks) == 70 * SAMPLE_RATE


# -- finding the model on disk, offline ---------------------------------------------------


def make_model_dir(root: Path, names=PARAKEET_FILES) -> Path:
    for name in names:
        (root / name).write_bytes(b"x")
    return root


def test_a_model_already_on_disk_is_used_without_any_download(tmp_path):
    folder = make_model_dir(tmp_path)
    calls = []

    def fake_download(repo, **kwargs):
        calls.append(kwargs)
        return str(folder)

    assert find_or_download_parakeet(fake_download) == folder
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True  # the only call was the offline one


def test_a_missing_model_is_downloaded_once(tmp_path):
    folder = make_model_dir(tmp_path)
    calls = []

    def fake_download(repo, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not cached")
        return str(folder)

    assert find_or_download_parakeet(fake_download) == folder
    assert [c.get("local_files_only", False) for c in calls] == [True, False]


def test_a_partial_download_is_completed(tmp_path):
    partial = make_model_dir(tmp_path, names=PARAKEET_FILES[:2])  # encoder is missing
    calls = []

    def fake_download(repo, **kwargs):
        calls.append(kwargs)
        if not kwargs.get("local_files_only"):
            make_model_dir(partial)  # the download fills in the missing files
        return str(partial)

    assert find_or_download_parakeet(fake_download) == partial
    assert [c.get("local_files_only", False) for c in calls] == [True, False]


# -- the Parakeet engine ------------------------------------------------------------------


class FakeParakeet:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def recognize(self, chunk, *, sample_rate):
        self.calls.append((len(chunk), sample_rate))
        return next(self.answers)


def engine_with(model, max_chunk_seconds=20.0) -> ParakeetEngine:
    e = ParakeetEngine(max_chunk_seconds=max_chunk_seconds)
    e._model = model
    return e


def test_short_speech_is_one_call_at_16khz():
    model = FakeParakeet(["Hello there."])
    text = engine_with(model).transcribe(np.zeros(5 * SAMPLE_RATE, dtype=np.float32))
    assert text == "Hello there."
    assert model.calls == [(5 * SAMPLE_RATE, SAMPLE_RATE)]


def test_long_speech_is_sent_in_chunks_and_the_texts_are_joined():
    model = FakeParakeet([" first ", "second", "  ", "third"])
    rng = np.random.default_rng(2)
    text = engine_with(model).transcribe(rng.standard_normal(70 * SAMPLE_RATE).astype(np.float32))
    assert len(model.calls) == 4
    assert all(n <= 20 * SAMPLE_RATE for n, _ in model.calls)
    assert text == "first second third"  # blank answers are dropped, spacing is cleaned


def test_the_engine_has_a_stable_name():
    assert ParakeetEngine.name == "parakeet-tdt-0.6b-v2"


def test_module_exposes_the_expected_files():
    assert "encoder-model.int8.onnx" in engines.PARAKEET_FILES
    with pytest.raises(ValueError):
        PARAKEET_FILES.index("encoder-model.onnx")  # the 2.4 GB full-precision file is not used
