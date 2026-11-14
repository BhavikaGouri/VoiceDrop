import threading
from types import SimpleNamespace

import numpy as np
import pytest

from dictate.recorder import SAMPLE_RATE, Recording
from dictate.transcriber import (
    MIN_SPEECH_SECONDS,
    SPEECH_PAD_SECONDS,
    Transcriber,
    TranscriptionWorker,
    trim_to_speech,
)

PAD = int(SPEECH_PAD_SECONDS * SAMPLE_RATE)  # padding in samples


def recording(seconds: float = 1.0, value: float = 0.1) -> Recording:
    samples = np.full(int(seconds * SAMPLE_RATE), value, dtype=np.float32)
    return Recording(samples=samples, sample_rate=SAMPLE_RATE, truncated=False)


class FakeEngine:
    """Stands in for a speech model and records what it was asked to transcribe."""

    name = "fake-engine"

    def __init__(self, text="Hello there."):
        self.text = text
        self.loaded = 0
        self.calls: list[np.ndarray] = []

    def load(self):
        self.loaded += 1

    def transcribe(self, samples):
        self.calls.append(samples)
        return self.text


def make(engine: FakeEngine, spans: list[tuple[int, int]] | None = None) -> Transcriber:
    """A transcriber whose speech detector reports `spans` (default: the whole recording)."""

    def finder(samples):
        return [(0, len(samples))] if spans is None else spans

    t = Transcriber(engine, speech_finder=finder)
    t.load()
    return t


def test_load_loads_the_engine_once_and_exposes_its_name():
    engine = FakeEngine()
    t = make(engine)
    assert engine.loaded == 1
    assert t.model_name == "fake-engine"


def test_transcribe_before_load_raises():
    t = Transcriber(FakeEngine())
    with pytest.raises(RuntimeError, match="load"):
        t.transcribe(recording())


# -- trimming silence ---------------------------------------------------------------------


def ramp(n: int) -> np.ndarray:
    """Distinct values per sample, so tests can tell exactly which samples were kept."""
    return np.arange(n, dtype=np.float32)


def test_trim_keeps_padding_around_speech_and_drops_the_rest():
    samples = ramp(10 * SAMPLE_RATE)
    start, end = 4 * SAMPLE_RATE, 5 * SAMPLE_RATE
    out = trim_to_speech(samples, [(start, end)])
    assert np.array_equal(out, samples[start - PAD : end + PAD])


def test_trim_never_cuts_the_start_when_speech_begins_near_the_first_tap():
    # Speech 50 ms into the recording (the first tap to speech gap of a fast speaker): the
    # padding reaches back past the start, so everything before the speech is kept.
    samples = ramp(3 * SAMPLE_RATE)
    start = int(0.05 * SAMPLE_RATE)
    out = trim_to_speech(samples, [(start, 2 * SAMPLE_RATE)])
    assert out[0] == 0  # the very first sample is still there
    assert np.array_equal(out[: start + 10], samples[: start + 10])


def test_trim_only_removes_the_leading_and_trailing_silence():
    samples = ramp(20 * SAMPLE_RATE)
    first = (2 * SAMPLE_RATE, 3 * SAMPLE_RATE)
    second = (12 * SAMPLE_RATE, 13 * SAMPLE_RATE)
    out = trim_to_speech(samples, [first, second])
    # From the padding before the first speech to the padding after the last: one unbroken piece.
    assert np.array_equal(out, samples[first[0] - PAD : second[1] + PAD])


def test_trim_keeps_a_long_silence_or_noise_between_two_pieces_of_speech():
    samples = ramp(20 * SAMPLE_RATE)
    first = (2 * SAMPLE_RATE, 3 * SAMPLE_RATE)
    second = (12 * SAMPLE_RATE, 13 * SAMPLE_RATE)
    out = trim_to_speech(samples, [first, second])
    gap = samples[3 * SAMPLE_RATE : 12 * SAMPLE_RATE]  # the 9 s in the middle
    start = 3 * SAMPLE_RATE - (first[0] - PAD)
    assert np.array_equal(out[start : start + len(gap)], gap)  # still there, sample for sample


def test_trim_keeps_a_short_pause_inside_a_sentence_whole():
    samples = ramp(10 * SAMPLE_RATE)
    first, second = (SAMPLE_RATE, 2 * SAMPLE_RATE), (int(2.5 * SAMPLE_RATE), 4 * SAMPLE_RATE)
    out = trim_to_speech(samples, [first, second])
    assert np.array_equal(out, samples[SAMPLE_RATE - PAD : 4 * SAMPLE_RATE + PAD])


def test_trim_with_no_speech_returns_nothing():
    assert len(trim_to_speech(ramp(SAMPLE_RATE), [])) == 0


def test_only_the_speech_reaches_the_engine_not_the_noise_around_it():
    engine = FakeEngine()
    rec = recording(10)
    rec.samples[:] = ramp(len(rec.samples))
    start, end = 4 * SAMPLE_RATE, 5 * SAMPLE_RATE  # one second of speech in ten
    result = make(engine, spans=[(start, end)]).transcribe(rec)
    assert np.array_equal(engine.calls[0], rec.samples[start - PAD : end + PAD])
    assert result.audio_seconds == pytest.approx(10.0)
    assert result.sent_seconds == pytest.approx(1 + 2 * SPEECH_PAD_SECONDS)


def test_the_engine_text_is_returned():
    result = make(FakeEngine("Hello, this is a test.")).transcribe(recording())
    assert result.text == "Hello, this is a test."
    assert result.skipped is None
    assert result.audio_seconds == pytest.approx(1.0)


def test_a_recording_without_speech_is_skipped_and_the_engine_is_not_run():
    engine = FakeEngine()
    result = make(engine, spans=[]).transcribe(recording())
    assert result.text == ""
    assert result.skipped == "no speech detected"
    assert engine.calls == []


def test_too_little_speech_is_skipped():
    engine = FakeEngine()
    short = int((MIN_SPEECH_SECONDS - 0.01) * SAMPLE_RATE)
    result = make(engine, spans=[(8000, 8000 + short)]).transcribe(recording(2))
    assert result.skipped == "no speech detected"
    assert engine.calls == []


def test_just_enough_speech_is_transcribed():
    engine = FakeEngine()
    enough = int(MIN_SPEECH_SECONDS * SAMPLE_RATE)
    result = make(engine, spans=[(8000, 8000 + enough)]).transcribe(recording(2))
    assert result.text == "Hello there."
    assert len(engine.calls) == 1


# -- the worker ---------------------------------------------------------------------------


def test_worker_transcribes_queued_recordings_in_order():
    results: list[str] = []
    done = threading.Event()
    texts = iter(["first", "second", "third"])

    def fake_transcribe(rec):
        return SimpleNamespace(text=next(texts), skipped=None)

    t = make(FakeEngine())
    t.transcribe = fake_transcribe  # type: ignore[method-assign]

    def on_transcript(tr):
        results.append(tr.text)
        if len(results) == 3:
            done.set()

    worker = TranscriptionWorker(t, on_transcript)
    worker.start()
    for _ in range(3):
        worker.submit(recording())
    assert done.wait(2)
    assert results == ["first", "second", "third"]
    worker.stop()


def test_worker_reports_busy_until_everything_is_done():
    release = threading.Event()
    started = threading.Event()
    t = make(FakeEngine())

    def slow_transcribe(rec):
        started.set()
        release.wait(2)
        return SimpleNamespace(text="x", skipped=None)

    t.transcribe = slow_transcribe  # type: ignore[method-assign]
    worker = TranscriptionWorker(t, lambda tr: None)
    worker.start()
    assert not worker.busy
    worker.submit(recording())
    worker.submit(recording())
    assert started.wait(2)
    assert worker.busy
    release.set()
    worker.stop()
    assert not worker.busy


def test_stopping_a_worker_that_never_started_is_harmless():
    worker = TranscriptionWorker(make(FakeEngine()), lambda tr: None)
    worker.stop()  # startup failed before the worker was started: must not raise
    assert not worker.busy


def test_worker_survives_a_failing_transcription():
    results: list[str] = []
    done = threading.Event()
    t = make(FakeEngine())
    calls = iter([RuntimeError("boom"), SimpleNamespace(text="ok", skipped=None)])

    def flaky(rec):
        outcome = next(calls)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    t.transcribe = flaky  # type: ignore[method-assign]

    def on_transcript(tr):
        results.append(tr.text)
        done.set()

    worker = TranscriptionWorker(t, on_transcript)
    worker.start()
    worker.submit(recording())
    worker.submit(recording())
    assert done.wait(2)
    assert results == ["ok"]
    worker.stop()
