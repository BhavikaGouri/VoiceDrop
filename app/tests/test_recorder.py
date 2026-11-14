import threading

import numpy as np
import pytest

from dictate.recorder import SAMPLE_RATE, Recorder, RecorderError


class FakeStream:
    """Stands in for the WASAPI stream. Tests push audio through `feed`."""

    def __init__(self, callback):
        self.callback = callback
        self.running = False
        self.starts = 0
        self.closed = False

    def start(self):
        self.running = True
        self.starts += 1

    def stop(self):
        self.running = False

    def close(self):
        self.closed = True

    def feed(self, value: float, frames: int = 1600, status=None):
        block = np.full((frames, 1), value, dtype=np.float32)
        self.callback(block, frames, None, status)


def make(**kwargs):
    streams: list[FakeStream] = []

    def factory(callback):
        stream = FakeStream(callback)
        streams.append(stream)
        return stream

    return Recorder(stream_factory=factory, **kwargs), streams


# -- what gets recorded -------------------------------------------------------------------


def test_audio_from_the_first_tap_is_kept_in_the_recording():
    rec, streams = make()
    rec.open()
    streams[0].feed(0.1)  # buffered before the second press: the first syllable
    assert rec.confirm()
    streams[0].feed(0.2)
    result = rec.stop()
    assert result is not None
    assert len(result.samples) == 3200
    assert result.samples[0] == pytest.approx(0.1)
    assert result.samples[-1] == pytest.approx(0.2)
    assert result.sample_rate == SAMPLE_RATE
    assert result.samples.dtype == np.float32
    assert not result.truncated
    assert not streams[0].running


def test_each_block_is_copied():
    rec, streams = make()
    rec.open()
    rec.confirm()
    block = np.full((1600, 1), 0.4, dtype=np.float32)
    streams[0].callback(block, 1600, None, None)
    block[:] = 0.9  # PortAudio reuses its buffer after the callback returns
    assert rec.stop().samples[0] == pytest.approx(0.4)


# -- discarding ---------------------------------------------------------------------------


def test_discard_drops_the_buffer_and_stops_the_stream():
    rec, streams = make()
    rec.open()
    streams[0].feed(0.5)
    rec.discard()
    assert not streams[0].running
    assert rec.stop() is None  # nothing left to return


def test_audio_after_discard_is_ignored_and_the_next_recording_is_clean():
    rec, streams = make()
    rec.open()
    rec.discard()
    streams[0].feed(0.5)  # a late callback from the stopped stream
    rec.open()
    streams[0].feed(0.3)
    rec.confirm()
    result = rec.stop()
    assert len(result.samples) == 1600
    assert result.samples[0] == pytest.approx(0.3)


def test_opening_again_drops_the_previous_recording():
    rec, streams = make()
    rec.open()
    streams[0].feed(0.9)
    rec.open()
    streams[0].feed(0.1)
    rec.confirm()
    assert len(rec.stop().samples) == 1600


def test_confirm_without_open_is_ignored():
    rec, _ = make()
    assert rec.confirm() is False


def test_stop_without_confirm_discards():
    rec, streams = make()
    rec.open()
    streams[0].feed(0.5)
    assert rec.stop() is None
    assert not streams[0].running


# -- the stream's lifetime ----------------------------------------------------------------


def test_prepare_creates_the_stream_without_starting_it():
    rec, streams = make()
    rec.prepare()
    assert len(streams) == 1
    assert streams[0].starts == 0


def test_prepared_stream_is_reused_by_every_dictation():
    rec, streams = make()
    rec.prepare()
    for _ in range(3):
        rec.open()
        streams[0].feed(0.1)
        rec.confirm()
        assert rec.stop() is not None
    assert len(streams) == 1  # created once
    assert streams[0].starts == 3  # started once per dictation
    assert not streams[0].closed


def test_without_keep_stream_a_new_stream_is_made_each_time():
    rec, streams = make(keep_stream=False)
    rec.open()
    rec.discard()
    assert streams[0].closed
    rec.open()
    assert len(streams) == 2


def test_close_releases_the_stream():
    rec, streams = make()
    rec.prepare()
    rec.open()
    rec.close()
    assert streams[0].closed
    assert not streams[0].running


def test_failure_to_create_raises_recorder_error_and_leaves_it_idle():
    def broken(callback):
        raise RecorderError("no microphone found")

    rec = Recorder(stream_factory=broken)
    with pytest.raises(RecorderError):
        rec.open()
    assert rec.confirm() is False
    assert rec.stop() is None


def test_failure_to_start_is_wrapped_and_the_next_tap_builds_a_fresh_stream():
    class FlakyStream(FakeStream):
        broken = True

        def start(self):
            if FlakyStream.broken:
                raise OSError("device busy")
            super().start()

    made: list[FlakyStream] = []

    def factory(callback):
        made.append(FlakyStream(callback))
        return made[-1]

    rec = Recorder(stream_factory=factory)
    with pytest.raises(RecorderError, match="device busy"):
        rec.open()
    assert made[0].closed  # the broken stream is thrown out

    FlakyStream.broken = False  # microphone back
    rec.open()
    assert len(made) == 2
    assert made[1].running


# -- the 5 minute limit -------------------------------------------------------------------


def test_max_length_truncates_and_fires_on_limit_once():
    fired = threading.Event()
    rec, streams = make(max_seconds=0.2, on_limit=fired.set)  # 3200 frames
    rec.open()
    rec.confirm()
    streams[0].feed(0.1, frames=1600)
    assert not fired.wait(0.05)
    streams[0].feed(0.1, frames=1600)
    assert fired.wait(1)
    streams[0].feed(0.1, frames=1600)  # past the limit: dropped
    result = rec.stop()
    assert len(result.samples) == 3200
    assert result.truncated


def test_limit_does_not_fire_before_the_second_press():
    fired = threading.Event()
    rec, streams = make(max_seconds=0.1, on_limit=fired.set)
    rec.open()
    streams[0].feed(0.1, frames=3200)
    assert not fired.wait(0.1)


def test_stop_after_the_limit_handler_has_stopped_returns_none():
    rec, streams = make(max_seconds=0.1)
    rec.open()
    rec.confirm()
    streams[0].feed(0.1, frames=3200)
    assert rec.stop() is not None  # what the limit handler does
    assert rec.stop() is None  # the later key release finds nothing
