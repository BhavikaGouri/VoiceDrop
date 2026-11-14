import threading

import numpy as np

from dictate.__main__ import DictationApp, deliver
from dictate.recorder import SAMPLE_RATE
from dictate.transcriber import Transcript


class FakeInserter:
    def __init__(self):
        self.texts: list[str] = []
        self.started = self.stopped = False

    def insert(self, text):
        self.texts.append(text)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class FakeTranscriber:
    model_name = "fake"

    def __init__(self, fail=False):
        self.fail = fail
        self.loaded = False

    def load(self):
        if self.fail:
            raise RuntimeError("model files missing")
        self.loaded = True

    def transcribe(self, recording):  # not used here
        raise AssertionError("no transcription is expected in these tests")


def transcript(text="Hello.", skipped=None):
    return Transcript(text, 2.0, 0.3, skipped, sent_seconds=1.5)


# -- what happens to a finished transcription --------------------------------------------


def test_a_transcript_is_printed_and_then_pasted(capsys):
    inserter = FakeInserter()
    deliver(transcript("Send the report."), inserter)
    assert ">>> Send the report." in capsys.readouterr().out
    assert inserter.texts == ["Send the report."]


def test_a_recording_without_speech_is_reported_but_nothing_is_pasted(capsys):
    inserter = FakeInserter()
    deliver(transcript("", skipped="no speech detected"), inserter)
    assert "no speech detected" in capsys.readouterr().out
    assert inserter.texts == []


# -- starting and stopping the app --------------------------------------------------------


def make_app(fail=False):
    inserter = FakeInserter()
    app = DictationApp(transcriber=FakeTranscriber(fail), inserter=inserter)
    # Do not touch the real microphone or install a real keyboard hook in a unit test.
    app.tasks.recorder.prepare = lambda: None
    app.tasks.recorder.close = lambda: None
    return app, inserter


def test_startup_loads_the_model_then_reports_ready(monkeypatch):
    app, inserter = make_app()
    started = threading.Event()

    class FakeListener:
        def __init__(self, handlers):
            self.stopped = False

        def start(self):
            started.set()

        def stop(self):
            self.stopped = True

    monkeypatch.setattr("dictate.__main__.HotkeyListener", FakeListener)
    app.start_in_background().join(5)
    assert app.transcriber.loaded  # the model is loaded once, up front
    assert started.is_set()  # and only then does the hotkey start listening
    assert inserter.started
    assert app.status.snapshot()[0] == "ready"
    assert not app.failed
    app.stop()
    assert app.listener.stopped
    assert inserter.stopped


def test_a_startup_failure_is_shown_in_the_window_and_stops_the_app():
    app, _ = make_app(fail=True)
    app.start_in_background().join(5)
    state, message = app.status.snapshot()
    assert state == "failed"
    assert "model files missing" in message
    assert app.failed
    app.stop()  # must not raise even though nothing finished starting


def test_stopping_twice_is_harmless():
    app, inserter = make_app()
    app.stop()
    app.stop()
    assert inserter.stopped


def test_the_hotkey_tasks_are_wired_to_the_recorder_and_worker():
    app, _ = make_app()
    submitted = []
    app.worker.submit = submitted.append
    rec = np.zeros(SAMPLE_RATE, dtype=np.float32)
    from dictate.recorder import Recording

    app.tasks.recorder.stop = lambda: Recording(rec, SAMPLE_RATE, False)
    app.tasks.stop_mic()
    assert len(submitted) == 1  # a released dictation is queued for transcription
