import pytest

from dictate.clipboard import ClipboardError, ClipboardSnapshot
from dictate.inserter import TextInserter

BEFORE = ClipboardSnapshot(formats=((13, b"old\0"), (49161, b"<b>old</b>")), skipped=())


class FakeClipboard:
    """Records every call, in order, and behaves like a clipboard with a sequence number."""

    def __init__(self, calls, before=BEFORE):
        self.calls = calls
        self.before = before
        self.sequence = 100
        self.on_put = None  # a callable run right after our text is set, to simulate a rival
        self.fail_snapshot = False
        self.fail_put = False

    def snapshot(self):
        self.calls.append("snapshot")
        if self.fail_snapshot:
            raise ClipboardError("busy")
        return self.before

    def put_text(self, text):
        self.calls.append(f"put:{text}")
        if self.fail_put:
            raise ClipboardError("busy")
        self.sequence += 1
        return self.sequence

    def restore(self, snapshot):
        self.calls.append("restore")
        self.sequence += 1

    def sequence_number(self):
        return self.sequence

    def close(self):
        self.calls.append("close")


def run(texts, *, clipboard_setup=None, held=None, notices=None, wait=0.05, rival=False):
    """Run an inserter over `texts` with fakes. Returns (calls, notices, clipboard)."""
    calls: list[str] = []
    notes: list[str] = notices if notices is not None else []
    clipboard = FakeClipboard(calls)
    if clipboard_setup:
        clipboard_setup(clipboard)

    def send_paste():
        calls.append("ctrl+v")
        if rival:
            clipboard.sequence += 1  # somebody else wrote to the clipboard right after the paste

    ins = TextInserter(
        on_notice=notes.append,
        clipboard_factory=lambda: clipboard,
        send_paste=send_paste,
        get_held_modifiers=held or (lambda: []),
        settle_seconds=0.0,
        modifier_wait_seconds=wait,
    )
    ins.start()
    for t in texts:
        ins.insert(t)
    ins.stop()
    return calls, notes, clipboard, ins


def test_the_five_steps_happen_in_order():
    calls, notes, _, ins = run(["Hello there."])
    assert calls == ["snapshot", "put:Hello there.", "ctrl+v", "restore", "close"]
    assert notes == []
    assert ins.last_text == "Hello there."  # step 1


def test_the_previous_clipboard_is_not_restored_if_something_else_wrote_to_it():
    calls, notes, *_ = run(["Hello."], rival=True)
    assert "ctrl+v" in calls
    assert "restore" not in calls  # the newer content is left alone
    assert notes == []


def test_pastes_run_one_at_a_time_in_order():
    calls, *_ = run(["first", "second"])
    puts = [c for c in calls if c.startswith("put:")]
    assert puts == ["put:first", "put:second"]
    # each paste is complete before the next one starts
    assert calls[:4] == ["snapshot", "put:first", "ctrl+v", "restore"]
    assert calls[4:8] == ["snapshot", "put:second", "ctrl+v", "restore"]


def test_empty_text_does_nothing():
    calls, notes, _, ins = run(["", "   "])
    assert calls == ["close"]
    assert ins.last_text == ""


def test_whitespace_around_the_text_is_trimmed():
    calls, *_ = run(["  hello  "])
    assert "put:hello" in calls


def test_waits_for_modifier_keys_then_pastes():
    states = iter([["Alt"], ["Alt"], []])
    calls, notes, *_ = run(["Hi."], held=lambda: next(states), wait=2.0)
    assert "ctrl+v" in calls
    assert notes == []


def test_a_modifier_that_never_comes_up_leaves_the_text_on_the_clipboard():
    calls, notes, *_ = run(["Hi."], held=lambda: ["Shift"], wait=0.05)
    assert "ctrl+v" not in calls
    assert "snapshot" not in calls  # nothing was touched except putting our text there
    assert calls[:1] == ["put:Hi."]
    assert "restore" not in calls
    assert len(notes) == 1 and "Shift" in notes[0] and "Ctrl+V" in notes[0]


def test_an_unreadable_clipboard_means_nothing_is_pasted_or_changed():
    calls, notes, *_ = run(["Hi."], clipboard_setup=lambda c: setattr(c, "fail_snapshot", True))
    assert calls == ["snapshot", "close"]
    assert len(notes) == 1 and "could not be read" in notes[0]


def test_a_failed_write_restores_what_was_there_and_does_not_paste():
    calls, notes, *_ = run(["Hi."], clipboard_setup=lambda c: setattr(c, "fail_put", True))
    assert "ctrl+v" not in calls
    assert calls.count("restore") == 1
    assert len(notes) == 1 and "could not be written" in notes[0]


def test_a_failing_paste_does_not_stop_later_ones():
    calls: list[str] = []
    notes: list[str] = []
    clipboard = FakeClipboard(calls)
    sent = []

    def flaky_paste():
        sent.append(1)
        if len(sent) == 1:
            raise OSError("boom")

    ins = TextInserter(
        on_notice=notes.append,
        clipboard_factory=lambda: clipboard,
        send_paste=flaky_paste,
        get_held_modifiers=lambda: [],
        settle_seconds=0.0,
    )
    ins.start()
    ins.insert("one")
    ins.insert("two")
    ins.stop()
    assert len(sent) == 2
    assert any("failed" in n for n in notes)


def test_stopping_an_inserter_that_never_started_is_harmless():
    TextInserter(clipboard_factory=lambda: FakeClipboard([])).stop()


def test_snapshot_reports_when_formats_were_left_out():
    complete = ClipboardSnapshot(formats=((13, b"x"),), skipped=())
    assert complete.complete
    only_bitmap_lost = ClipboardSnapshot(formats=((8, b"dib"),), skipped=(2,))
    assert only_bitmap_lost.complete  # Windows rebuilds the bitmap from the DIB
    really_lost = ClipboardSnapshot(formats=((13, b"x"),), skipped=(14,))
    assert not really_lost.complete


@pytest.mark.parametrize("held", [["Alt"], ["Shift", "Win"]])
def test_every_held_modifier_is_named_in_the_notice(held):
    _, notes, *_ = run(["Hi."], held=lambda: held, wait=0.02)
    assert all(name in notes[0] for name in held)
