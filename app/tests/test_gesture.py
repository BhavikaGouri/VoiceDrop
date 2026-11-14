from dictate.gesture import (
    DiscardReason,
    Emission,
    GestureMachine,
    Signal,
    State,
)

OPEN = Emission(Signal.STREAM_OPEN)
START = Emission(Signal.MIC_START)
STOP = Emission(Signal.MIC_STOP)


def discard(reason: DiscardReason) -> Emission:
    return Emission(Signal.DISCARD, reason)


def test_clean_gesture_opens_stream_starts_mic_then_stops():
    m = GestureMachine()
    assert m.key_down(0.00) == [OPEN]
    assert m.key_up(0.10) == []
    assert m.key_down(0.20) == [START]
    assert m.key_up(1.50) == [STOP]
    assert m.state is State.IDLE


def test_lone_tap_is_discarded_when_gap_expires():
    m = GestureMachine()
    m.key_down(0.00)
    m.key_up(0.10)
    deadline = m.deadline
    assert deadline == 0.10 + m.config.gap_s
    assert m.tick(deadline - 0.01) == []
    assert m.tick(deadline) == [discard(DiscardReason.NO_SECOND_PRESS)]
    assert m.state is State.IDLE
    assert m.deadline is None


def test_second_press_after_gap_expired_discards_then_starts_a_new_gesture():
    m = GestureMachine()
    m.key_down(0.00)
    m.key_up(0.10)
    assert m.key_down(0.35) == [discard(DiscardReason.NO_SECOND_PRESS), OPEN]
    assert m.state is State.TAP_DOWN


def test_first_press_held_too_long_is_not_a_tap():
    m = GestureMachine()
    m.key_down(0.00)
    assert m.tick(0.26) == [discard(DiscardReason.HELD_TOO_LONG)]
    assert m.key_up(0.50) == []
    assert m.state is State.IDLE


def test_other_key_during_first_press_discards():
    m = GestureMachine()
    m.key_down(0.00)
    assert m.other_key_down(0.05) == [discard(DiscardReason.OTHER_KEY)]
    assert m.key_up(0.10) == []
    assert m.state is State.IDLE
    assert m.deadline is None


def test_other_key_in_gap_discards():
    m = GestureMachine()
    m.key_down(0.00)
    m.key_up(0.10)
    assert m.other_key_down(0.15) == [discard(DiscardReason.OTHER_KEY)]
    assert m.state is State.IDLE
    assert m.tick(1.0) == []


def test_other_key_while_recording_cancels():
    m = GestureMachine()
    m.key_down(0.00)
    m.key_up(0.10)
    m.key_down(0.20)
    assert m.other_key_down(0.80) == [discard(DiscardReason.CANCELLED)]
    assert m.key_up(1.00) == []


def test_second_press_released_too_soon_is_discarded():
    m = GestureMachine()
    m.key_down(0.00)
    m.key_up(0.10)
    assert m.key_down(0.20) == [START]
    assert m.key_up(0.40) == [discard(DiscardReason.TOO_SHORT)]


def test_blocked_first_press_does_not_arm():
    m = GestureMachine()
    assert m.key_down(0.00, blocked=True) == []
    assert m.key_up(0.10) == []
    assert m.state is State.IDLE


def test_blocked_second_press_discards():
    m = GestureMachine()
    m.key_down(0.00)
    m.key_up(0.10)
    assert m.key_down(0.20, blocked=True) == [discard(DiscardReason.OTHER_KEY)]
    assert m.state is State.IDLE


def test_other_key_when_idle_is_ignored():
    m = GestureMachine()
    assert m.other_key_down(0.0) == []


def test_repeated_key_down_while_recording_is_ignored():
    m = GestureMachine()
    m.key_down(0.00)
    m.key_up(0.10)
    m.key_down(0.20)
    assert m.key_down(0.25) == []
    assert m.key_up(1.00) == [STOP]


def test_a_new_gesture_works_after_a_discard():
    m = GestureMachine()
    m.key_down(0.00)
    m.other_key_down(0.05)
    m.key_up(0.10)
    assert m.key_down(1.00) == [OPEN]
    m.key_up(1.10)
    assert m.key_down(1.20) == [START]
    assert m.key_up(2.50) == [STOP]
