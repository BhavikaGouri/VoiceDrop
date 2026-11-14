"""Pure state machine for the tap-then-hold Right Alt gesture.

Gesture: tap Right Alt (heads-up: the audio stream opens), then press it again
within ``gap_s`` and hold (the mic starts). Release stops the recording. Anything
else discards what was captured.

No I/O and no clock of its own: every method takes ``now`` (seconds, any
monotonic origin) and returns the signals to emit, so it is trivial to test.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class State(enum.Enum):
    IDLE = "idle"
    TAP_DOWN = "tap_down"  # first press is down, stream is open
    GAP = "gap"  # first press released, waiting for the second press
    RECORDING = "recording"  # second press is held, mic is running


class Signal(enum.Enum):
    STREAM_OPEN = "stream_open"  # first press: open the stream and start buffering
    MIC_START = "mic_start"  # second press held: the speech starts here
    MIC_STOP = "mic_stop"  # released after a real hold: stop and transcribe
    DISCARD = "discard"  # drop the buffer and close the stream


class DiscardReason(enum.Enum):
    NO_SECOND_PRESS = "no_second_press"  # gap window expired after the first tap
    HELD_TOO_LONG = "held_too_long"  # first press was a hold, not a tap
    OTHER_KEY = "other_key"  # a shortcut, not a gesture
    TOO_SHORT = "too_short"  # second press released before min_hold_s
    CANCELLED = "cancelled"  # a key was pressed while recording


@dataclass(frozen=True)
class Emission:
    signal: Signal
    reason: DiscardReason | None = None


@dataclass(frozen=True)
class GestureConfig:
    tap_max_s: float = 0.25  # longest a first press may be held and still count as a tap
    gap_s: float = 0.20  # time allowed between first release and second press
    min_hold_s: float = 0.30  # shortest second press that counts as a recording


def _discard(reason: DiscardReason) -> Emission:
    return Emission(Signal.DISCARD, reason)


class GestureMachine:
    def __init__(self, config: GestureConfig | None = None) -> None:
        self.config = config or GestureConfig()
        self.state = State.IDLE
        self._deadline: float | None = None
        self._hold_started = 0.0

    @property
    def deadline(self) -> float | None:
        """When ``tick`` next needs to run, or None if no timer is pending."""
        return self._deadline

    def tick(self, now: float) -> list[Emission]:
        """Apply any expired timer. Every other method calls this first."""
        if self._deadline is None or now < self._deadline:
            return []
        self._deadline = None
        if self.state is State.TAP_DOWN:
            self.state = State.IDLE
            return [_discard(DiscardReason.HELD_TOO_LONG)]
        if self.state is State.GAP:
            self.state = State.IDLE
            return [_discard(DiscardReason.NO_SECOND_PRESS)]
        return []

    def key_down(self, now: float, *, blocked: bool = False) -> list[Emission]:
        """Right Alt went down. ``blocked``: another key is already held (a shortcut)."""
        out = self.tick(now)
        if self.state is State.IDLE:
            if blocked:
                return out
            self.state = State.TAP_DOWN
            self._deadline = now + self.config.tap_max_s
            out.append(Emission(Signal.STREAM_OPEN))
        elif self.state is State.GAP:
            self._deadline = None
            if blocked:
                self.state = State.IDLE
                out.append(_discard(DiscardReason.OTHER_KEY))
            else:
                self.state = State.RECORDING
                self._hold_started = now
                out.append(Emission(Signal.MIC_START))
        return out

    def key_up(self, now: float) -> list[Emission]:
        """Right Alt went up."""
        out = self.tick(now)
        if self.state is State.TAP_DOWN:
            self.state = State.GAP
            self._deadline = now + self.config.gap_s
        elif self.state is State.RECORDING:
            self.state = State.IDLE
            if now - self._hold_started < self.config.min_hold_s:
                out.append(_discard(DiscardReason.TOO_SHORT))
            else:
                out.append(Emission(Signal.MIC_STOP))
        return out

    def other_key_down(self, now: float) -> list[Emission]:
        """Any key other than Right Alt went down."""
        out = self.tick(now)
        if self.state in (State.TAP_DOWN, State.GAP):
            self.state = State.IDLE
            self._deadline = None
            out.append(_discard(DiscardReason.OTHER_KEY))
        elif self.state is State.RECORDING:
            self.state = State.IDLE
            out.append(_discard(DiscardReason.CANCELLED))
        return out
