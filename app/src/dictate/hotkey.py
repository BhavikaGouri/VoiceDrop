"""The Right Alt hotkey: tap it, then press it again and hold.

Requirements
1. Tap Right Alt            -> heads-up: the audio stream opens and starts buffering.
2. Press it again, hold     -> the dictation really starts; releasing it ends it.
3. The first tap was a mistake, and is discarded, when:
     - Right Alt was used in a combination with any other key (Alt+Tab, Ctrl+Alt, ...), or
     - no second press follows within the gap window, or
     - the first press was held too long to be a tap.
The decision logic lives in gesture.py. This module feeds it real key events and timers and
delivers the resulting tasks to ``Handlers``.

Threads
- hook thread:       receives key events, must return fast (see winkeys.KeyboardHook)
- timer thread:      fires the gesture's tap and gap deadlines
- dispatcher thread: runs the handlers, so slow work in a handler never stalls typing
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType

from dictate import winkeys
from dictate.gesture import DiscardReason, Emission, GestureConfig, GestureMachine, Signal

log = logging.getLogger(__name__)


def _ignore() -> None:
    pass


def _ignore_discard(reason: DiscardReason) -> None:
    pass


@dataclass
class Handlers:
    """One callback per task. They run on the dispatcher thread, one at a time, in order."""

    on_stream_open: Callable[[], None] = _ignore  # first tap: open the stream, start buffering
    on_mic_start: Callable[[], None] = _ignore  # second press held: speech starts here
    on_mic_stop: Callable[[], None] = _ignore  # released after a real hold: transcribe
    on_discard: Callable[[DiscardReason], None] = _ignore_discard  # mistake: drop and close


class HotkeyListener:
    def __init__(
        self,
        handlers: Handlers,
        config: GestureConfig | None = None,
        *,
        accept_injected: bool = False,
    ) -> None:
        self._handlers = handlers
        self._accept_injected = accept_injected  # tests only: let synthetic keys through
        self._machine = GestureMachine(config)

        # _lock guards the machine, which the hook thread and the timer thread both drive.
        # It doubles as the timer thread's wake-up signal when the machine's deadline changes.
        self._lock = threading.Condition()
        self._stopping = False
        self._tasks: queue.SimpleQueue[Emission | None] = queue.SimpleQueue()

        self._hook = winkeys.KeyboardHook(self._on_key)
        self._timer = threading.Thread(target=self._run_timer, name="gesture-timer", daemon=True)
        self._dispatcher = threading.Thread(
            target=self._run_dispatcher, name="gesture-dispatch", daemon=True
        )

        # Hook-thread state: what the keyboard is doing around Right Alt.
        self._others_down: set[int] = set()  # non-Right-Alt keys currently held
        self._alt_down = False
        self._combo = False  # Right Alt has been used together with another key

    def start(self) -> None:
        self._timer.start()
        self._dispatcher.start()
        self._hook.start()

    def stop(self) -> None:
        self._hook.stop()
        with self._lock:
            self._stopping = True
            self._lock.notify_all()
        self._timer.join(timeout=2)
        self._tasks.put(None)  # tells the dispatcher to finish
        self._dispatcher.join(timeout=2)

    def __enter__(self) -> HotkeyListener:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # -- hook thread: translate raw key events into gesture inputs --------------------------

    def _on_key(self, event: winkeys.KeyEvent) -> None:
        # Our own dummy key and other software's synthetic input are not the user's gesture.
        if event.vk == winkeys.VK_DUMMY or (event.injected and not self._accept_injected):
            return
        if event.vk == winkeys.VK_RMENU:
            self._on_right_alt(event)
        else:
            self._on_other_key(event)

    def _on_right_alt(self, event: winkeys.KeyEvent) -> None:
        now = time.monotonic()
        if event.is_down:
            if self._alt_down:
                return  # keyboard auto-repeat while held
            self._alt_down = True
            # Another key already held (Ctrl+Alt, Shift+Alt, ...) means this is a shortcut, so
            # the machine will not start a gesture. Remember that for the key-up below.
            self._combo = self._other_keys_held()
            self._feed(self._machine.key_down, now, blocked=self._combo)
            return

        if not self._alt_down:
            return
        self._alt_down = False
        if not self._combo:
            # A clean Right Alt press: stop Windows opening the app's menu bar on release.
            # Skipped for combos so Alt+Tab, Alt+F4 and friends behave exactly as usual.
            winkeys.send_dummy_key()
        self._feed(self._machine.key_up, now)

    def _on_other_key(self, event: winkeys.KeyEvent) -> None:
        if not event.is_down:
            self._others_down.discard(event.vk)
            return
        self._others_down.add(event.vk)
        if self._alt_down:
            self._combo = True
        # Any key during the gesture means Right Alt was part of a shortcut: first tap discarded.
        self._feed(self._machine.other_key_down, time.monotonic())

    def _other_keys_held(self) -> bool:
        # Forget keys whose release we missed (lock screen, another desktop, hook restart),
        # otherwise a stuck entry would block the hotkey until restart.
        self._others_down = {vk for vk in self._others_down if winkeys.is_key_down(vk)}
        return bool(self._others_down)

    # -- shared: run a machine step and queue its tasks -------------------------------------

    def _feed(self, step: Callable[..., list[Emission]], *args: float, **kwargs: bool) -> None:
        with self._lock:
            self._queue(step(*args, **kwargs))
            self._lock.notify()  # the deadline may have changed: wake the timer thread

    def _queue(self, emissions: list[Emission]) -> None:
        # Always called with _lock held, so tasks reach the dispatcher in the order they happened.
        for emission in emissions:
            self._tasks.put(emission)

    # -- timer thread: tap and gap deadlines ------------------------------------------------

    def _run_timer(self) -> None:
        with self._lock:
            while not self._stopping:
                deadline = self._machine.deadline
                if deadline is None:
                    self._lock.wait()  # nothing pending
                    continue
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    self._lock.wait(remaining)  # woken early if the deadline changes
                    continue
                # Deadline passed with no key event: e.g. the tap was never followed by a press.
                self._queue(self._machine.tick(time.monotonic()))

    # -- dispatcher thread: run the handlers ------------------------------------------------

    def _run_dispatcher(self) -> None:
        while (emission := self._tasks.get()) is not None:
            try:
                self._dispatch(emission)
            except Exception:
                log.exception("handler failed for %s", emission)

    def _dispatch(self, emission: Emission) -> None:
        match emission.signal:
            case Signal.STREAM_OPEN:
                self._handlers.on_stream_open()
            case Signal.MIC_START:
                self._handlers.on_mic_start()
            case Signal.MIC_STOP:
                self._handlers.on_mic_stop()
            case Signal.DISCARD if emission.reason is not None:
                self._handlers.on_discard(emission.reason)
