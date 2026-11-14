"""Text insertion: paste a dictation wherever the cursor is, in whatever app has focus.

The workflow for each piece of text
-----------------------------------
  1. Save the transcription ... kept in ``last_text``, so it is never lost even if pasting fails.
  2. Save the clipboard ....... everything currently on it is copied, not only text.
  3. Copy the transcription ... the text goes on the clipboard (kept out of Win+V history).
  4. Simulate Ctrl+V .......... the app with the cursor reads the clipboard and inserts the text.
  5. Restore the clipboard .... only when it is safe: that is, the clipboard still holds exactly
                                what we put there. If anything else has written to it in the
                                meantime (you copied something, an app set it), your newer copy
                                wins and nothing is restored over it.

Safety rules around those steps
  - Ctrl+V is only sent when Shift, Alt and Win are all released. Otherwise it would arrive as
    Alt+Ctrl+V or Win+V. If you are still holding Right Alt for the next dictation, the paste
    waits for you to let go.
  - If the paste cannot be done safely (a key is still held after the wait, or the clipboard is
    busy), the text is NOT lost: it is left on the clipboard so you can press Ctrl+V yourself,
    and you are told.
  - Pastes run one at a time, in the order the dictations were spoken, on their own thread, so
    neither the hotkey nor the transcription ever waits for the clipboard.

There is no check that a text field is focused: Ctrl+V goes to whatever has focus. Waiting
until a text field appears is a later step.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Protocol

from dictate import winkeys
from dictate.clipboard import Clipboard, ClipboardError, ClipboardSnapshot

log = logging.getLogger(__name__)

SETTLE_SECONDS = 0.35  # how long the target app gets to read the clipboard before we restore it
MODIFIER_WAIT_SECONDS = 10.0  # how long to wait for Shift, Alt and Win to be released

# Virtual-key codes of the keys that would change what Ctrl+V does.
_MODIFIER_KEYS = {"Shift": 0x10, "Alt": 0x12, "Win": 0x5B}  # 0x12 covers both Alt keys
_RIGHT_WIN = 0x5C


class ClipboardLike(Protocol):
    def snapshot(self) -> ClipboardSnapshot: ...
    def put_text(self, text: str) -> int: ...
    def restore(self, snapshot: ClipboardSnapshot) -> None: ...
    def sequence_number(self) -> int: ...
    def close(self) -> None: ...


def held_modifiers() -> list[str]:
    """Names of the modifier keys that are physically held down right now."""
    held = [name for name, vk in _MODIFIER_KEYS.items() if winkeys.is_key_down(vk)]
    if "Win" not in held and winkeys.is_key_down(_RIGHT_WIN):
        held.append("Win")
    return held


class TextInserter:
    def __init__(
        self,
        *,
        on_notice: Callable[[str], None] = lambda message: None,
        clipboard_factory: Callable[[], ClipboardLike] = Clipboard,
        send_paste: Callable[[], None] = winkeys.send_ctrl_v,
        get_held_modifiers: Callable[[], list[str]] = held_modifiers,
        settle_seconds: float = SETTLE_SECONDS,
        modifier_wait_seconds: float = MODIFIER_WAIT_SECONDS,
    ) -> None:
        self._on_notice = on_notice  # told when something could not be done normally
        self._clipboard_factory = clipboard_factory
        self._send_paste = send_paste
        self._held_modifiers = get_held_modifiers
        self._settle_seconds = settle_seconds
        self._modifier_wait_seconds = modifier_wait_seconds
        self._texts: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name="inserter", daemon=True)
        self.last_text = ""  # step 1: the most recent transcription, kept whatever happens next

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 15.0) -> None:
        """Finish the pastes already queued, then end the thread."""
        if not self._thread.is_alive():  # never started (startup failed) or already stopped
            return
        self._texts.put(None)
        self._thread.join(timeout)

    def insert(self, text: str) -> None:
        """Queue text for pasting and return at once."""
        text = text.strip()
        if text:
            self.last_text = text  # Step 1: save the transcription result
            self._texts.put(text)

    # -- the inserter thread -------------------------------------------------------------

    def _run(self) -> None:
        # The clipboard helper window must be created and used on this one thread.
        clipboard = self._clipboard_factory()
        try:
            while (text := self._texts.get()) is not None:
                try:
                    self._paste(clipboard, text)
                except Exception:
                    log.exception("pasting failed")
                    self._on_notice("pasting failed unexpectedly; the text is in the terminal")
        finally:
            clipboard.close()

    def _wait_for_modifiers_released(self) -> list[str]:
        """Wait for Shift, Alt and Win to be released. Returns whatever is still held at the end."""
        deadline = time.monotonic() + self._modifier_wait_seconds
        held = self._held_modifiers()
        while held and time.monotonic() < deadline:
            time.sleep(0.02)
            held = self._held_modifiers()
        return held

    def _paste(self, clipboard: ClipboardLike, text: str) -> None:
        # Safety: Ctrl+V must arrive as plain Ctrl+V.
        held = self._wait_for_modifiers_released()
        if held:
            self._leave_on_clipboard(clipboard, text, f"{' and '.join(held)} was still held down")
            return

        # Step 2: save the current clipboard contents.
        try:
            before = clipboard.snapshot()
        except ClipboardError as exc:
            # Without a copy we could not give the clipboard back, so do not touch it.
            self._on_notice(f"not pasted, the clipboard could not be read ({exc})")
            return
        if not before.complete:
            log.warning("clipboard formats not copied (will not be restored): %s", before.skipped)

        # Step 3: copy the transcription to the clipboard.
        try:
            ours = clipboard.put_text(text)
        except ClipboardError as exc:
            self._on_notice(f"not pasted, the clipboard could not be written ({exc})")
            self._try_restore(clipboard, before)  # EmptyClipboard may already have run
            return

        # Step 4: simulate Ctrl+V.
        self._send_paste()

        # Step 5: restore the previous clipboard contents, when it is safe. Give the app a
        # moment to read what we put there first.
        time.sleep(self._settle_seconds)
        if clipboard.sequence_number() == ours:
            self._try_restore(clipboard, before)
        else:
            log.info("clipboard changed after the paste: leaving the newer content alone")

    def _try_restore(self, clipboard: ClipboardLike, before: ClipboardSnapshot) -> None:
        try:
            clipboard.restore(before)
        except ClipboardError as exc:
            self._on_notice(f"your previous clipboard could not be put back ({exc})")

    def _leave_on_clipboard(self, clipboard: ClipboardLike, text: str, reason: str) -> None:
        try:
            clipboard.put_text(text)
            self._on_notice(f"not pasted: {reason}. The text is on the clipboard: press Ctrl+V.")
        except ClipboardError as exc:
            self._on_notice(f"not pasted: {reason}, and the clipboard is busy ({exc}).")
