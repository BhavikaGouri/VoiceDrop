"""Windows-only plumbing for the hotkey: a global keyboard hook and key injection.

Talks to user32 through ctypes, so there are no third-party dependencies.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Virtual-key codes we care about.
VK_RMENU = 0xA5  # Right Alt
VK_DUMMY = 0xE8  # unassigned key; tapped to stop Windows reading an Alt release as "open menu"

# Win32 constants.
WH_KEYBOARD_LL = 13  # low-level (global) keyboard hook
HC_ACTION = 0  # hook code: the event carries a real key message
WM_QUIT = 0x0012
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104  # what Windows sends instead of WM_KEYDOWN while Alt is held
WM_SYSKEYUP = 0x0105
LLKHF_INJECTED = 0x10  # event was synthesised (SendInput), not typed on the keyboard
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001  # Right Alt/Ctrl are "extended" keys
KEYEVENTF_KEYUP = 0x0002

LRESULT = ctypes.c_ssize_t
ULONG_PTR = ctypes.c_size_t

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


# -- C structures ----------------------------------------------------------------------------


class KBDLLHOOKSTRUCT(ctypes.Structure):
    """What Windows passes to the hook for every key event."""

    _fields_ = (
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class _InputUnion(ctypes.Union):
    # MOUSEINPUT is the largest member. It has to be here or sizeof(INPUT) comes out wrong
    # and SendInput rejects the call.
    _fields_ = (("ki", KEYBDINPUT), ("mi", MOUSEINPUT))


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = (("type", wintypes.DWORD), ("u", _InputUnion))


# -- function signatures (required for correct 64-bit calls) ---------------------------------

HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

user32.SetWindowsHookExW.argtypes = (ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD)
user32.SetWindowsHookExW.restype = wintypes.HHOOK
user32.CallNextHookEx.argtypes = (wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
user32.CallNextHookEx.restype = LRESULT
user32.UnhookWindowsHookEx.argtypes = (wintypes.HHOOK,)
user32.UnhookWindowsHookEx.restype = wintypes.BOOL
user32.GetMessageW.argtypes = (
    ctypes.POINTER(wintypes.MSG),
    wintypes.HWND,
    wintypes.UINT,
    wintypes.UINT,
)
user32.GetMessageW.restype = ctypes.c_int
user32.PostThreadMessageW.argtypes = (
    wintypes.DWORD,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
)
user32.PostThreadMessageW.restype = wintypes.BOOL
user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT
user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
user32.GetAsyncKeyState.restype = ctypes.c_short
kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
kernel32.GetModuleHandleW.restype = wintypes.HMODULE
kernel32.GetCurrentThreadId.restype = wintypes.DWORD


# -- key state and injection -----------------------------------------------------------------


@dataclass(frozen=True)
class KeyEvent:
    vk: int  # virtual-key code, e.g. VK_RMENU
    is_down: bool  # True for press (including auto-repeat), False for release
    injected: bool  # synthesised by software rather than typed


def is_key_down(vk: int) -> bool:
    """Physical state of a key right now, regardless of what the hook has seen."""
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def _key_input(vk: int, flags: int = 0) -> INPUT:
    return INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=vk, dwFlags=flags))


def send_key(vk: int, *, down: bool, extended: bool = False) -> None:
    """Inject a single key event. Used by tests and demos to drive the hook."""
    flags = (0 if down else KEYEVENTF_KEYUP) | (KEYEVENTF_EXTENDEDKEY if extended else 0)
    event = _key_input(vk, flags)
    user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))


def send_dummy_key() -> None:
    """Tap the unassigned VK_DUMMY key.

    Windows opens the foreground app's menu bar or ribbon when Alt is pressed and released
    with nothing in between. Sending this just before the Alt release puts a key in between,
    so a Right Alt tap or hold never triggers that.
    """
    events = (INPUT * 2)(_key_input(VK_DUMMY), _key_input(VK_DUMMY, KEYEVENTF_KEYUP))
    sent = user32.SendInput(2, events, ctypes.sizeof(INPUT))
    if sent != 2:
        log.warning("SendInput injected %d of 2 events (error %d)", sent, ctypes.get_last_error())


VK_CONTROL = 0x11
VK_V = 0x56


def send_ctrl_v() -> None:
    """Press and release Ctrl+V as one atomic SendInput call, so no other key can slip in between.

    These are injected keys, so the hotkey hook ignores them.
    """
    events = (INPUT * 4)(
        _key_input(VK_CONTROL),
        _key_input(VK_V),
        _key_input(VK_V, KEYEVENTF_KEYUP),
        _key_input(VK_CONTROL, KEYEVENTF_KEYUP),
    )
    sent = user32.SendInput(4, events, ctypes.sizeof(INPUT))
    if sent != 4:
        log.warning("SendInput injected %d of 4 events (error %d)", sent, ctypes.get_last_error())


# -- the hook --------------------------------------------------------------------------------


class KeyboardHook:
    """Global keyboard hook with its own message-loop thread.

    ``callback`` runs inside the hook, so it must return quickly: Windows silently removes a
    hook that takes more than a few hundred milliseconds. Keys are never swallowed, so every
    shortcut keeps working while this is installed.
    """

    def __init__(self, callback: Callable[[KeyEvent], None]) -> None:
        self._callback = callback
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._ready = threading.Event()
        self._error: OSError | None = None
        # Keep a reference: if the callback object is garbage collected, Windows calls freed memory.
        self._proc = HOOKPROC(self._on_event)

    def start(self) -> None:
        """Install the hook. Raises OSError if Windows refuses."""
        self._thread = threading.Thread(target=self._run, name="keyboard-hook", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            raise self._error

    def stop(self) -> None:
        if self._thread is None:
            return
        user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)  # ends the message loop
        self._thread.join(timeout=2)
        self._thread = None

    def _run(self) -> None:
        # A hook only fires while its thread pumps messages, so install and pump on this thread.
        self._thread_id = kernel32.GetCurrentThreadId()
        hook = user32.SetWindowsHookExW(
            WH_KEYBOARD_LL, self._proc, kernel32.GetModuleHandleW(None), 0
        )
        if not hook:
            self._error = ctypes.WinError(ctypes.get_last_error())
            self._ready.set()
            return
        self._ready.set()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            pass  # returns 0 on WM_QUIT
        user32.UnhookWindowsHookEx(hook)

    def _on_event(self, n_code: int, w_param: int, l_param: int) -> int:
        if n_code == HC_ACTION:
            try:
                data = ctypes.cast(l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                is_down = w_param in (WM_KEYDOWN, WM_SYSKEYDOWN)
                is_up = w_param in (WM_KEYUP, WM_SYSKEYUP)
                if is_down or is_up:
                    self._callback(
                        KeyEvent(
                            vk=data.vkCode,
                            is_down=is_down,
                            injected=bool(data.flags & LLKHF_INJECTED),
                        )
                    )
            except Exception:
                # An exception escaping into Windows would break typing system-wide.
                log.exception("keyboard hook callback failed")
        return user32.CallNextHookEx(None, n_code, w_param, l_param)
