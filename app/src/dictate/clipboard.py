"""The Windows clipboard, through ctypes: save everything on it, put text on it, put it back.

Used by the text inserter, which borrows the clipboard for a moment to paste a dictation.

What "save everything" means here: every format whose data is plain memory (text, rich text,
HTML, file lists, images as DIB, and the private formats of most apps) is copied byte for byte.
Formats whose data is a GDI handle (bitmaps, palettes, metafiles) cannot be copied this way and
are skipped; Windows rebuilds a bitmap from the DIB format on request, so that is harmless for
ordinary copied images.

The dictated text is put on the clipboard with three flags that tell Windows to keep it out of
clipboard history (Win+V), the cloud clipboard and clipboard monitors. The same flags go on the
restored content, so the restore does not show up as a new history entry either.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import time
from collections.abc import Iterator
from ctypes import wintypes
from dataclasses import dataclass

log = logging.getLogger(__name__)

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002

# Formats that hold a GDI handle rather than a block of memory: CF_BITMAP, CF_METAFILEPICT,
# CF_PALETTE, CF_ENHMETAFILE, CF_OWNERDISPLAY and the CF_DSP* display twins.
_HANDLE_FORMATS = frozenset({2, 3, 9, 14, 0x80, 0x82, 0x83, 0x8E})
_GDI_OBJECT_RANGE = range(0x300, 0x400)  # CF_GDIOBJFIRST .. CF_GDIOBJLAST
_CF_BITMAP, _CF_DIB, _CF_DIBV5 = 2, 8, 17

MAX_FORMAT_BYTES = 64 * 1024 * 1024  # a single format bigger than this is not copied

EXCLUDE_FORMATS = (
    "ExcludeClipboardContentFromMonitorProcessing",
    "CanIncludeInClipboardHistory",
    "CanUploadToCloudClipboard",
)


class ClipboardError(RuntimeError):
    """The clipboard could not be opened or written (usually: another program is using it)."""


@dataclass(frozen=True)
class ClipboardSnapshot:
    formats: tuple[tuple[int, bytes], ...]  # (format id, raw bytes), in the original order
    skipped: tuple[int, ...]  # formats that could not be copied

    @property
    def complete(self) -> bool:
        """True if nothing that matters was left out of the copy."""
        lost = set(self.skipped)
        have = {fmt for fmt, _ in self.formats}
        if _CF_BITMAP in lost and have & {_CF_DIB, _CF_DIBV5}:
            lost.discard(_CF_BITMAP)  # Windows rebuilds the bitmap from the DIB
        return not lost


user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

user32.OpenClipboard.argtypes = (wintypes.HWND,)
user32.OpenClipboard.restype = wintypes.BOOL
user32.CloseClipboard.restype = wintypes.BOOL
user32.EmptyClipboard.restype = wintypes.BOOL
user32.EnumClipboardFormats.argtypes = (wintypes.UINT,)
user32.EnumClipboardFormats.restype = wintypes.UINT
user32.GetClipboardData.argtypes = (wintypes.UINT,)
user32.GetClipboardData.restype = ctypes.c_void_p
user32.SetClipboardData.argtypes = (wintypes.UINT, ctypes.c_void_p)
user32.SetClipboardData.restype = ctypes.c_void_p
user32.GetClipboardSequenceNumber.restype = wintypes.DWORD
user32.RegisterClipboardFormatW.argtypes = (wintypes.LPCWSTR,)
user32.RegisterClipboardFormatW.restype = wintypes.UINT
user32.CreateWindowExW.argtypes = (
    wintypes.DWORD,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.HWND,
    wintypes.HMENU,
    wintypes.HINSTANCE,
    wintypes.LPVOID,
)
user32.CreateWindowExW.restype = wintypes.HWND
user32.DestroyWindow.argtypes = (wintypes.HWND,)
user32.DestroyWindow.restype = wintypes.BOOL
kernel32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
kernel32.GlobalAlloc.restype = ctypes.c_void_p
kernel32.GlobalLock.argtypes = (ctypes.c_void_p,)
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalUnlock.argtypes = (ctypes.c_void_p,)
kernel32.GlobalUnlock.restype = wintypes.BOOL
kernel32.GlobalSize.argtypes = (ctypes.c_void_p,)
kernel32.GlobalSize.restype = ctypes.c_size_t
kernel32.GlobalFree.argtypes = (ctypes.c_void_p,)
kernel32.GlobalFree.restype = ctypes.c_void_p


def _read_memory(handle: int) -> bytes | None:
    """Copy the bytes behind a clipboard memory handle, or None if it is not readable memory."""
    size = kernel32.GlobalSize(handle)
    if size == 0 or size > MAX_FORMAT_BYTES:
        return None
    pointer = kernel32.GlobalLock(handle)
    if not pointer:
        return None
    try:
        return ctypes.string_at(pointer, size)
    finally:
        kernel32.GlobalUnlock(handle)


class Clipboard:
    """Not thread-safe: create it, and use it, on one thread (it owns a hidden helper window)."""

    def __init__(self, *, open_attempts: int = 20, retry_delay: float = 0.025) -> None:
        self._open_attempts = open_attempts
        self._retry_delay = retry_delay
        self._owner: int | None = None
        self._exclude_ids = [user32.RegisterClipboardFormatW(name) for name in EXCLUDE_FORMATS]

    # -- plumbing ------------------------------------------------------------------------

    def _owner_window(self) -> int:
        # EmptyClipboard() only lets SetClipboardData work afterwards if the clipboard was
        # opened by a real window, so keep a hidden one. "STATIC" is a built-in window class.
        if self._owner is None:
            self._owner = user32.CreateWindowExW(
                0, "STATIC", None, 0, 0, 0, 0, 0, None, None, None, None
            )
            if not self._owner:
                raise ClipboardError(
                    f"could not create the clipboard window ({ctypes.get_last_error()})"
                )
        return self._owner

    @contextlib.contextmanager
    def _open(self) -> Iterator[None]:
        # Only one program can have the clipboard open at a time, so retry for a moment.
        owner = self._owner_window()
        for _ in range(self._open_attempts):
            if user32.OpenClipboard(owner):
                break
            time.sleep(self._retry_delay)
        else:
            raise ClipboardError("the clipboard is busy (another program has it open)")
        try:
            yield
        finally:
            user32.CloseClipboard()

    def _set(self, fmt: int, data: bytes) -> None:
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, max(1, len(data)))
        if not handle:
            raise ClipboardError("out of memory while writing to the clipboard")
        pointer = kernel32.GlobalLock(handle)
        ctypes.memmove(pointer, data, len(data))
        kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(fmt, handle):
            kernel32.GlobalFree(handle)  # on success Windows owns the memory; on failure we do
            raise ClipboardError(
                f"could not write clipboard format {fmt} ({ctypes.get_last_error()})"
            )

    def _set_exclusions(self) -> None:
        for fmt in self._exclude_ids:
            if fmt:
                self._set(fmt, b"\0\0\0\0")  # 0 = "no": keep out of history, cloud and monitors

    def close(self) -> None:
        if self._owner:
            user32.DestroyWindow(self._owner)
            self._owner = None

    # -- the four operations -------------------------------------------------------------

    def sequence_number(self) -> int:
        """Changes every time anything writes to the clipboard. Needs no open clipboard."""
        return int(user32.GetClipboardSequenceNumber())

    def snapshot(self) -> ClipboardSnapshot:
        """Copy everything currently on the clipboard."""
        formats: list[tuple[int, bytes]] = []
        skipped: list[int] = []
        with self._open():
            fmt = 0
            while True:
                fmt = user32.EnumClipboardFormats(fmt)
                if not fmt:
                    break
                if fmt in _HANDLE_FORMATS or fmt in _GDI_OBJECT_RANGE:
                    skipped.append(fmt)
                    continue
                handle = user32.GetClipboardData(fmt)
                data = _read_memory(handle) if handle else None
                if data is None:
                    skipped.append(fmt)
                else:
                    formats.append((fmt, data))
        return ClipboardSnapshot(tuple(formats), tuple(skipped))

    def put_text(self, text: str) -> int:
        """Replace the clipboard with plain text. Returns its sequence number.

        Later, "sequence_number() == this" means nothing has written to the clipboard since.
        The number is read AFTER the clipboard is closed: closing bumps it again (measured:
        157 while open, 160 right after closing), so a number read while open would never match.
        """
        data = (text + "\0").encode("utf-16-le")
        with self._open():
            user32.EmptyClipboard()
            self._set(CF_UNICODETEXT, data)
            self._set_exclusions()
        return int(user32.GetClipboardSequenceNumber())

    def restore(self, snapshot: ClipboardSnapshot) -> None:
        """Put a snapshot back, replacing whatever is on the clipboard now."""
        with self._open():
            user32.EmptyClipboard()
            for fmt, data in snapshot.formats:
                self._set(fmt, data)
            if snapshot.formats:
                self._set_exclusions()
