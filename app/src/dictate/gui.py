"""The startup window: a short user guide with an OK button. Nothing else, for now.

It opens as soon as the app starts, while the speech model is still loading in the background,
so it carries one status line ("Loading ..." then "Ready"). Pressing OK (or closing the window)
just closes the window: the app keeps running in the background and the hotkey keeps working.

Layout: the title and status line stay at the top, the guide itself is in a scrollable area of
fixed height, and the OK button stays at the bottom. So the button can never end up off the
screen, however long the guide gets or however large the screen's text scaling is.

Tk must run on the main thread, so the app's other work (loading the model, the hotkey, the
recorder, transcription, pasting) happens on background threads and reports through
StartupStatus, which the window polls a few times a second.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import threading
import time
import tkinter as tk
from ctypes import wintypes
from tkinter import font as tkfont
from tkinter import ttk

log = logging.getLogger(__name__)

TITLE = "Dictate"
SUBTITLE = "Push-to-talk speech to text: speak, and the words appear at your cursor."

# (heading, explanation): shown as numbered steps.
STEPS = [
    (
        "Wait until the status says Ready",
        "The speech model loads first (the very first start also downloads it, about 630 MB).",
    ),
    ("Click where you want the text", "Any text box in any app: a document, a web page, a chat."),
    (
        "Tap Right Alt, then press it again and hold",
        "One quick tap, then press again straight away and keep holding.",
    ),
    ("Speak", "Talk normally while you hold the key, for up to 5 minutes."),
    ("Let go of Right Alt", "A moment later the text is pasted at your cursor."),
]

GOOD_TO_KNOW = [
    "Press any other key while holding Right Alt (for example Alt+Tab) and the recording is "
    "cancelled.",
    "A single tap, or two quick taps, does nothing.",
    "Your clipboard is borrowed to paste, then put back. If you copy something yourself right "
    "after, your copy is kept.",
    "Everything runs on this computer: no internet after the first download, no audio saved to "
    "disk. English only. The text is also printed in the console.",
]

OK_TEXT = "OK"

_STATE_COLOURS = {"loading": "#b45309", "ready": "#15803d", "failed": "#b91c1c"}
_STATE_DOT = "●"  # a filled circle

TEXT_WIDTH_CHARS = 70  # width of the guide area, in characters
#  Height of the guide area, in lines. Chosen to show the whole guide (about 21 lines at this
#  width); the window shrinks it, and the guide scrolls, only on a screen too small for that.
#  If the guide text grows, raise this.
TEXT_HEIGHT_LINES = 23
MIN_GUIDE_LINES = 8  # the guide area is never shorter than this, even on a tiny screen
SPI_GETWORKAREA = 0x0030  # Windows: the screen area not covered by the taskbar


class StartupStatus:
    """What the window shows on its status line. Safe to update from any thread."""

    def __init__(self, message: str = "Loading the speech model...") -> None:
        self._lock = threading.Lock()
        self._state = "loading"
        self._message = message

    def set_loading(self, message: str) -> None:
        with self._lock:
            self._state, self._message = "loading", message

    def set_ready(
        self, message: str = "Ready. Tap Right Alt, then press it again and hold."
    ) -> None:
        with self._lock:
            self._state, self._message = "ready", message

    def set_failed(self, message: str) -> None:
        with self._lock:
            self._state, self._message = "failed", message

    def snapshot(self) -> tuple[str, str]:
        """(state, message), where state is "loading", "ready" or "failed"."""
        with self._lock:
            return self._state, self._message


def _make_dpi_aware() -> None:
    # Without this Windows scales the window up as a blurry bitmap on high-resolution screens.
    with contextlib.suppress(AttributeError, OSError):
        ctypes.windll.shcore.SetProcessDpiAwareness(1)


class GuideWindow:
    def __init__(self, status: StartupStatus, master: tk.Tk | None = None) -> None:
        """``master`` is only for tests, which share one hidden Tk root: creating and
        destroying several Tk roots in one process is unreliable on Windows. The app passes
        nothing and the window becomes the Tk root itself."""
        _make_dpi_aware()
        self._status = status
        self.closed = False
        self._owns_root = master is None
        self.root: tk.Tk | tk.Toplevel = tk.Tk() if master is None else tk.Toplevel(master)
        self.root.title(f"{TITLE}: how to use")
        self.root.resizable(False, False)
        # Stay hidden while the window is built and sized. A window that is already on screen
        # keeps the position Windows gave it, so the centring below would be ignored.
        self.root.withdraw()
        self._build()
        self._centre()
        self.root.deiconify()
        self.root.bind("<Return>", lambda event: self.close())
        self.root.bind("<Escape>", lambda event: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._refresh()

    # -- layout --------------------------------------------------------------------------

    def _build(self) -> None:
        base = tkfont.nametofont("TkDefaultFont")
        base.configure(family="Segoe UI", size=10)
        self._title_font = tkfont.Font(family="Segoe UI", size=20, weight="bold")
        self._heading_font = tkfont.Font(family="Segoe UI", size=12, weight="bold")
        self._bold = tkfont.Font(family="Segoe UI", size=10, weight="bold")

        outer = ttk.Frame(self.root, padding=(22, 18, 22, 16))
        outer.grid(sticky="nsew")
        outer.columnconfigure(0, weight=1)

        # Top: title and the live status line (always visible).
        ttk.Label(outer, text=TITLE, font=self._title_font).grid(row=0, column=0, sticky="w")
        ttk.Label(outer, text=SUBTITLE).grid(row=1, column=0, sticky="w", pady=(2, 8))
        status_row = ttk.Frame(outer)
        status_row.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self._dot = tk.Label(status_row, text=_STATE_DOT, font=("Segoe UI", 12))
        self._dot.grid(row=0, column=0, padx=(0, 6))
        self._status_text = ttk.Label(status_row, font=self._bold, wraplength=520)
        self._status_text.grid(row=0, column=1, sticky="w")
        ttk.Separator(outer).grid(row=3, column=0, sticky="ew", pady=(0, 6))

        # Middle: the guide, in a scrollable area of fixed height.
        self._guide = self._build_guide(outer)

        # Bottom: the OK button, pinned in its own row so it is always on screen.
        buttons = ttk.Frame(outer)
        buttons.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        buttons.columnconfigure(0, weight=1)
        self.ok_button = ttk.Button(buttons, text=OK_TEXT, command=self.close, default="active")
        self.ok_button.grid(row=0, column=1, sticky="e", ipadx=16)

    def _build_guide(self, parent: ttk.Frame) -> tk.Text:
        area = ttk.Frame(parent)
        area.grid(row=4, column=0, sticky="nsew")
        area.columnconfigure(0, weight=1)
        background = ttk.Style().lookup("TFrame", "background") or self.root.cget("background")
        text = tk.Text(
            area,
            width=TEXT_WIDTH_CHARS,
            height=TEXT_HEIGHT_LINES,
            wrap="word",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            background=background,
            cursor="arrow",
            takefocus=0,
            padx=2,
            pady=2,
            font=tkfont.nametofont("TkDefaultFont"),
        )
        self._scrollbar = ttk.Scrollbar(area, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=self._on_scroll)
        text.grid(row=0, column=0, sticky="nsew")
        self._scrollbar.grid(row=0, column=1, sticky="ns")

        text.tag_configure("heading", font=self._heading_font, spacing1=6, spacing3=4)
        text.tag_configure("step", font=self._bold, lmargin1=0, lmargin2=26)
        text.tag_configure("detail", lmargin1=26, lmargin2=26, foreground="#444444", spacing3=6)
        text.tag_configure("note", lmargin1=8, lmargin2=24, spacing3=4)

        text.insert("end", "How to use it\n", "heading")
        for number, (heading, detail) in enumerate(STEPS, start=1):
            text.insert("end", f"{number}.  {heading}\n", "step")
            text.insert("end", f"{detail}\n", "detail")
        text.insert("end", "Good to know\n", "heading")
        for note in GOOD_TO_KNOW:
            text.insert("end", f"•  {note}\n", "note")
        text.configure(state="disabled")  # read-only
        return text

    def _on_scroll(self, first: str, last: str) -> None:
        # Show the scrollbar only when the guide is longer than its area.
        if float(first) <= 0.0 and float(last) >= 1.0:
            self._scrollbar.grid_remove()
        else:
            self._scrollbar.grid()
        self._scrollbar.set(first, last)

    def work_area(self) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) of the usable screen: the screen minus the taskbar."""
        rect = wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0):
            return rect.left, rect.top, rect.right, rect.bottom
        return 0, 0, self.root.winfo_screenwidth(), self.root.winfo_screenheight()

    def _title_bar_allowance(self) -> int:
        # The title bar sits above the area we size (about 30 px at 100% scaling, 45 px at 150%).
        scale = float(self.root.tk.call("tk", "scaling")) / (96 / 72)  # 1.0 at 100%, 1.5 at 150%
        return int(34 * scale)

    def _fit_guide_height(self) -> None:
        """Show the whole guide without scrolling if the screen is tall enough.

        Otherwise make the guide area as tall as fits in the usable screen (the part not covered
        by the taskbar, with room for the title bar) and let it scroll. The OK button is outside
        this area, so it always stays visible.
        """
        self.root.update_idletasks()
        _, top, _, bottom = self.work_area()
        limit = (bottom - top) - self._title_bar_allowance() - 24
        lines = TEXT_HEIGHT_LINES
        self._guide.configure(height=lines)
        self.root.update_idletasks()
        while self.root.winfo_reqheight() > limit and lines > MIN_GUIDE_LINES:
            lines -= 1
            self._guide.configure(height=lines)
            self.root.update_idletasks()

    def _centre(self) -> None:
        self._fit_guide_height()
        self.root.update_idletasks()
        left, top, right, bottom = self.work_area()
        width, height = self.root.winfo_reqwidth(), self.root.winfo_reqheight()
        spare = (bottom - top) - height - self._title_bar_allowance()
        x = left + max(0, (right - left - width) // 2)
        y = top + max(0, spare // 3)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        # Come up in front of the console window that started the app.
        self.root.attributes("-topmost", True)
        self.root.after(400, lambda: self.root.attributes("-topmost", False))
        self.ok_button.focus_set()

    # -- behaviour -----------------------------------------------------------------------

    def _refresh(self) -> None:
        if self.closed:
            return
        state, message = self._status.snapshot()
        self._dot.configure(fg=_STATE_COLOURS[state])
        self._status_text.configure(text=message)
        # Polling also lets Ctrl+C reach Python while Tk is waiting for events.
        self.root.after(200, self._refresh)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            if not self._owns_root:
                self.root.quit()  # a shared root keeps living, so stop its loop by hand
            self.root.destroy()  # destroying the app's own root also ends its loop

    def run(self) -> None:
        """Show the window and wait until it is closed."""
        self.root.mainloop()


def show_user_guide(status: StartupStatus, attempts: int = 3) -> bool:
    """Open the user guide and block until the user presses OK or closes it.

    Returns False if the window could not be opened at all. That must never stop the app:
    the guide is a convenience, and dictation works without it. Tk very occasionally fails to
    start on Windows (seen once or twice in many runs), so a failed start is retried a few times.
    """
    for attempt in range(attempts):
        try:
            window = GuideWindow(status)
            break
        except tk.TclError as exc:
            log.warning("the guide window could not open (attempt %d): %s", attempt + 1, exc)
            time.sleep(0.3)
    else:
        return False
    window.run()
    return True
