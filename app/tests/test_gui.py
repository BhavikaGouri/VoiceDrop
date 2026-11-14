import tkinter as tk

import pytest

from dictate import gui
from dictate.gui import GOOD_TO_KNOW, STEPS, GuideWindow, StartupStatus

# -- the words ----------------------------------------------------------------------------


def all_text() -> str:
    return " ".join([gui.SUBTITLE] + [h + " " + t for h, t in STEPS] + GOOD_TO_KNOW)


def test_the_guide_explains_the_gesture():
    text = all_text()
    assert "Right Alt" in text
    assert "tap" in text.lower() and "hold" in text.lower()


def test_the_guide_mentions_the_limits_that_matter():
    text = all_text()
    assert "5 minutes" in text
    assert "English" in text
    assert "clipboard" in text.lower()
    assert "no audio saved" in text


def test_the_steps_are_in_order_and_each_has_a_heading_and_a_text():
    assert len(STEPS) >= 4
    assert all(heading and text for heading, text in STEPS)
    assert "Ready" in STEPS[0][0]  # waiting for the model comes first


# -- the status line ----------------------------------------------------------------------


def test_status_starts_as_loading_and_can_become_ready_or_failed():
    s = StartupStatus()
    assert s.snapshot()[0] == "loading"
    s.set_loading("Downloading the model...")
    assert s.snapshot() == ("loading", "Downloading the model...")
    s.set_ready()
    assert s.snapshot()[0] == "ready"
    s.set_failed("no microphone")
    assert s.snapshot() == ("failed", "no microphone")


# -- the window (needs a desktop, so it is skipped where there is none) -------------------


@pytest.fixture(scope="module")
def shared_root():
    # One hidden Tk root for every window test: making and destroying several Tk roots in a
    # single process fails now and then on Windows ("can't find a usable tk.tcl").
    gui._make_dpi_aware()  # the app does this before it creates its Tk root; so must the test
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        if "display" in str(exc).lower():
            pytest.skip("no display available for Tk")
        raise  # any other Tk problem is a real failure, not a reason to skip
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def window(shared_root):
    w = GuideWindow(StartupStatus(), master=shared_root)
    yield w
    if not w.closed:
        w.close()


def test_the_ok_button_is_always_inside_the_usable_screen(window):
    window.root.update()
    _, top, _, bottom = window.work_area()  # the screen minus the taskbar
    bottom_of_button = window.ok_button.winfo_rooty() + window.ok_button.winfo_height()
    assert bottom_of_button <= bottom
    assert window.root.winfo_y() >= top  # the title bar is on screen too


def test_the_whole_guide_is_visible_without_scrolling_when_the_screen_allows(window):
    window.root.update()
    _, top, _, bottom = window.work_area()
    if bottom - top < 900:
        pytest.skip("a small screen: the guide is meant to scroll here")
    first, last = window._guide.yview()
    assert (first, last) == (0.0, 1.0)  # nothing is cut off, no scrollbar needed


def test_a_window_that_cannot_open_is_reported_not_raised(monkeypatch):
    attempts = []

    def broken(status):
        attempts.append(1)
        raise tk.TclError("Can't find a usable tk.tcl")

    monkeypatch.setattr(gui, "GuideWindow", broken)
    monkeypatch.setattr(gui.time, "sleep", lambda s: None)
    assert gui.show_user_guide(StartupStatus()) is False
    assert len(attempts) == 3  # it tried a few times before giving up


def test_a_window_that_fails_once_is_retried(monkeypatch):
    calls = []

    class Flaky:
        def __init__(self, status):
            calls.append(1)
            if len(calls) == 1:
                raise tk.TclError("transient")

        def run(self):
            calls.append("ran")

    monkeypatch.setattr(gui, "GuideWindow", Flaky)
    monkeypatch.setattr(gui.time, "sleep", lambda s: None)
    assert gui.show_user_guide(StartupStatus()) is True
    assert calls == [1, 1, "ran"]


def test_the_window_shows_the_status_and_follows_changes(window):
    window.root.update()
    assert "Loading" in window._status_text.cget("text")
    window._status.set_ready()
    window._refresh()
    assert window._status_text.cget("text").startswith("Ready")
    window._status.set_failed("The microphone is missing")
    window._refresh()
    assert "microphone" in window._status_text.cget("text")


def test_pressing_ok_closes_the_window_and_run_returns(window):
    assert window.ok_button.cget("text") == "OK"
    window.root.after(100, window.ok_button.invoke)
    window.run()  # returns only because OK closed the window
    assert window.closed


def test_escape_and_enter_are_bound_to_closing_it(window):
    # (A synthetic key press only reaches a window that has keyboard focus, which a test
    # window cannot rely on, so check the bindings exist and that closing works instead.)
    assert window.root.bind("<Escape>")
    assert window.root.bind("<Return>")


def test_closing_twice_is_harmless(window):
    window.close()
    window.close()
    assert window.closed
