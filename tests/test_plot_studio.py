"""Publication plot window: the settings column scrolls with the mouse wheel."""
import tkinter as tk

import pytest

from edb.gui.plot_studio import PlotStudio

SERIES = [dict(name="AIOMFAC", kind="line", label="AIOMFAC", color="black"),
          dict(name="Experiment", kind="points", label="Experiment", color="black")]


@pytest.fixture(scope="module")
def root():
    """One Tk root for the module: creating several in one process is unreliable on Windows."""
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"no display: {error}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def studio(root):
    window = PlotStudio(root, [dict(s) for s in SERIES], "x", "y")
    window.geometry("1280x400")  # shorter than the settings column
    window.update()
    yield window
    window.destroy()


def wheel(widget, delta):
    """Wheel event at the centre of widget."""
    dx, dy = widget.winfo_width() // 2, widget.winfo_height() // 2
    widget.event_generate("<MouseWheel>", delta=delta, x=dx, y=dy, rootx=widget.winfo_rootx() + dx,
                          rooty=widget.winfo_rooty() + dy, when="now")
    widget.update()


def column_canvas(studio):
    return next(w for w in studio.winfo_children()[0].winfo_children() if isinstance(w, tk.Canvas))


def test_wheel_scrolls_settings(studio):
    canvas = column_canvas(studio)
    entry = next(w for w in canvas.winfo_children()[0].winfo_children() if w.winfo_class() == "TEntry")
    assert canvas.yview()[0] == 0.0 and canvas.yview()[1] < 1.0
    wheel(entry, -120)
    assert canvas.yview()[0] > 0.0
    wheel(canvas, 120)
    assert canvas.yview()[0] == 0.0


def test_wheel_over_combobox_scrolls_without_changing_value(studio):
    canvas = column_canvas(studio)
    combo = next(w for w in canvas.winfo_children()[0].winfo_children() if w.winfo_class() == "TCombobox")
    before = combo.get()
    wheel(combo, -120)
    assert canvas.yview()[0] > 0.0 and combo.get() == before


def test_wheel_over_plot_does_not_scroll_settings(studio):
    canvas = column_canvas(studio)
    wheel(studio.canvas.get_tk_widget(), -120)
    assert canvas.yview()[0] == 0.0
