"""Small Tk building blocks shared by the stages."""
import queue
import threading
import tkinter as tk
from tkinter import ttk

from ..config import MAX_PLOT_POINTS
from ..trace import Cancelled


def scrollable(parent):
    """A vertically scrolling frame. The content fills the window when it fits and scrolls when it does not."""
    outer = ttk.Frame(parent)
    canvas = tk.Canvas(outer, highlightthickness=0)
    bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=bar.set)
    canvas.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")
    content = ttk.Frame(canvas)
    window = canvas.create_window((0, 0), window=content, anchor="nw")

    def fit(_=None):
        height = max(canvas.winfo_height(), content.winfo_reqheight())
        canvas.itemconfigure(window, width=canvas.winfo_width(), height=height)
        canvas.configure(scrollregion=(0, 0, canvas.winfo_width(), height))

    content.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", fit)
    return outer, content


def scroll_column(parent):
    """A column as wide as its content that scrolls vertically, with a scroll bar and the mouse wheel anywhere over
    it. Returns (outer, content, wheel): bind wheel to widgets whose own wheel binding should scroll the column
    instead (comboboxes otherwise change value)."""
    outer = ttk.Frame(parent)
    canvas = tk.Canvas(outer, highlightthickness=0, yscrollincrement=20)
    bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=bar.set)
    bar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="y")
    content = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=content, anchor="nw")
    content.bind("<Configure>", lambda _: canvas.configure(width=content.winfo_reqwidth(),
                                                           scrollregion=canvas.bbox("all")))

    def wheel(event):
        if canvas.yview() != (0.0, 1.0):
            notches = event.delta / 120 if event.delta else (1 if event.num == 4 else -1)  # Button-4/5: X11
            canvas.yview_scroll(-round(3 * notches) or (-1 if notches > 0 else 1), "units")
        return "break"

    def over_column(event):
        # Tcl path, not a widget: the combobox drop-down list has no tkinter object
        path = str(outer.tk.call("winfo", "containing", event.x_root, event.y_root))
        if path == str(outer) or path.startswith(str(outer) + "."):
            return wheel(event)

    for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        parent.winfo_toplevel().bind(sequence, over_column, add="+")
    return outer, content, wheel


def nav_bar(parent, back=None, next_=None, next_text="Next →"):
    bar = ttk.Frame(parent, padding=(6, 4))
    bar.pack(fill="x", side="bottom")
    ttk.Separator(parent, orient="horizontal").pack(fill="x", side="bottom")
    if next_ is not None:
        ttk.Button(bar, text=next_text, command=next_).pack(side="right", padx=4)
    if back is not None:
        ttk.Button(bar, text="← Back", command=back).pack(side="right", padx=4)
    return bar


class Progress(ttk.Frame):
    """A progress bar with a Cancel button that runs slow work (reading or saving a large file) on a background
    thread, so the window keeps responding. Hidden unless something is running."""

    def __init__(self, parent):
        super().__init__(parent)
        self.busy = False
        self.text = tk.StringVar()
        self.bar = ttk.Progressbar(self, length=220, maximum=1.0)
        self.button = ttk.Button(self, text="Cancel", command=self.cancel)
        self.stop = threading.Event()
        for widget in (ttk.Label(self, textvariable=self.text), self.bar, self.button):
            widget.pack(side="left", padx=3)

    def cancel(self):
        self.stop.set()
        self.text.set("Cancelling...")

    def run(self, text, work, done, failed, cancelled=None, lock=()):
        """Run work(progress, cancel) on a thread, then call done(result) or failed(error) on the Tk thread.
        progress takes a fraction from 0 to 1; cancel() is True once Cancel is pressed, and work should then raise
        Cancelled. The widgets in lock are disabled meanwhile. Returns False if something is already running."""
        if self.busy:
            return False
        self.busy, self.stop = True, threading.Event()
        messages, stop = queue.SimpleQueue(), self.stop
        self.text.set(text)
        self.bar["value"] = 0
        self.pack(side="left", padx=8)
        for widget in lock:
            widget.state(["disabled"])

        def target():
            try:
                messages.put(("done", work(lambda fraction: messages.put(("progress", fraction)), stop.is_set)))
            except Cancelled:
                messages.put(("cancelled", None))
            except Exception as error:
                messages.put(("failed", error))

        def poll():
            kind = value = None
            while not messages.empty():
                kind, value = messages.get()
                if kind != "progress":
                    break
            if kind == "progress":
                self.bar["value"] = value
                if not stop.is_set():
                    self.text.set(f"{text} {100 * value:.0f}%")
            if kind in (None, "progress"):
                self.after(40, poll)
                return
            self.busy = False
            self.pack_forget()
            for widget in lock:
                widget.state(["!disabled"])
            if kind == "done":
                done(value)
            elif kind == "failed":
                failed(value)
            elif cancelled is not None:
                cancelled()

        threading.Thread(target=target, daemon=True).start()
        self.after(40, poll)
        return True


class PointView:
    """The points of a trace that are drawn (see Trace.display), recomputed only when the x range, the stride or the
    culled points change, and re-thinned shortly after the x range of the axes stops changing so that zooming in
    brings back the detail."""

    def __init__(self, widget, axis, redraw, delay=120):
        self.widget, self.axis, self.redraw, self.delay = widget, axis, redraw, delay
        self.trace = self.key = self.job = None
        self.points, self.thinned = (), False
        axis.callbacks.connect("xlim_changed", self.moved)

    def moved(self, _):
        if self.job is not None:
            self.widget.after_cancel(self.job)
        self.job = self.widget.after(self.delay, self.settle)

    def settle(self):
        self.job = None
        self.redraw()

    def get(self, trace, stride, whole=False):
        """(t, r, changed) for the current x range plus half a width each side, or for the whole trace."""
        lo = hi = None
        if stride is None and not whole:
            lo, hi = sorted(self.axis.get_xlim())
            lo, hi = lo - (hi - lo) / 2, hi + (hi - lo) / 2
        rows = None if stride is not None else trace.window(lo, hi)
        key = (trace.version, stride, rows)
        changed = trace is not self.trace or key != self.key
        if changed:
            self.trace, self.key, self.points = trace, key, trace.display(lo, hi, stride)
            self.thinned = rows is not None and rows[1] - rows[0] > MAX_PLOT_POINTS
        return (*self.points, changed)
