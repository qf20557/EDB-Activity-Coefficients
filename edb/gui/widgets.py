"""Small Tk building blocks shared by the stages."""
import tkinter as tk
from tkinter import ttk


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
