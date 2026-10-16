"""Main window. Stage 1 is built at start-up; stages 2 and 3 (and pandas, which only they use) are imported and
built the first time they are opened or receive data."""
import importlib
import tkinter as tk
from tkinter import messagebox, ttk

from ..compounds import load_compounds
from ..config import COMPOUNDS_PATH


class PipelineApp(tk.Tk):
    STAGES = (("1 · Clean & fit", ".clean_fit", "CleanFitStage"),
              ("2 · MFS & gradients", ".mfs_stage", "MfsStage"),
              ("3 · Activity coefficients", ".activity_stage", "ActivityStage"))

    def __init__(self):
        super().__init__()
        self.title("EDB Pipeline: evaporation data → activity coefficients")
        self.geometry("1250x940")
        self.compounds = {}
        try:
            self.compounds = load_compounds()
        except Exception as error:
            messagebox.showerror("compounds.json", f"Place a valid {COMPOUNDS_PATH.name} beside this script.\n"
                                                   f"{error}")
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        self.tabs, self.stages = [], {}
        for text, _, _ in self.STAGES:
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=text)
            self.tabs.append(tab)
        self.stage(0)
        self.notebook.bind("<<NotebookTabChanged>>", lambda _: self.stage(self.notebook.index("current")))

    def stage(self, index):
        """The stage frame for a tab, built on first use."""
        if index not in self.stages:
            _, module, name = self.STAGES[index]
            cls = getattr(importlib.import_module(module, __package__), name)
            self.config(cursor="watch")
            self.update_idletasks()
            try:
                frame = cls(self.tabs[index], self)
            finally:
                self.config(cursor="")
            frame.pack(fill="both", expand=True)
            self.stages[index] = frame
        return self.stages[index]

    @property
    def stage1(self):
        return self.stage(0)

    @property
    def stage2(self):
        return self.stage(1)

    @property
    def stage3(self):
        return self.stage(2)

    def select(self, index):
        self.notebook.select(index)

    def to_stage2(self):
        data = self.stage1.handoff()
        if data is None:
            return
        try:
            self.stage2.receive(data)
        except ValueError as error:
            messagebox.showerror("Hand-over failed", str(error), parent=self)
            return
        self.select(1)

    def to_stage3(self):
        experiment = self.stage2.handoff()
        if experiment is None:
            return
        self.stage3.receive(experiment)
        self.select(2)


def main():
    PipelineApp().mainloop()
