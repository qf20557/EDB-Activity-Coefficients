import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

DB_PATH = Path(__file__).resolve().parent / "compounds.json"

DEFAULT_COMPOUNDS = {
    "Malonic acid": {"molar_mass_g_mol": 104.061, "density_g_ml": 1.600},
    "Tween 20": {"molar_mass_g_mol": 1227.53, "density_g_ml": 1.095},
    "Water": {"molar_mass_g_mol": 18.01528, "density_g_ml": 0.997},
}

MODE_DRY = "Dry binary formulation (A + B)"
MODE_PRESET = "Mole fraction preset (measured A + B masses)"
MODE_SINGLE = "Single solute in temporary solvent (A + solvent)"

# Which input rows are visible (in order) for each formulation type
MODE_FIELDS = {
    MODE_DRY: ["comp_a", "comp_b", "solvent", "basis", "fraction",
               "total_mass", "solvent_mass", "radius"],
    MODE_PRESET: ["comp_a", "comp_b", "solvent", "mass_a", "mass_b",
                  "solvent_mass", "radius"],
    MODE_SINGLE: ["comp_a", "solvent", "mass_a", "solvent_mass", "radius"],
}


def parse_number(value, label, allow_zero=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    if number < 0 or (number == 0 and not allow_zero):
        raise ValueError(
            f"{label} must be {'zero or a positive number' if allow_zero else 'a positive number'}."
        )
    return number


def load_compounds():
    if not DB_PATH.exists():
        return {k: v.copy() for k, v in DEFAULT_COMPOUNDS.items()}
    try:
        with DB_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not data:
            raise ValueError("Database must contain at least one compound.")
        for name, values in data.items():
            if not isinstance(values, dict):
                raise ValueError("Invalid compound record.")
            parse_number(values.get("molar_mass_g_mol"), f"{name} molar mass")
            parse_number(values.get("density_g_ml"), f"{name} density")
        return data
    except (OSError, json.JSONDecodeError, ValueError):
        messagebox.showwarning(
            "Compounds warning",
            "Could not read compounds.json. The built-in starter compounds will be used.",
        )
        return {k: v.copy() for k, v in DEFAULT_COMPOUNDS.items()}


class EDBSolutionDesigner(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EDB Solution Designer")
        self.minsize(560, 560)
        self.compounds = load_compounds()
        self._build_ui()
        self.update_fields()

    def _build_ui(self):
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)

        self.mode_var = tk.StringVar(value=MODE_DRY)
        mode_box = ttk.Combobox(frame, textvariable=self.mode_var, state="readonly",
                                values=list(MODE_FIELDS))
        mode_box.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        mode_box.bind("<<ComboboxSelected>>", self.update_fields)

        self.form = ttk.Frame(frame)
        self.form.grid(row=1, column=0, sticky="ew")
        self.form.columnconfigure(1, weight=1)

        names = sorted(self.compounds)
        pick = lambda wanted: wanted if wanted in names else names[0]
        self.vars = {
            "comp_a": tk.StringVar(value=pick("Malonic acid")),
            "comp_b": tk.StringVar(value=pick("Tween 20")),
            "solvent": tk.StringVar(value=pick("Water")),
            "basis": tk.StringVar(value="Mole fraction"),
            "fraction": tk.StringVar(value="0.300"),
            "total_mass": tk.StringVar(value="10.000"),
            "mass_a": tk.StringVar(value="1.000"),
            "mass_b": tk.StringVar(value="1.000"),
            "solvent_mass": tk.StringVar(value="5.000"),
            "radius": tk.StringVar(value="25.000"),
        }
        combo = lambda key, values: ttk.Combobox(
            self.form, textvariable=self.vars[key], values=values, state="readonly")
        entry = lambda key: ttk.Entry(self.form, textvariable=self.vars[key])

        # key -> (label text, widget); labels for mass_a / total_mass set per mode
        self.rows = {
            "comp_a": ("Component A", combo("comp_a", names)),
            "comp_b": ("Component B", combo("comp_b", names)),
            "solvent": ("Temporary solvent", combo("solvent", names)),
            "basis": ("Fraction basis", combo("basis", ["Mole fraction", "Mass fraction"])),
            "fraction": ("Fraction of A (0 to 1)", entry("fraction")),
            "total_mass": ("Total dry A + B mass (g)", entry("total_mass")),
            "mass_a": ("Mass of A (g)", entry("mass_a")),
            "mass_b": ("Mass of B (g)", entry("mass_b")),
            "solvent_mass": ("Solvent mass (g)", entry("solvent_mass")),
            "radius": ("Initial droplet radius (µm)", entry("radius")),
        }
        self.row_labels = {k: ttk.Label(self.form, text=v[0]) for k, v in self.rows.items()}

        ttk.Button(frame, text="Calculate", command=self.calculate).grid(
            row=2, column=0, sticky="ew", pady=10)
        self.result_text = tk.Text(frame, height=16, wrap="word", font=("TkFixedFont", 10))
        self.result_text.grid(row=3, column=0, sticky="nsew")
        self.result_text.configure(state="disabled")
        ttk.Label(
            frame, wraplength=520, foreground="#555555",
            text=("Dry-radius estimate assumes complete solvent loss, no loss of solutes, "
                  "spherical droplets and additive volumes. Planning estimate only."),
        ).grid(row=4, column=0, sticky="w", pady=(8, 0))

    def update_fields(self, _event=None):
        for key in self.rows:
            self.row_labels[key].grid_remove()
            self.rows[key][1].grid_remove()
        for i, key in enumerate(MODE_FIELDS[self.mode_var.get()]):
            self.row_labels[key].grid(row=i, column=0, sticky="w", padx=(0, 12), pady=4)
            self.rows[key][1].grid(row=i, column=1, sticky="ew", pady=4)
        self.set_result("")

    def _compound(self, key):
        name = self.vars[key].get()
        if name not in self.compounds:
            raise ValueError("Select all required compounds.")
        return name, self.compounds[name]

    def calculate(self):
        try:
            mode = self.mode_var.get()
            v = self.vars
            name_a, a = self._compound("comp_a")
            name_s, s = self._compound("solvent")
            radius = parse_number(v["radius"].get(), "Initial droplet radius")
            mass_s = parse_number(v["solvent_mass"].get(), "Solvent mass", allow_zero=True)
            mw_a, rho_a = a["molar_mass_g_mol"], a["density_g_ml"]
            extra = []

            if mode == MODE_SINGLE:
                if name_a == name_s:
                    raise ValueError("Component A and the solvent must be different compounds.")
                mass_a = parse_number(v["mass_a"].get(), "Mass of A")
                dry_volume = mass_a / rho_a
                lines = ["ADD", "", f"{mass_a:.6f} g {name_a}", f"{mass_s:.6f} g {name_s}"]
                extra = [f"MASS FRACTION OF {name_a}: {mass_a / (mass_a + mass_s):.6f}"]
            else:
                name_b, b = self._compound("comp_b")
                if name_a == name_b:
                    raise ValueError("Select different compounds for A and B.")
                mw_b, rho_b = b["molar_mass_g_mol"], b["density_g_ml"]

                if mode == MODE_PRESET:
                    mass_a = parse_number(v["mass_a"].get(), "Mass of A")
                    mass_b = parse_number(v["mass_b"].get(), "Mass of B")
                    n_a, n_b = mass_a / mw_a, mass_b / mw_b
                else:
                    frac = parse_number(v["fraction"].get(), "Fraction of A", allow_zero=True)
                    if frac > 1:
                        raise ValueError("Fraction of A must be between 0 and 1.")
                    total = parse_number(v["total_mass"].get(), "Total dry mass")
                    if v["basis"].get().startswith("Mole"):
                        n_total = total / (frac * mw_a + (1 - frac) * mw_b)
                        mass_a, mass_b = frac * n_total * mw_a, (1 - frac) * n_total * mw_b
                    else:
                        mass_a, mass_b = frac * total, (1 - frac) * total
                    n_a, n_b = mass_a / mw_a, mass_b / mw_b

                x_a = n_a / (n_a + n_b)
                dry_volume = mass_a / rho_a + mass_b / rho_b
                lines = ["ADD", "", f"{mass_a:.6f} g {name_a}", f"{mass_b:.6f} g {name_b}",
                         f"{mass_s:.6f} g {name_s}"]
                extra = [f"MOLE FRACTION OF {name_a}: {x_a:.6f}",
                         f"MOLE FRACTION OF {name_b}: {1 - x_a:.6f}",
                         f"MASS FRACTION OF {name_a}: {mass_a / (mass_a + mass_b):.6f}"]

            wet_volume = dry_volume + mass_s / s["density_g_ml"]
            dry_radius = radius * (dry_volume / wet_volume) ** (1 / 3)
            lines += [""] + extra + [
                "", f"INITIAL DROPLET RADIUS: {radius:.3f} µm",
                f"INITIAL DROPLET DIAMETER: {2 * radius:.3f} µm",
                "", "ESTIMATED DRY RADIUS", "", f"{dry_radius:.3f} µm",
                f"Equivalent dry diameter: {2 * dry_radius:.3f} µm",
            ]
            self.set_result("\n".join(lines) + "\n")
        except ValueError as exc:
            messagebox.showerror("Calculation input", str(exc))

    def set_result(self, text):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")


if __name__ == "__main__":
    EDBSolutionDesigner().mainloop()