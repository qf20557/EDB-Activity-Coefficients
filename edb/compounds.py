"""compounds.json: densities, molar masses and refractive indices."""
import json
import math

from .config import COMPOUNDS_PATH


def load_compounds(path=COMPOUNDS_PATH):
    """Return {name: record} with numeric density, molar mass and (optional) refractive index."""
    with open(path, encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "compounds" in data:
        data = data["compounds"]
    if isinstance(data, list):
        data = {record["name"]: record for record in data}
    for name, record in data.items():
        for key in ("density_g_ml", "molar_mass_g_mol", "refractive_index"):
            if key not in record:
                if key == "refractive_index":
                    continue
                raise ValueError(f"{name}: missing {key}.")
            value = record[key]
            value = float(value["value"] if isinstance(value, dict) else value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}: {key} must be positive.")
            record[key] = value
    return data
