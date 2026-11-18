"""Streamlit Solution Designer (migrated from EDB Solutions.py)."""
import math

import streamlit as st

from ..compounds import load_compounds
from .widgets import match_compound

DEFAULT_COMPOUNDS = {
    "Malonic acid": {"molar_mass_g_mol": 104.061, "density_g_ml": 1.600},
    "Tween 20": {"molar_mass_g_mol": 1227.53, "density_g_ml": 1.095},
    "Water": {"molar_mass_g_mol": 18.01528, "density_g_ml": 0.997},
}

MODE_DRY = "Dry binary formulation (A + B)"
MODE_PRESET = "Mole fraction preset (measured A + B masses)"
MODE_SINGLE = "Single solute in temporary solvent (A + solvent)"
MODES = (MODE_DRY, MODE_PRESET, MODE_SINGLE)


def _pos_number(value, label, allow_zero=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    if number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"{label} must be {'zero or a positive number' if allow_zero else 'a positive number'}.")
    return number


def compute_solution_summary(compounds, mode, name_a, name_b, name_s, basis, fraction, total_mass, mass_a, mass_b,
                             solvent_mass, radius):
    if name_a not in compounds or name_s not in compounds:
        raise ValueError("Select all required compounds.")
    a, s = compounds[name_a], compounds[name_s]
    r0 = _pos_number(radius, "Initial droplet radius")
    m_s = _pos_number(solvent_mass, "Solvent mass", allow_zero=True)
    mw_a, rho_a = a["molar_mass_g_mol"], a["density_g_ml"]

    if mode == MODE_SINGLE:
        if name_a == name_s:
            raise ValueError("Component A and the solvent must be different compounds.")
        m_a = _pos_number(mass_a, "Mass of A")
        dry_volume = m_a / rho_a
        lines = ["ADD", "", f"{m_a:.6f} g {name_a}", f"{m_s:.6f} g {name_s}"]
        extra = [f"MASS FRACTION OF {name_a}: {m_a / (m_a + m_s):.6f}"]
    else:
        if name_b not in compounds:
            raise ValueError("Select all required compounds.")
        if name_a == name_b:
            raise ValueError("Select different compounds for A and B.")
        b = compounds[name_b]
        mw_b, rho_b = b["molar_mass_g_mol"], b["density_g_ml"]
        if mode == MODE_PRESET:
            m_a = _pos_number(mass_a, "Mass of A")
            m_b = _pos_number(mass_b, "Mass of B")
        else:
            frac = _pos_number(fraction, "Fraction of A", allow_zero=True)
            if frac > 1:
                raise ValueError("Fraction of A must be between 0 and 1.")
            total = _pos_number(total_mass, "Total dry mass")
            if str(basis).startswith("Mole"):
                n_total = total / (frac * mw_a + (1 - frac) * mw_b)
                m_a, m_b = frac * n_total * mw_a, (1 - frac) * n_total * mw_b
            else:
                m_a, m_b = frac * total, (1 - frac) * total
        n_a, n_b = m_a / mw_a, m_b / mw_b
        x_a = n_a / (n_a + n_b)
        dry_volume = m_a / rho_a + m_b / rho_b
        lines = ["ADD", "", f"{m_a:.6f} g {name_a}", f"{m_b:.6f} g {name_b}", f"{m_s:.6f} g {name_s}"]
        extra = [
            f"MOLE FRACTION OF {name_a}: {x_a:.6f}",
            f"MOLE FRACTION OF {name_b}: {1 - x_a:.6f}",
            f"MASS FRACTION OF {name_a}: {m_a / (m_a + m_b):.6f}",
        ]

    wet_volume = dry_volume + m_s / s["density_g_ml"]
    dry_radius = r0 * (dry_volume / wet_volume) ** (1 / 3)
    lines += [""] + extra + [
        "",
        f"INITIAL DROPLET RADIUS: {r0:.3f} µm",
        f"INITIAL DROPLET DIAMETER: {2 * r0:.3f} µm",
        "",
        "ESTIMATED DRY RADIUS",
        "",
        f"{dry_radius:.3f} µm",
        f"Equivalent dry diameter: {2 * dry_radius:.3f} µm",
    ]
    return "\n".join(lines)


def render_solution_designer(compounds=None):
    if not compounds:
        try:
            compounds = load_compounds()
        except Exception:
            compounds = {k: v.copy() for k, v in DEFAULT_COMPOUNDS.items()}
            st.warning("Could not read compounds.json; using built-in starter compounds.")

    names = sorted(compounds)
    defaults = {
        "sd_mode": MODE_DRY,
        "sd_comp_a": match_compound(compounds, "Malonic acid"),
        "sd_comp_b": match_compound(compounds, "Tween 20"),
        "sd_solvent": match_compound(compounds, "Water"),
        "sd_basis": "Mole fraction",
        "sd_fraction": "0.300",
        "sd_total_mass": "10.000",
        "sd_mass_a": "1.000",
        "sd_mass_b": "1.000",
        "sd_solvent_mass": "5.000",
        "sd_radius": "25.000",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    col_form, col_out = st.columns([1.1, 1.1], gap="large")
    with col_form:
        with st.container(border=True):
            st.markdown("##### Droplet Solution Formulation")
            mode = st.selectbox("Formulation mode", MODES, key="sd_mode")
            st.selectbox("Component A", names, key="sd_comp_a")
            if mode != MODE_SINGLE:
                st.selectbox("Component B", names, key="sd_comp_b")
            st.selectbox("Temporary solvent", names, key="sd_solvent")

            if mode == MODE_DRY:
                st.selectbox("Fraction basis", ("Mole fraction", "Mass fraction"), key="sd_basis")
                st.text_input("Fraction of A (0 to 1)", key="sd_fraction")
                st.text_input("Total dry A + B mass (g)", key="sd_total_mass")
            elif mode == MODE_PRESET:
                st.text_input("Mass of A (g)", key="sd_mass_a")
                st.text_input("Mass of B (g)", key="sd_mass_b")
            else:
                st.text_input("Mass of A (g)", key="sd_mass_a")

            st.text_input("Solvent mass (g)", key="sd_solvent_mass")
            st.text_input("Initial droplet radius (µm)", key="sd_radius")
            st.caption(
                "Dry-radius estimate assumes complete solvent loss, no loss of solutes, "
                "spherical droplets and additive volumes. Planning estimate only."
            )

    with col_out:
        with st.container(border=True):
            st.markdown("##### Calculated Recipe & Dry Radius")
            try:
                summary = compute_solution_summary(
                    compounds=compounds,
                    mode=st.session_state["sd_mode"],
                    name_a=st.session_state["sd_comp_a"],
                    name_b=st.session_state["sd_comp_b"],
                    name_s=st.session_state["sd_solvent"],
                    basis=st.session_state["sd_basis"],
                    fraction=st.session_state["sd_fraction"],
                    total_mass=st.session_state["sd_total_mass"],
                    mass_a=st.session_state["sd_mass_a"],
                    mass_b=st.session_state["sd_mass_b"],
                    solvent_mass=st.session_state["sd_solvent_mass"],
                    radius=st.session_state["sd_radius"],
                )
                st.code(summary, language="text")
            except ValueError as err:
                st.error(str(err))
