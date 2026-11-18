"""PROTAC ternary complex equilibrium (Douglass exact solver), hook-effect curves, and degradation ODEs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

from chiral_weave.chem_types import MoleculeGraph, ProtacOutcome
from chiral_weave.descriptors import count_rotatable_bonds


@dataclass(frozen=True)
class TernarySpeciesState:
    """Exact equilibrium concentrations (uM) of all species in a P-L-E ternary system."""

    p_free_uM: float
    e_free_uM: float
    l_free_uM: float
    pl_uM: float
    le_uM: float
    ple_uM: float
    mass_balance_error_uM: float


def _free_pe_given_l(
    l_free: float,
    p_tot: float,
    e_tot: float,
    kd_p: float,
    kd_e: float,
    alpha: float,
) -> Tuple[float, float]:
    """Solve exact coupled quadratic for free [P] and [E] given free [L] >= 0."""
    if l_free <= 0.0:
        return p_tot, e_tot

    ap = 1.0 + l_free / kd_p
    ae = 1.0 + l_free / kd_e
    c = (alpha * l_free) / (kd_p * kd_e)

    if c <= 1e-18:
        return p_tot / ap, e_tot / ae

    # [P]*(ap + c*[E]) = p_tot, [E]*(ae + c*[P]) = e_tot
    # => ap*c*[P]^2 + (ap*ae + c*(e_tot - p_tot))*[P] - p_tot*ae = 0
    a_quad = ap * c
    b_quad = ap * ae + c * (e_tot - p_tot)
    c_quad = -p_tot * ae
    disc = max(0.0, b_quad * b_quad - 4.0 * a_quad * c_quad)
    p_free = (-b_quad + math.sqrt(disc)) / (2.0 * a_quad)
    p_free = max(0.0, min(p_tot, p_free))
    e_free = e_tot / (ae + c * p_free)
    return p_free, e_free


def solve_ternary_equilibrium(
    p_tot_uM: float,
    e_tot_uM: float,
    l_tot_uM: float,
    kd_p_uM: float,
    kd_e_uM: float,
    alpha: float = 1.0,
) -> TernarySpeciesState:
    """Solve exact Douglass ternary complex equilibrium via monotonic Brent root-finding.

    Guarantees mass conservation:
      [P]_free + [PL] + [PLE] = P_tot
      [E]_free + [LE] + [PLE] = E_tot
      [L]_free + [PL] + [LE] + [PLE] = L_tot
    to machine precision (~1e-13 uM).
    """
    if l_tot_uM <= 0.0:
        return TernarySpeciesState(p_tot_uM, e_tot_uM, 0.0, 0.0, 0.0, 0.0, 0.0)
    if kd_p_uM <= 0.0 or kd_e_uM <= 0.0 or alpha <= 0.0:
        raise ValueError("Dissociation constants and cooperativity alpha must be positive.")

    def _l_total_residual(l_free: float) -> float:
        p_f, e_f = _free_pe_given_l(l_free, p_tot_uM, e_tot_uM, kd_p_uM, kd_e_uM, alpha)
        pl = (p_f * l_free) / kd_p_uM
        le = (l_free * e_f) / kd_e_uM
        ple = (alpha * p_f * l_free * e_f) / (kd_p_uM * kd_e_uM)
        return (l_free + pl + le + ple) - l_tot_uM

    l_free = float(brentq(_l_total_residual, 0.0, l_tot_uM, xtol=1e-14, rtol=1e-13, maxiter=200))
    p_f, e_f = _free_pe_given_l(l_free, p_tot_uM, e_tot_uM, kd_p_uM, kd_e_uM, alpha)
    pl = (p_f * l_free) / kd_p_uM
    le = (l_free * e_f) / kd_e_uM
    ple = (alpha * p_f * l_free * e_f) / (kd_p_uM * kd_e_uM)

    err_p = abs((p_f + pl + ple) - p_tot_uM)
    err_e = abs((e_f + le + ple) - e_tot_uM)
    err_l = abs((l_free + pl + le + ple) - l_tot_uM)
    return TernarySpeciesState(
        p_free_uM=p_f,
        e_free_uM=e_f,
        l_free_uM=l_free,
        pl_uM=pl,
        le_uM=le,
        ple_uM=ple,
        mass_balance_error_uM=max(err_p, err_e, err_l),
    )


def estimate_linker_cooperativity(
    mol: MoleculeGraph,
    target_gap_angstrom: float = 11.5,
    persistence_length_angstrom: float = 4.2,
) -> float:
    """Estimate ternary cooperativity alpha from polymer worm-like chain linker statistics.

    Models effective local concentration and warhead-E3 steric clash penalty
    as a function of rotatable linker bonds and molecular flexibility.
    """
    rotb = count_rotatable_bonds(mol)
    contour_len = max(3.5, 1.45 * rotb + 4.0)
    if contour_len < target_gap_angstrom * 0.72:
        # Linker too short: steric clash prevents simultaneous P + E engagement (alpha < 1)
        clash_ratio = (target_gap_angstrom * 0.72 - contour_len) / target_gap_angstrom
        return float(max(0.08, math.exp(-5.0 * clash_ratio)))

    # Gaussian / WLC end-to-end probability density peak around r ~ sqrt(2 * lp * Lc)
    r_rms = math.sqrt(2.0 * persistence_length_angstrom * contour_len)
    strain = ((r_rms - target_gap_angstrom) / max(3.0, r_rms)) ** 2
    entropic_cost = 0.08 * max(0, rotb - 8)
    alpha = 8.5 * math.exp(-2.8 * strain - entropic_cost)
    return float(max(0.15, min(25.0, alpha)))


def simulate_protac_campaign(
    kd_target_nM: float = 15.0,
    kd_e3_nM: float = 45.0,
    cooperativity_alpha: float = 6.5,
    p_tot_nM: float = 100.0,
    e_tot_nM: float = 50.0,
    hours: float = 24.0,
    k_deg_basal_per_h: float = 0.05,
    k_cat_ub_per_h: float = 2.8,
) -> ProtacOutcome:
    """Simulate PROTAC dose-response hook effect and target protein degradation kinetics.

    Scans PROTAC concentration from 0.01 nM to 100 uM (1e5 nM), solves exact
    ternary equilibrium at each dose, finds TF50 and hook-peak concentration,
    and integrates target degradation ODEs at the peak ternary dose.
    """
    p_tot_uM = p_tot_nM * 1e-3
    e_tot_uM = e_tot_nM * 1e-3
    kd_p_uM = kd_target_nM * 1e-3
    kd_e_uM = kd_e3_nM * 1e-3

    dose_grid_nM = np.logspace(-2.0, 5.0, 95)
    ternary_curve_uM = np.zeros_like(dose_grid_nM)
    for idx, d_nM in enumerate(dose_grid_nM):
        state = solve_ternary_equilibrium(
            p_tot_uM=p_tot_uM,
            e_tot_uM=e_tot_uM,
            l_tot_uM=float(d_nM) * 1e-3,
            kd_p_uM=kd_p_uM,
            kd_e_uM=kd_e_uM,
            alpha=cooperativity_alpha,
        )
        ternary_curve_uM[idx] = state.ple_uM

    peak_idx = int(np.argmax(ternary_curve_uM))
    lo_d = float(dose_grid_nM[max(0, peak_idx - 1)])
    hi_d = float(dose_grid_nM[min(len(dose_grid_nM) - 1, peak_idx + 1)])
    peak_opt = minimize_scalar(
        lambda log_d: -solve_ternary_equilibrium(
            p_tot_uM=p_tot_uM,
            e_tot_uM=e_tot_uM,
            l_tot_uM=(10.0 ** float(log_d)) * 1e-3,
            kd_p_uM=kd_p_uM,
            kd_e_uM=kd_e_uM,
            alpha=cooperativity_alpha,
        ).ple_uM,
        bounds=(math.log10(lo_d), math.log10(hi_d)),
        method="bounded",
    )
    hook_peak_nM = float(10.0 ** float(peak_opt.x))
    peak_ple_uM = float(-peak_opt.fun)
    limiting_uM = min(p_tot_uM, e_tot_uM)
    peak_fraction = peak_ple_uM / max(1e-12, limiting_uM)

    # Half-maximal ternary formation (TF50) on ascending limb
    half_peak = 0.5 * peak_ple_uM
    tf50_nM = hook_peak_nM
    for i in range(peak_idx):
        if ternary_curve_uM[i] <= half_peak <= ternary_curve_uM[i + 1]:
            w = (half_peak - ternary_curve_uM[i]) / max(1e-12, ternary_curve_uM[i + 1] - ternary_curve_uM[i])
            tf50_nM = float(dose_grid_nM[i] * (1.0 - w) + dose_grid_nM[i + 1] * w)
            break

    # Target protein degradation ODE at each dose up to `hours` to extract DC50 & Dmax
    def _remaining_protein_after_hours(dose_nM: float) -> Tuple[float, np.ndarray, np.ndarray]:
        t_eval = np.linspace(0.0, hours, 97)
        l_uM = dose_nM * 1e-3

        def _rhs(_t: float, y: np.ndarray) -> list[float]:
            p_cur = max(1e-6, float(y[0]))
            st = solve_ternary_equilibrium(p_cur, e_tot_uM, l_uM, kd_p_uM, kd_e_uM, cooperativity_alpha)
            k_syn = k_deg_basal_per_h * p_tot_uM
            dp_dt = k_syn - k_deg_basal_per_h * p_cur - k_cat_ub_per_h * st.ple_uM
            return [dp_dt]

        sol = solve_ivp(_rhs, (0.0, hours), [p_tot_uM], t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-9)
        frac = np.clip(sol.y[0] / max(1e-12, p_tot_uM), 0.0, 1.5)
        return float(frac[-1]), t_eval, frac

    # Evaluate degradation at peak dose and ascending doses to compute DC50 & Dmax
    rem_peak, time_grid_h, peak_time_frac = _remaining_protein_after_hours(hook_peak_nM)
    dmax_fraction = float(max(0.0, 1.0 - rem_peak))

    dc50_nM = hook_peak_nM
    target_rem = 1.0 - 0.5 * dmax_fraction
    scan_indices = np.linspace(0, peak_idx, min(14, peak_idx + 1), dtype=int)
    prev_d = float(dose_grid_nM[scan_indices[0]])
    prev_r, _, _ = _remaining_protein_after_hours(prev_d)
    for idx_s in scan_indices[1:]:
        cur_d = float(dose_grid_nM[int(idx_s)])
        cur_r, _, _ = _remaining_protein_after_hours(cur_d)
        if prev_r >= target_rem >= cur_r:
            w = (prev_r - target_rem) / max(1e-12, prev_r - cur_r)
            dc50_nM = float(prev_d * (1.0 - w) + cur_d * w)
            break
        prev_d, prev_r = cur_d, cur_r

    return ProtacOutcome(
        kd_target_nM=kd_target_nM,
        kd_e3_nM=kd_e3_nM,
        cooperativity_alpha=cooperativity_alpha,
        hook_peak_nM=hook_peak_nM,
        tf50_nM=tf50_nM,
        dc50_nM=dc50_nM,
        dmax_fraction=dmax_fraction,
        peak_ternary_fraction=float(peak_fraction),
        dose_grid_nM=dose_grid_nM,
        ternary_curve_uM=ternary_curve_uM,
        time_grid_h=time_grid_h,
        target_protein_fraction=peak_time_frac,
    )
