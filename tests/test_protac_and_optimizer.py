"""Tests for PROTAC ternary equilibrium conservation, hook effect, degradation ODEs, and MCTS."""

from __future__ import annotations

import math

from chiral_weave.molecular_optimizer import compute_pareto_front, generate_valence_safe_neighbors, optimize_lead_mcts
from chiral_weave.protac_cooperativity import (
    estimate_linker_cooperativity,
    simulate_protac_campaign,
    solve_ternary_equilibrium,
)
from chiral_weave.smiles_parser import parse_smiles


def test_douglass_ternary_equilibrium_exact_mass_conservation() -> None:
    """Solved species [P], [E], [L], [PL], [LE], [PLE] must conserve total mass to 1e-11 uM."""
    for l_tot in [0.001, 0.05, 0.5, 5.0, 100.0]:
        for alpha in [0.2, 1.0, 12.0]:
            st = solve_ternary_equilibrium(
                p_tot_uM=0.10,
                e_tot_uM=0.05,
                l_tot_uM=l_tot,
                kd_p_uM=0.02,
                kd_e_uM=0.06,
                alpha=alpha,
            )
            assert st.mass_balance_error_uM < 1e-11
            assert st.ple_uM >= 0.0
            assert st.ple_uM <= min(0.10, 0.05) + 1e-12


def test_protac_hook_effect_analytical_peak_and_cooperativity() -> None:
    """Ternary formation must follow a bell-shaped hook curve peaking near sqrt(Kd_P * Kd_E / alpha)."""
    # In the low-protein limit (P_tot, E_tot << Kd), hook peak = sqrt(Kd_P * Kd_E) independent of alpha
    kd_p, kd_e = 500.0, 2000.0
    out_low = simulate_protac_campaign(
        kd_target_nM=kd_p,
        kd_e3_nM=kd_e,
        cooperativity_alpha=5.0,
        p_tot_nM=5.0,
        e_tot_nM=5.0,
        hours=24.0,
    )
    expected_peak = math.sqrt(kd_p * kd_e)
    assert 0.5 * expected_peak <= out_low.hook_peak_nM <= 2.0 * expected_peak
    # Hook effect: highest dose (1e5 nM) must have far less ternary complex than the peak
    assert out_low.ternary_curve_uM[-1] < 0.25 * float(out_low.ternary_curve_uM.max())

    # Positive cooperativity alpha=10 must yield higher ternary peak & Dmax than negative alpha=0.2
    pos = simulate_protac_campaign(kd_target_nM=20.0, kd_e3_nM=40.0, cooperativity_alpha=10.0)
    neg = simulate_protac_campaign(kd_target_nM=20.0, kd_e3_nM=40.0, cooperativity_alpha=0.2)
    assert pos.peak_ternary_fraction > neg.peak_ternary_fraction
    assert pos.dmax_fraction > neg.dmax_fraction


def test_linker_cooperativity_and_valence_safe_mcts_pareto() -> None:
    """Linker cooperativity estimator and MCTS lead optimizer must stay chemically valid."""
    short_mol = parse_smiles("c1ccccc1Nc1ccccc1")
    mid_mol = parse_smiles("O=C(CCCCCOc1ccccc1)Nc1ccc(Cl)cc1")
    assert estimate_linker_cooperativity(mid_mol) > estimate_linker_cooperativity(short_mol)

    seed_mol = parse_smiles("c1ccc(Nc2ncnc3ccccc23)cc1")
    neighbors = generate_valence_safe_neighbors(seed_mol, max_neighbors=10)
    assert len(neighbors) >= 4
    for smi in neighbors:
        reparsed = parse_smiles(smi)
        assert reparsed.num_heavy_atoms >= 6

    report = optimize_lead_mcts(seed_mol.canonical_smiles, iterations=12, max_depth=2, seed=42)
    assert report.unique_evaluated >= 5
    assert report.best_candidate.mpo_score >= report.mpo_history[0]
    assert len(report.pareto_front) >= 1
    recomputed_front = compute_pareto_front(report.top_candidates)
    assert len(recomputed_front) >= 1


def test_synthetic_sar_cohort_json_provenance_and_negative_cooperativity(tmp_path) -> None:
    """Synthetic SAR cohort JSON export must record SHA-256 and alpha < 1 must depress ternary complex."""
    from chiral_weave.chemical_library import CompoundLibrary

    cohort = CompoundLibrary.synthetic_sar_cohort(seed=42)
    assert len(cohort.entries) == 36
    out_json = tmp_path / "cohort.json"
    cohort.save_json(out_json)
    text = out_json.read_text(encoding="utf-8")
    assert cohort.provenance_sha256() in text

    st_neg = solve_ternary_equilibrium(0.10, 0.05, 0.10, 0.02, 0.06, alpha=0.1)
    st_neutral = solve_ternary_equilibrium(0.10, 0.05, 0.10, 0.02, 0.06, alpha=1.0)
    assert st_neg.ple_uM < st_neutral.ple_uM
    assert st_neg.mass_balance_error_uM < 1e-11
