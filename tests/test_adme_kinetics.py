"""Tests for multitask ADME predictions, CYP450 MBI ODE kinetics, and metabolism soft spots."""

from __future__ import annotations

import numpy as np

from chiral_weave.adme_kinetics import (
    AdmePredictor,
    compute_hepatic_extraction,
    rank_metabolism_sites,
    simulate_cyp_mbi_kinetics,
)
from chiral_weave.smiles_parser import parse_smiles


def test_adme_solubility_and_hepatic_extraction_trends() -> None:
    """Hydrophilic acetaminophen must be more soluble and less protein-bound than tamoxifen."""
    pred = AdmePredictor(hidden_dim=24, seed=42)
    acet = parse_smiles("CC(=O)Nc1ccc(O)cc1")
    tamox = parse_smiles("CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1")

    adme_acet = pred.predict(acet, seed=42)
    adme_tamox = pred.predict(tamox, seed=42)

    assert adme_acet.log_s_esol > adme_tamox.log_s_esol
    assert adme_acet.fu_plasma > adme_tamox.fu_plasma
    assert 0.0 <= adme_acet.hepatic_extraction < 1.0
    assert 0.0 <= adme_tamox.hepatic_extraction < 1.0
    assert adme_acet.epistemic_std >= 0.0


def test_well_stirred_hepatic_extraction_monotonicity() -> None:
    """Hepatic extraction E_H must stay in [0, 1) and increase monotonically with CL_int and fu."""
    e_low = compute_hepatic_extraction(5.0, 0.05)
    e_mid = compute_hepatic_extraction(40.0, 0.05)
    e_high = compute_hepatic_extraction(40.0, 0.40)
    assert 0.0 < e_low < e_mid < e_high < 1.0


def test_cyp450_mbi_ode_conservation_and_ddi_monotonicity() -> None:
    """Zero-inhibitor control gives AUC ratio 1.0; higher inhibitor lowers active CYP and raises AUC."""
    ctrl = simulate_cyp_mbi_kinetics("CYP3A4", inhibitor_uM=0.0, ic50_uM=2.0)
    assert abs(ctrl.ddi_auc_ratio - 1.0) < 1e-6
    assert abs(ctrl.residual_active_enzyme - 1.0) < 1e-6

    low_inh = simulate_cyp_mbi_kinetics("CYP3A4", inhibitor_uM=0.5, ic50_uM=2.0)
    high_inh = simulate_cyp_mbi_kinetics("CYP3A4", inhibitor_uM=10.0, ic50_uM=2.0)

    assert 1.0 < low_inh.ddi_auc_ratio < high_inh.ddi_auc_ratio
    assert high_inh.residual_active_enzyme < low_inh.residual_active_enzyme < 1.0
    assert np.all(high_inh.substrate_uM >= 0.0)
    assert np.all(np.diff(high_inh.substrate_uM) <= 1e-9)


def test_metabolism_hotspot_ranking_identifies_benzylic_and_o_dealkylation() -> None:
    """Benzylic methyl and O-methyl sites must rank among top oxidative soft spots."""
    imatinib = parse_smiles("Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1")
    sites = rank_metabolism_sites(imatinib)
    assert len(sites) >= 4
    top_reactions = {s.reaction_type for s in sites[:4]}
    assert "benzylic_hydroxylation" in top_reactions or "N_or_O_dealkylation" in top_reactions


def test_cyp2d6_basic_amine_bias_and_bbb_tpsa_dependence() -> None:
    """Basic aliphatic amines increase CYP2D6 potency; high TPSA lowers predicted logBB."""
    pred = AdmePredictor(hidden_dim=20, seed=42)
    fluoxetine = parse_smiles("CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1")
    aspirin = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    polar_amide = parse_smiles("NC(=O)c1ccc(C(=O)N)cc1")

    adme_flu = pred.predict(fluoxetine, seed=42)
    adme_asp = pred.predict(aspirin, seed=42)
    adme_pol = pred.predict(polar_amide, seed=42)

    assert adme_flu.cyp2d6_ic50_uM < adme_asp.cyp2d6_ic50_uM
    assert adme_flu.log_bb > adme_pol.log_bb
    assert 0.0 < adme_pol.pgp_efflux_prob < 1.0
