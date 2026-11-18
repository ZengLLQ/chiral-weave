"""Tests for Gasteiger charges, Crippen clogP, TPSA, QED, Murcko scaffolds, fingerprints, and MMPs."""

from __future__ import annotations

from chiral_weave.descriptors import (
    assign_pharmacophores,
    compute_gasteiger_charges,
    compute_physicochemical_profile,
    detect_structural_alerts,
    extract_bemis_murcko_scaffold,
)
from chiral_weave.fingerprints import (
    butina_cluster,
    compute_atom_pair_fingerprint,
    compute_maccs_keys,
    compute_morgan_fingerprint,
    dice_similarity,
    extract_mmp_pairs,
    maxmin_diversity_pick,
    tanimoto_similarity,
    tversky_similarity,
)
from chiral_weave.smiles_parser import parse_smiles


def test_gasteiger_charge_conservation_and_polarity() -> None:
    """Gasteiger charges must conserve total formal charge and reflect electronegativity."""
    for smi, expected_q in [
        ("CC(=O)Oc1ccccc1C(=O)O", 0.0),
        ("C[NH3+]", 1.0),
        ("CC(=O)[O-]", -1.0),
        ("C[C@H]([NH3+])C(=O)[O-]", 0.0),
    ]:
        mol = parse_smiles(smi)
        charges = compute_gasteiger_charges(mol)
        assert abs(float(charges.sum()) - expected_q) < 1e-10

    ethanol = parse_smiles("CCO")
    q_eth = compute_gasteiger_charges(ethanol)
    # Oxygen (idx 2) has net negative group charge relative to carbons
    assert q_eth[2] < -0.15
    assert q_eth[2] < q_eth[0]
    assert q_eth[1] > q_eth[0]


def test_physicochemical_qed_tpsa_and_alerts() -> None:
    """Verify Crippen clogP, Ertl TPSA, QED range, Ro5, and PAINS alerts."""
    gefitinib = parse_smiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1")
    prof = compute_physicochemical_profile(gefitinib)
    assert 440.0 < prof.molecular_weight < 455.0
    assert 0.2 < prof.qed < 0.9
    assert 1.0 <= prof.sa_score <= 10.0
    assert 50.0 < prof.tpsa < 85.0
    assert prof.lipinski_violations == 0

    michael = parse_smiles("C=CC(=O)Nc1ccccc1")
    alerts = detect_structural_alerts(michael)
    assert "michael_acceptor" in alerts

    pharma = assign_pharmacophores(gefitinib)
    assert len(pharma) >= 10


def test_bemis_murcko_scaffold_invariance() -> None:
    """Congeneric analogs sharing a core ring system must yield the same Murcko scaffold."""
    q1 = parse_smiles("c1ccc(Nc2ncnc3ccccc23)cc1")
    q2 = parse_smiles("Fc1cccc(Nc2ncnc3ccccc23)c1")
    q3 = parse_smiles("N#Cc1cccc(Nc2ncnc3ccccc23)c1")
    s1 = extract_bemis_murcko_scaffold(q1)
    s2 = extract_bemis_murcko_scaffold(q2)
    s3 = extract_bemis_murcko_scaffold(q3)
    assert s1 == s2 == s3
    assert len(s1) > 8

    acyclic = parse_smiles("CCCCCC(=O)O")
    assert extract_bemis_murcko_scaffold(acyclic) == ""


def test_fingerprint_similarity_metrics_and_triangle_inequality() -> None:
    """ECFP4, FCFP4, AtomPair, and MACCS must obey identity, bounds, and Soergel triangle inequality."""
    m1 = parse_smiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1")
    m2 = parse_smiles("COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC")
    m3 = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")

    fp1 = compute_morgan_fingerprint(m1, radius=2, nbits=1024)
    fp2 = compute_morgan_fingerprint(m2, radius=2, nbits=1024)
    fp3 = compute_morgan_fingerprint(m3, radius=2, nbits=1024)

    assert tanimoto_similarity(fp1, fp1) == 1.0
    assert dice_similarity(fp1, fp1) == 1.0
    assert tversky_similarity(fp1, fp2, alpha=0.5, beta=0.5) == dice_similarity(fp1, fp2)
    assert tanimoto_similarity(fp1, fp2) > tanimoto_similarity(fp1, fp3)

    # Soergel distance d = 1 - Tanimoto satisfies the metric triangle inequality
    d12 = 1.0 - tanimoto_similarity(fp1, fp2)
    d23 = 1.0 - tanimoto_similarity(fp2, fp3)
    d13 = 1.0 - tanimoto_similarity(fp1, fp3)
    assert d13 <= d12 + d23 + 1e-12

    fcfp1 = compute_morgan_fingerprint(m1, radius=2, nbits=1024, use_features=True)
    ap1 = compute_atom_pair_fingerprint(m1, nbits=512)
    maccs1 = compute_maccs_keys(m1)
    assert fcfp1.sum() > 5
    assert ap1.sum() > 10
    assert maccs1[3] == 1 and maccs1[20] == 1  # halogen & aromatic N


def test_butina_clustering_maxmin_and_mmp_cliffs() -> None:
    """Verify Butina clustering, MaxMin diversity selection, and MMP SALI activity cliffs."""
    smiles_act = [
        ("c1ccc(Nc2ncnc3ccccc23)cc1", 6.50),
        ("Fc1cccc(Nc2ncnc3ccccc23)c1", 7.10),
        ("FC(F)(F)c1cccc(Nc2ncnc3ccccc23)c1", 8.20),
        ("CC(=O)Oc1ccccc1C(=O)O", 4.60),
        ("CC(C)Cc1ccc(C(C)C(=O)O)cc1", 5.10),
    ]
    mols = [parse_smiles(s) for s, _ in smiles_act]
    acts = [a for _, a in smiles_act]
    fps = [compute_morgan_fingerprint(m, radius=2, nbits=1024) for m in mols]

    clust = butina_cluster(fps, similarity_threshold=0.55)
    assert sum(len(c) for c in clust.clusters) == len(mols)
    # The 3 quinazolines must cluster together
    assert any(0 in c and 1 in c for c in clust.clusters)

    picked = maxmin_diversity_pick(fps, k=3, seed_idx=0)
    assert len(picked) == 3
    assert len(set(picked)) == 3
    assert 3 in picked or 4 in picked

    mmps = extract_mmp_pairs(mols, acts)
    assert len(mmps) >= 1
    assert mmps[0].sali_score > 0.0


def test_lipinski_veber_boundaries_and_tpsa_scaling() -> None:
    """Lipinski Ro5 and Veber rules must flag oversized lipophilic scaffolds cleanly."""
    small = parse_smiles("CC(=O)Nc1ccc(O)cc1")
    p_small = compute_physicochemical_profile(small)
    assert p_small.lipinski_violations == 0
    assert p_small.veber_pass is True

    large_lipo = parse_smiles("CCCCCCCCCCCCCCCCc1ccc(C(=O)Nc2ccc(CCCCCCCC)cc2)cc1")
    p_large = compute_physicochemical_profile(large_lipo)
    assert p_large.lipinski_violations >= 2
    assert p_large.veber_pass is False


def test_morgan_radius_and_tversky_asymmetry() -> None:
    """ECFP6 (r=3) captures more environments than ECFP4 (r=2); Tversky detects substructures."""
    from chiral_weave.fingerprints import compute_morgan_counts, pairwise_tanimoto_matrix

    core = parse_smiles("c1ccc(Nc2ncnc3ccccc23)cc1")
    decorated = parse_smiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1")

    c_r2 = compute_morgan_counts(decorated, radius=2)
    c_r3 = compute_morgan_counts(decorated, radius=3)
    assert len(c_r3) > len(c_r2)

    fp_core = compute_morgan_fingerprint(core, radius=2, nbits=1024)
    fp_dec = compute_morgan_fingerprint(decorated, radius=2, nbits=1024)
    # Substructure focus (alpha=0.9, beta=0.1) is higher when core is the reference
    tv_sub = tversky_similarity(fp_core, fp_dec, alpha=0.9, beta=0.1)
    tv_sup = tversky_similarity(fp_core, fp_dec, alpha=0.1, beta=0.9)
    assert tv_sub > tv_sup

    mat = pairwise_tanimoto_matrix([fp_core, fp_dec])
    assert mat.shape == (2, 2)
    assert abs(mat[0, 1] - mat[1, 0]) < 1e-12
    assert mat[0, 0] == 1.0 and mat[1, 1] == 1.0
