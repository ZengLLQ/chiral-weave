"""Tests for 3D Distance Geometry conformer embedding, chirality preservation, and docking."""

from __future__ import annotations

import numpy as np

from chiral_weave.conformer_docking import (
    build_kinase_hinge_pocket,
    dock_molecule,
    generate_3d_conformer,
    score_ligand_pose,
    signed_chiral_volume,
)
from chiral_weave.smiles_parser import parse_smiles


def test_3d_conformer_bond_lengths_and_planar_ring() -> None:
    """Generated 3D coordinates must honor covalent bond lengths and benzene planarity."""
    benzene = parse_smiles("c1ccccc1")
    coords = generate_3d_conformer(benzene, seed=42)
    assert coords.shape == (6, 3)

    for b in benzene.bonds:
        dist = float(np.linalg.norm(coords[b.begin_idx] - coords[b.end_idx]))
        assert 1.28 < dist < 1.52

    # Smallest covariance eigenvalue measures out-of-plane deviation
    centered = coords - coords.mean(axis=0, keepdims=True)
    eigvals = np.linalg.eigvalsh(centered.T @ centered)
    assert eigvals[0] < 0.05


def test_chiral_volume_opposite_handedness_preserved() -> None:
    """L-alanine (@) and D-alanine (@@) must embed with opposite signed tetrahedral volumes."""
    mol_ccw = parse_smiles("C[C@H](N)C(=O)O")
    mol_cw = parse_smiles("C[C@@H](N)C(=O)O")
    c_ccw = generate_3d_conformer(mol_ccw, seed=42)
    c_cw = generate_3d_conformer(mol_cw, seed=42)

    nbrs_ccw = sorted(mol_ccw.neighbors(1))[:3]
    nbrs_cw = sorted(mol_cw.neighbors(1))[:3]
    v_ccw = signed_chiral_volume(c_ccw, 1, nbrs_ccw)
    v_cw = signed_chiral_volume(c_cw, 1, nbrs_cw)

    assert v_ccw < -0.2
    assert v_cw > 0.2


def test_empirical_docking_score_decomposition_and_pose_improvement() -> None:
    """Pose optimization must improve or match initial unoptimized pose and decompose cleanly."""
    pocket = build_kinase_hinge_pocket()
    gefitinib = parse_smiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1")
    init_coords = generate_3d_conformer(gefitinib, seed=42)
    init_pose = score_ligand_pose(gefitinib, init_coords, pocket)

    opt_pose = dock_molecule(gefitinib, pocket=pocket, steps=55, seed=42)
    assert opt_pose.total_affinity_kcal <= init_pose.total_affinity_kcal + 0.05
    assert opt_pose.total_affinity_kcal < -1.0
    assert opt_pose.gauss1_score < 0.0
    assert opt_pose.repulsion_score >= 0.0


def test_distance_bounds_triangle_inequality_and_rotatable_torsions() -> None:
    """Smoothed distance bounds must satisfy U_ij <= U_ik + U_kj and skip ring bonds in torsions."""
    from chiral_weave.conformer_docking import _compute_bounds_matrices, _find_rotatable_torsions

    mol = parse_smiles("COc1ccc(C(=O)Nc2ccccc2)cc1")
    lower, upper = _compute_bounds_matrices(mol)
    assert np.all(upper >= lower)
    n = mol.num_atoms
    for i in range(n):
        for j in range(n):
            for k in range(n):
                assert upper[i, j] <= upper[i, k] + upper[k, j] + 1e-8

    torsions = _find_rotatable_torsions(mol)
    for u, v, _ in torsions:
        bond = mol.find_bond(u, v)
        assert bond is not None
        assert not bond.in_ring
        assert not bond.is_aromatic


def test_rodrigues_and_pose_reproducibility_by_seed() -> None:
    """Docking with identical PRNG seeds must be deterministic and preserve coordinate shape."""
    pocket = build_kinase_hinge_pocket()
    mol_a = parse_smiles("Fc1cccc(Nc2ncnc3ccccc23)c1")
    mol_b = parse_smiles("Fc1cccc(Nc2ncnc3ccccc23)c1")
    pose_a = dock_molecule(mol_a, pocket=pocket, steps=25, seed=99)
    pose_b = dock_molecule(mol_b, pocket=pocket, steps=25, seed=99)
    assert np.allclose(pose_a.coords, pose_b.coords, atol=1e-8)
    assert abs(pose_a.total_affinity_kcal - pose_b.total_affinity_kcal) < 1e-8


def test_pocket_clash_wall_penalizes_steric_overlap() -> None:
    """Placing ligand directly onto a pocket receptor site must incur positive repulsion penalty."""
    pocket = build_kinase_hinge_pocket()
    benzene = parse_smiles("c1ccccc1")
    coords = generate_3d_conformer(benzene, seed=42)
    shifted = coords + pocket.sites[0].coord[None, :]
    clash_pose = score_ligand_pose(benzene, shifted, pocket)
    assert clash_pose.repulsion_score > 1.0
