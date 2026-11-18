"""3D Distance Geometry conformer generation, chiral volume restraints, and Vina-style empirical docking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import math
import numpy as np
from scipy.optimize import minimize

from chiral_weave.chem_types import (
    ELEMENT_TABLE,
    BondOrder,
    DockingPose,
    Hybridization,
    MoleculeGraph,
    StereoConfig,
)


@dataclass(frozen=True)
class PocketSite:
    """Receptor binding-pocket interaction anchor or steric wall atom."""

    residue: str
    role: str  # "HBD", "HBA", "HYDROPHOBIC", "AROMATIC", "WALL"
    coord: np.ndarray
    vdw_radius: float = 1.70


@dataclass(frozen=True)
class ReceptorPocket:
    """3D protein binding pocket definition for empirical ligand docking."""

    name: str
    center: np.ndarray
    sites: Tuple[PocketSite, ...]


def build_kinase_hinge_pocket() -> ReceptorPocket:
    """Construct an EGFR-like ATP-competitive kinase cleft with open solvent front (Angstrom)."""
    sites = (
        # Hinge backbone NH/CO along +X end of the cleft
        PocketSite("Met793_NH", "HBD", np.array([7.4, 0.8, 0.4]), 1.55),
        PocketSite("Met793_CO", "HBA", np.array([7.2, -1.8, 0.2]), 1.52),
        PocketSite("Glu791_CO", "HBA", np.array([8.0, 2.4, -0.8]), 1.52),
        PocketSite("Thr790_OG", "HBD", np.array([6.6, 4.8, -2.2]), 1.55),
        # Roof (+Z) and Floor (-Z) hydrophobic beta-sheet sandwich lining the adenine cleft
        PocketSite("Leu718_CD1", "HYDROPHOBIC", np.array([1.2, -1.8, 6.2]), 1.75),
        PocketSite("Leu718_CD2", "HYDROPHOBIC", np.array([-1.6, -2.4, 6.4]), 1.75),
        PocketSite("Val726_CG1", "HYDROPHOBIC", np.array([2.8, 2.6, 6.1]), 1.75),
        PocketSite("Val726_CG2", "HYDROPHOBIC", np.array([0.0, 3.0, 6.5]), 1.75),
        PocketSite("Ala743_CB", "HYDROPHOBIC", np.array([2.0, 0.6, -6.2]), 1.75),
        PocketSite("Leu844_CD1", "HYDROPHOBIC", np.array([-0.8, -1.0, -6.1]), 1.75),
        PocketSite("Leu844_CD2", "HYDROPHOBIC", np.array([-3.0, 1.8, -6.3]), 1.75),
        PocketSite("Ile744_CD", "HYDROPHOBIC", np.array([-5.6, 4.2, -5.2]), 1.75),
        PocketSite("Met766_CE", "HYDROPHOBIC", np.array([-6.8, -2.8, -5.0]), 1.75),
        PocketSite("Cys797_SG", "HYDROPHOBIC", np.array([5.4, -5.4, 3.2]), 1.80),
        PocketSite("Gly796_CA", "HYDROPHOBIC", np.array([4.6, -5.6, -1.6]), 1.75),
        PocketSite("Leu792_CD", "HYDROPHOBIC", np.array([5.4, 5.2, 2.8]), 1.75),
        # Aromatic gatekeeper & P-loop aromatic clamp
        PocketSite("Phe723_CG", "AROMATIC", np.array([-1.2, 5.4, 4.8]), 1.70),
        PocketSite("Phe723_CZ", "AROMATIC", np.array([-3.2, 5.8, 4.0]), 1.70),
        PocketSite("Tyr801_CG", "AROMATIC", np.array([5.0, -4.6, 4.2]), 1.70),
        # Catalytic Lys745 / DFG Asp855 polar subpocket
        PocketSite("Lys745_NZ", "HBD", np.array([-6.2, 4.4, 2.6]), 1.55),
        PocketSite("Asp855_OD1", "HBA", np.array([-6.4, -3.8, 2.2]), 1.52),
        PocketSite("Asp855_OD2", "HBA", np.array([-5.2, -5.2, -2.0]), 1.52),
        PocketSite("Asn842_ND2", "HBD", np.array([-2.2, -5.6, 3.0]), 1.55),
        PocketSite("Glu762_OE1", "HBA", np.array([-7.8, 2.0, -2.6]), 1.52),
        # Outer cavity boundary walls (solvent front on -X left open)
        PocketSite("BackWall_1", "WALL", np.array([12.0, 0.0, 0.0]), 1.80),
        PocketSite("SideWall_1", "WALL", np.array([0.0, 9.2, 0.0]), 1.80),
        PocketSite("SideWall_2", "WALL", np.array([0.0, -9.2, 0.0]), 1.80),
    )
    return ReceptorPocket(name="Kinase_ATP_Hinge", center=np.array([0.0, 0.0, 0.0]), sites=sites)


def _ideal_bond_length(mol: MoleculeGraph, u: int, v: int, order: BondOrder) -> float:
    r_u = ELEMENT_TABLE[mol.atoms[u].symbol].covalent_radius
    r_v = ELEMENT_TABLE[mol.atoms[v].symbol].covalent_radius
    base = r_u + r_v
    if order == BondOrder.AROMATIC:
        return base - 0.11
    if order == BondOrder.DOUBLE:
        return base - 0.18
    if order == BondOrder.TRIPLE:
        return base - 0.28
    return base


def _ideal_angle_rad(hyb: Hybridization) -> float:
    if hyb == Hybridization.SP:
        return math.pi
    if hyb == Hybridization.SP2:
        return math.radians(120.0)
    return math.radians(109.47)


def _compute_bounds_matrices(mol: MoleculeGraph) -> Tuple[np.ndarray, np.ndarray]:
    """Construct and triangle-smooth lower (L) and upper (U) pairwise distance bounds."""
    n = mol.num_atoms
    upper = np.full((n, n), 100.0, dtype=np.float64)
    lower = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(upper, 0.0)

    # Non-bonded vdW lower bounds
    for i in range(n):
        ri = ELEMENT_TABLE[mol.atoms[i].symbol].vdw_radius
        for j in range(i + 1, n):
            rj = ELEMENT_TABLE[mol.atoms[j].symbol].vdw_radius
            lb = 0.72 * (ri + rj)
            lower[i, j] = lower[j, i] = lb

    # 1-2 Covalent bond bounds
    for b in mol.bonds:
        u, v = b.begin_idx, b.end_idx
        d_ideal = _ideal_bond_length(mol, u, v, b.order)
        lower[u, v] = lower[v, u] = d_ideal - 0.04
        upper[u, v] = upper[v, u] = d_ideal + 0.04

    # 1-3 Bond angle bounds
    for center in range(n):
        nbrs = mol.neighbors(center)
        if len(nbrs) < 2:
            continue
        theta = _ideal_angle_rad(mol.atoms[center].hybridization)
        for i in range(len(nbrs)):
            u = nbrs[i]
            b_cu = mol.find_bond(center, u)
            assert b_cu is not None
            d1 = _ideal_bond_length(mol, center, u, b_cu.order)
            for j in range(i + 1, len(nbrs)):
                v = nbrs[j]
                if mol.find_bond(u, v) is not None:
                    continue
                b_cv = mol.find_bond(center, v)
                assert b_cv is not None
                d2 = _ideal_bond_length(mol, center, v, b_cv.order)
                d13 = math.sqrt(max(0.5, d1 * d1 + d2 * d2 - 2.0 * d1 * d2 * math.cos(theta)))
                tol = 0.25 if mol.atoms[center].in_ring else 0.14
                lower[u, v] = lower[v, u] = max(lower[u, v], d13 - tol)
                upper[u, v] = upper[v, u] = min(upper[u, v], d13 + tol)

    # Floyd-Warshall triangle inequality smoothing
    for k in range(n):
        u_ik = upper[:, k : k + 1]
        u_kj = upper[k : k + 1, :]
        upper = np.minimum(upper, u_ik + u_kj)
        l_ik = lower[:, k : k + 1]
        l_kj = lower[k : k + 1, :]
        lower = np.maximum(lower, np.maximum(l_ik - u_kj, l_kj - u_ik))

    lower = np.maximum(lower, 0.0)
    np.fill_diagonal(lower, 0.0)
    np.fill_diagonal(upper, 0.0)
    upper = np.maximum(upper, lower + 1e-4)
    np.fill_diagonal(upper, 0.0)
    return lower, upper


def _embed_metric_matrix(lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample distance matrix inside [L, U] and embed 3D coordinates via Gram eigendecomposition."""
    n = lower.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if n == 1:
        return np.zeros((1, 3), dtype=np.float64)

    rand_u = rng.uniform(0.15, 0.85, size=(n, n))
    rand_u = 0.5 * (rand_u + rand_u.T)
    dist = lower + rand_u * (upper - lower)
    np.fill_diagonal(dist, 0.0)

    d_sq = dist ** 2
    row_mean = d_sq.mean(axis=1, keepdims=True)
    col_mean = d_sq.mean(axis=0, keepdims=True)
    grand_mean = d_sq.mean()
    gram = -0.5 * (d_sq - row_mean - col_mean + grand_mean)

    eigvals, eigvecs = np.linalg.eigh(gram)
    top_idx = np.argsort(eigvals)[::-1][:3]
    vals = np.maximum(eigvals[top_idx], 1e-4)
    vecs = eigvecs[:, top_idx]
    coords = vecs * np.sqrt(vals)[None, :]
    if coords.shape[1] < 3:
        pad = np.zeros((n, 3 - coords.shape[1]), dtype=np.float64)
        coords = np.hstack([coords, pad])
    return coords.astype(np.float64)


def signed_chiral_volume(coords: np.ndarray, center_idx: int, nbrs: Sequence[int]) -> float:
    """Compute signed tetrahedral volume det([v1-v0, v2-v0, v3-v0]) for chiral restraint."""
    if len(nbrs) < 3:
        return 0.0
    p0 = coords[center_idx]
    v1 = coords[nbrs[0]] - p0
    v2 = coords[nbrs[1]] - p0
    v3 = coords[nbrs[2]] - p0
    return float(np.dot(np.cross(v1, v2), v3))


def generate_3d_conformer(
    mol: MoleculeGraph,
    seed: int = 42,
    max_opt_iter: int = 160,
) -> np.ndarray:
    """Generate a low-strain 3D conformer using Distance Geometry + BFGS force-field refinement.

    Enforces covalent bond lengths, 1-3 angle terms, ring planarity, non-bonded
    vdW repulsion, and tetrahedral chiral volume signs (@ vs @@).
    """
    n = mol.num_atoms
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    rng = np.random.default_rng(seed)
    lower, upper = _compute_bounds_matrices(mol)
    init_coords = _embed_metric_matrix(lower, upper, rng)

    # Precompute force-field terms
    bond_terms: List[Tuple[int, int, float]] = [
        (b.begin_idx, b.end_idx, _ideal_bond_length(mol, b.begin_idx, b.end_idx, b.order))
        for b in mol.bonds
    ]
    angle_terms: List[Tuple[int, int, int, float]] = []
    for c in range(n):
        nbrs = mol.neighbors(c)
        if len(nbrs) >= 2:
            cos_t = math.cos(_ideal_angle_rad(mol.atoms[c].hybridization))
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    angle_terms.append((nbrs[i], c, nbrs[j], cos_t))

    # Chiral centers with explicit @ or @@
    chiral_terms: List[Tuple[int, Tuple[int, ...], float]] = []
    for a in mol.atoms:
        if a.stereo in (StereoConfig.TETRAHEDRAL_CCW, StereoConfig.TETRAHEDRAL_CW):
            nbrs = sorted(mol.neighbors(a.index))
            if len(nbrs) >= 3:
                target_sign = -1.0 if a.stereo == StereoConfig.TETRAHEDRAL_CCW else 1.0
                chiral_terms.append((a.index, tuple(nbrs[:3]), target_sign * 2.2))

    # Aromatic ring planarity quadruplets
    planar_rings = [
        ring for ring in mol.sssr_rings if len(ring) >= 5 and all(mol.atoms[idx].is_aromatic for idx in ring)
    ]

    bonded_pairs = {(min(u, v), max(u, v)) for u, v, _ in bond_terms}
    nonbonded_pairs: List[Tuple[int, int, float]] = []
    for i in range(n):
        ri = ELEMENT_TABLE[mol.atoms[i].symbol].vdw_radius
        for j in range(i + 1, n):
            if (i, j) not in bonded_pairs:
                rj = ELEMENT_TABLE[mol.atoms[j].symbol].vdw_radius
                nonbonded_pairs.append((i, j, 0.78 * (ri + rj)))

    b_u = np.asarray([t[0] for t in bond_terms], dtype=np.int64) if bond_terms else np.zeros(0, dtype=np.int64)
    b_v = np.asarray([t[1] for t in bond_terms], dtype=np.int64) if bond_terms else np.zeros(0, dtype=np.int64)
    b_d0 = np.asarray([t[2] for t in bond_terms], dtype=np.float64) if bond_terms else np.zeros(0, dtype=np.float64)

    a_u = np.asarray([t[0] for t in angle_terms], dtype=np.int64) if angle_terms else np.zeros(0, dtype=np.int64)
    a_c = np.asarray([t[1] for t in angle_terms], dtype=np.int64) if angle_terms else np.zeros(0, dtype=np.int64)
    a_v = np.asarray([t[2] for t in angle_terms], dtype=np.int64) if angle_terms else np.zeros(0, dtype=np.int64)
    a_cos0 = np.asarray([t[3] for t in angle_terms], dtype=np.float64) if angle_terms else np.zeros(0, dtype=np.float64)

    nb_i = (
        np.asarray([t[0] for t in nonbonded_pairs], dtype=np.int64)
        if nonbonded_pairs
        else np.zeros(0, dtype=np.int64)
    )
    nb_j = (
        np.asarray([t[1] for t in nonbonded_pairs], dtype=np.int64)
        if nonbonded_pairs
        else np.zeros(0, dtype=np.int64)
    )
    nb_min = (
        np.asarray([t[2] for t in nonbonded_pairs], dtype=np.float64)
        if nonbonded_pairs
        else np.zeros(0, dtype=np.float64)
    )

    def _strain_energy_and_grad(flat: np.ndarray) -> Tuple[float, np.ndarray]:
        xyz = flat.reshape(n, 3)
        energy = 0.0
        grad = np.zeros((n, 3), dtype=np.float64)

        if len(b_u) > 0:
            diff_b = xyz[b_u] - xyz[b_v]
            r_b = np.sqrt(np.sum(diff_b * diff_b, axis=1) + 1e-9)
            err_b = r_b - b_d0
            energy += 180.0 * float(np.sum(err_b * err_b))
            g_b = (360.0 * err_b / r_b)[:, None] * diff_b
            np.add.at(grad, b_u, g_b)
            np.add.at(grad, b_v, -g_b)

        if len(a_u) > 0:
            vu = xyz[a_u] - xyz[a_c]
            vv = xyz[a_v] - xyz[a_c]
            nu2 = np.sum(vu * vu, axis=1) + 1e-9
            nv2 = np.sum(vv * vv, axis=1) + 1e-9
            nu = np.sqrt(nu2)
            nv = np.sqrt(nv2)
            inv_prod = 1.0 / (nu * nv)
            dot_uv = np.sum(vu * vv, axis=1)
            cos_cur = dot_uv * inv_prod
            err_a = cos_cur - a_cos0
            energy += 35.0 * float(np.sum(err_a * err_a))
            coef = (70.0 * err_a)[:, None]
            g_u = coef * (vv * inv_prod[:, None] - (cos_cur / nu2)[:, None] * vu)
            g_v = coef * (vu * inv_prod[:, None] - (cos_cur / nv2)[:, None] * vv)
            np.add.at(grad, a_u, g_u)
            np.add.at(grad, a_v, g_v)
            np.add.at(grad, a_c, -(g_u + g_v))

        for ring in planar_rings:
            r_idx = list(ring)
            pts = xyz[r_idx]
            centered = pts - pts.mean(axis=0, keepdims=True)
            cov = centered.T @ centered
            eigvals, eigvecs = np.linalg.eigh(cov)
            l0 = float(max(0.0, eigvals[0]))
            energy += 80.0 * l0
            if l0 > 0.0:
                n0 = eigvecs[:, 0]
                proj = centered @ n0
                grad[r_idx] += 160.0 * proj[:, None] * n0[None, :]

        if len(nb_i) > 0:
            diff_nb = xyz[nb_i] - xyz[nb_j]
            r_nb = np.sqrt(np.sum(diff_nb * diff_nb, axis=1) + 1e-9)
            overlap = np.maximum(0.0, nb_min - r_nb)
            energy += 50.0 * float(np.sum(overlap * overlap))
            active = overlap > 0.0
            if np.any(active):
                g_nb = (-100.0 * overlap[active] / r_nb[active])[:, None] * diff_nb[active]
                np.add.at(grad, nb_i[active], g_nb)
                np.add.at(grad, nb_j[active], -g_nb)

        for c_idx, nbrs_t, target_vol in chiral_terms:
            p0 = xyz[c_idx]
            v1 = xyz[nbrs_t[0]] - p0
            v2 = xyz[nbrs_t[1]] - p0
            v3 = xyz[nbrs_t[2]] - p0
            vol = float(np.dot(np.cross(v1, v2), v3))
            if (target_vol > 0 and vol < target_vol) or (target_vol < 0 and vol > target_vol):
                err_v = vol - target_vol
                energy += 60.0 * err_v * err_v
                w = 120.0 * err_v
                gv1 = w * np.cross(v2, v3)
                gv2 = w * np.cross(v3, v1)
                gv3 = w * np.cross(v1, v2)
                grad[nbrs_t[0]] += gv1
                grad[nbrs_t[1]] += gv2
                grad[nbrs_t[2]] += gv3
                grad[c_idx] -= gv1 + gv2 + gv3

        return energy, grad.ravel()

    # Ensure initial chiral sign matches before local optimization
    for c_idx, nbrs_t, target_vol in chiral_terms:
        vol = signed_chiral_volume(init_coords, c_idx, nbrs_t)
        if vol * target_vol < 0:
            init_coords[:, 2] *= -1.0
            break

    res = minimize(
        _strain_energy_and_grad,
        init_coords.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_opt_iter, "ftol": 1e-6},
    )
    opt_coords = np.asarray(res.x.reshape(n, 3), dtype=np.float64)
    opt_coords -= opt_coords.mean(axis=0, keepdims=True)
    mol.coords = opt_coords
    return opt_coords


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Euler XYZ 3x3 rotation matrix."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz_m @ ry_m @ rx_m


def _rodrigues_rotate(points: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 3D points around an axis passing through origin using Rodrigues' formula."""
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return points
    k = axis / norm
    rel = points - origin
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cross = np.cross(k, rel)
    dot = np.dot(rel, k)[:, None]
    rotated = rel * cos_a + cross * sin_a + k[None, :] * dot * (1.0 - cos_a)
    return np.asarray(origin + rotated, dtype=np.float64)


def _find_rotatable_torsions(mol: MoleculeGraph) -> List[Tuple[int, int, List[int]]]:
    """Identify rotatable single bonds and the downstream mobile atom subgraph."""
    torsions: List[Tuple[int, int, List[int]]] = []
    for b in mol.bonds:
        if b.in_ring or b.is_aromatic or b.order != BondOrder.SINGLE:
            continue
        u, v = b.begin_idx, b.end_idx
        if mol.degree(u) <= 1 or mol.degree(v) <= 1:
            continue
        # BFS from v without crossing u
        seen = {v}
        stack = [v]
        while stack:
            curr = stack.pop()
            for nbr in mol.neighbors(curr):
                if nbr != u and nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        if 1 < len(seen) < mol.num_atoms - 1:
            torsions.append((u, v, sorted(seen)))
    return torsions[:6]


def _build_scoring_arrays(
    mol: MoleculeGraph,
    pocket: ReceptorPocket,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    """Precompute vectorized atom/site feature masks for fast empirical pose scoring."""
    heavy_ids = [a.index for a in mol.atoms if a.atomic_number > 1]
    lig_vdw = np.asarray([ELEMENT_TABLE[mol.atoms[i].symbol].vdw_radius for i in heavy_ids], dtype=np.float64)
    lig_hbd = np.asarray(
        [mol.atoms[i].symbol in ("N", "O") and mol.atoms[i].total_h > 0 for i in heavy_ids], dtype=bool
    )
    lig_hba = np.asarray(
        [mol.atoms[i].symbol in ("N", "O") and mol.atoms[i].formal_charge <= 0 for i in heavy_ids], dtype=bool
    )
    lig_hydro = np.asarray(
        [
            mol.atoms[i].symbol in ("C", "S", "Cl", "Br", "I", "F")
            and not any(mol.atoms[nbr].symbol in ("N", "O") for nbr in mol.neighbors(i))
            for i in heavy_ids
        ],
        dtype=bool,
    )
    lig_arom = np.asarray([mol.atoms[i].is_aromatic for i in heavy_ids], dtype=bool)

    site_xyz = np.vstack([s.coord for s in pocket.sites])
    site_vdw = np.asarray([s.vdw_radius for s in pocket.sites], dtype=np.float64)
    site_roles = np.asarray([s.role for s in pocket.sites])
    heavy_idx_arr = np.asarray(heavy_ids, dtype=np.int64)
    n_rot = len(_find_rotatable_torsions(mol))
    return (
        heavy_idx_arr,
        lig_vdw,
        lig_hbd,
        lig_hba,
        lig_hydro,
        lig_arom,
        site_xyz,
        site_vdw,
        site_roles,
        n_rot,
    )


def _score_vectorized(
    coords: np.ndarray,
    heavy_idx_arr: np.ndarray,
    lig_vdw: np.ndarray,
    lig_hbd: np.ndarray,
    lig_hba: np.ndarray,
    lig_hydro: np.ndarray,
    lig_arom: np.ndarray,
    site_xyz: np.ndarray,
    site_vdw: np.ndarray,
    site_roles: np.ndarray,
    n_rot: int,
    pocket: ReceptorPocket,
    strain_energy: float = 0.0,
) -> Tuple[float, float, float, float, float, float, float, float, List[Tuple[int, str]]]:
    if len(heavy_idx_arr) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, []

    lig_xyz = coords[heavy_idx_arr]
    diff = lig_xyz[:, None, :] - site_xyz[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
    mask = dist <= 8.0
    d = dist - lig_vdw[:, None] - site_vdw[None, :]

    g1 = float(np.sum(np.where(mask, np.exp(-((d / 0.5) ** 2)), 0.0)))
    g2 = float(np.sum(np.where(mask, np.exp(-(((d - 3.0) / 2.0) ** 2)), 0.0)))
    rep_mat = np.where(mask & (d < 0.0), d * d, 0.0)
    rep = float(np.sum(rep_mat))

    hydro_mask = mask & lig_hydro[:, None] & (site_roles[None, :] == "HYDROPHOBIC")
    hydro_vals = np.where(d <= 0.5, 1.0, np.where(d < 1.5, 1.5 - d, 0.0))
    hydro = float(np.sum(np.where(hydro_mask, hydro_vals, 0.0)))

    hb_pair = (lig_hba[:, None] & (site_roles[None, :] == "HBD")) | (
        lig_hbd[:, None] & (site_roles[None, :] == "HBA")
    )
    hb_mask = mask & hb_pair & (d < 0.35)
    hb_vals = np.where(d <= -0.7, 1.0, (0.35 - d) / 1.05)
    hb_mat = np.where(hb_mask, hb_vals, 0.0)
    hbond = float(np.sum(hb_mat))

    pi_mask = mask & lig_arom[:, None] & (site_roles[None, :] == "AROMATIC") & (d < 0.8)
    pi_vals = np.maximum(0.0, 1.0 - np.abs(d) / 0.8)
    pi_stack = float(np.sum(np.where(pi_mask, pi_vals, 0.0)))

    contacts: List[Tuple[int, str]] = []
    hit_rows, hit_cols = np.where(hb_mat > 0.35)
    for r_i, c_j in zip(hit_rows[:12], hit_cols[:12]):
        contacts.append((int(heavy_idx_arr[int(r_i)]), pocket.sites[int(c_j)].residue))

    # Coarse-grained residue-anchor multiplicity (~2.8 heavy atoms per residue site)
    site_mult = 2.8
    raw_energy = site_mult * (
        -0.035579 * g1
        - 0.005156 * g2
        + 0.840245 * rep
        - 0.035069 * hydro
        - 0.587439 * hbond
        - 0.140000 * pi_stack
    )
    torsion_divisor = 1.0 + 0.05846 * n_rot
    total_kcal = (raw_energy / torsion_divisor) + 0.05 * strain_energy
    torsion_penalty = abs(raw_energy) * (1.0 - 1.0 / torsion_divisor)
    return (
        float(total_kcal),
        site_mult * (-0.035579 * g1),
        site_mult * (-0.005156 * g2),
        site_mult * (0.840245 * rep),
        site_mult * (-0.035069 * hydro),
        site_mult * (-0.587439 * hbond),
        site_mult * (-0.140000 * pi_stack),
        float(torsion_penalty),
        contacts,
    )


def score_ligand_pose(
    mol: MoleculeGraph,
    coords: np.ndarray,
    pocket: ReceptorPocket,
    strain_energy: float = 0.0,
) -> DockingPose:
    """Score a 3D ligand pose in ReceptorPocket using Vina-inspired empirical terms (kcal/mol)."""
    arrays = _build_scoring_arrays(mol, pocket)
    (
        total_kcal,
        g1_s,
        g2_s,
        rep_s,
        hydro_s,
        hb_s,
        pi_s,
        tor_p,
        contacts,
    ) = _score_vectorized(coords, *arrays, pocket=pocket, strain_energy=strain_energy)

    return DockingPose(
        smiles=mol.canonical_smiles or mol.source_smiles,
        coords=coords.copy(),
        total_affinity_kcal=total_kcal,
        gauss1_score=g1_s,
        gauss2_score=g2_s,
        repulsion_score=rep_s,
        hydrophobic_score=hydro_s,
        hbond_score=hb_s,
        pi_stacking_score=pi_s,
        torsion_penalty=tor_p,
        hbond_contacts=contacts,
        strain_energy=float(strain_energy),
    )


def dock_molecule(
    mol: MoleculeGraph,
    pocket: Optional[ReceptorPocket] = None,
    steps: int = 80,
    seed: int = 42,
) -> DockingPose:
    """Generate 3D conformer and optimize rigid + torsion pose inside ReceptorPocket."""
    target_pocket = pocket if pocket is not None else build_kinase_hinge_pocket()
    base_coords = generate_3d_conformer(mol, seed=seed, max_opt_iter=45)
    if mol.num_atoms == 0:
        return score_ligand_pose(mol, base_coords, target_pocket)

    torsions = _find_rotatable_torsions(mol)[:3]
    arrays = _build_scoring_arrays(mol, target_pocket)
    rng = np.random.default_rng(seed)

    # Align ligand principal axes to pocket cleft axes (largest span along X)
    centered = base_coords - base_coords.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    aligned_coords = centered @ vt.T
    if np.linalg.det(vt) < 0:
        aligned_coords[:, 2] *= -1.0

    def _apply_pose(vec: np.ndarray) -> np.ndarray:
        xyz = aligned_coords.copy()
        for t_idx, (u, v, mobile) in enumerate(torsions):
            angle = float(vec[6 + t_idx])
            if abs(angle) > 1e-4:
                xyz[mobile] = _rodrigues_rotate(xyz[mobile], xyz[u], xyz[v] - xyz[u], angle)
        rot = _rotation_matrix(float(vec[3]), float(vec[4]), float(vec[5]))
        t_vec = np.clip(vec[:3], -2.8, 2.8)
        xyz = np.asarray(xyz @ rot.T + target_pocket.center + t_vec, dtype=np.float64)
        return xyz

    def _objective(vec: np.ndarray) -> float:
        xyz = _apply_pose(vec)
        total_kcal = _score_vectorized(xyz, *arrays, pocket=target_pocket, strain_energy=0.0)[0]
        out_of_box = float(np.sum(np.maximum(0.0, np.abs(vec[:3]) - 2.8) ** 2))
        return total_kcal + 2.0 * out_of_box

    dim = 6 + len(torsions)
    best_vec = np.zeros(dim, dtype=np.float64)
    best_val = _objective(best_vec)

    # Multi-start Monte Carlo screening followed by local Powell refinement
    n_random = max(48, steps)
    for _ in range(n_random):
        cand = np.zeros(dim, dtype=np.float64)
        cand[:3] = rng.uniform(-2.0, 2.0, size=3)
        cand[3:6] = rng.uniform(-math.pi, math.pi, size=3)
        if torsions:
            cand[6:] = rng.uniform(-math.pi, math.pi, size=len(torsions))
        val = _objective(cand)
        if val < best_val:
            best_val = val
            best_vec = cand

    bounds = [(-2.8, 2.8)] * 3 + [(-2.0 * math.pi, 2.0 * math.pi)] * (3 + len(torsions))
    res = minimize(
        _objective,
        best_vec,
        method="Powell",
        bounds=bounds,
        options={"maxfev": max(160, steps * 5), "ftol": 1e-2},
    )
    if float(res.fun) < best_val:
        best_vec = res.x.copy()

    final_coords = _apply_pose(best_vec)
    mol.coords = final_coords
    return score_ligand_pose(mol, final_coords, target_pocket)
