"""Circular Morgan/ECFP/FCFP fingerprints, similarity metrics, Butina clustering, and MMP cliff analysis."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple
import numpy as np

from chiral_weave.chem_types import Atom, Bond, BondOrder, MoleculeGraph
from chiral_weave.smiles_parser import has_smarts_match, to_canonical_smiles


def _mix64(value: int) -> int:
    """SplitMix64 deterministic 64-bit integer avalanche mixer."""
    z = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def _hash_tuple(items: Sequence[int]) -> int:
    """Deterministic 64-bit hash over an integer tuple."""
    h = 0xCBF29CE484222325
    for item in items:
        h ^= item & 0xFFFFFFFFFFFFFFFF
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        h = _mix64(h)
    return h


def _morgan_atom_invariant(atom: Atom, mol: MoleculeGraph, feature_mode: bool) -> int:
    """Compute initial integer atom invariant for ECFP (topological) or FCFP (pharmacophoric)."""
    if not feature_mode:
        return _hash_tuple(
            (
                atom.atomic_number,
                mol.degree(atom.index),
                atom.total_h,
                atom.formal_charge + 16,
                1 if atom.in_ring else 0,
                1 if atom.is_aromatic else 0,
            )
        )
    is_hbd = 1 if (atom.symbol in ("N", "O") and atom.total_h > 0) else 0
    is_hba = 1 if (atom.symbol in ("N", "O") and atom.formal_charge <= 0) else 0
    is_pos = 1 if (atom.formal_charge > 0 or (atom.symbol == "N" and not atom.is_aromatic)) else 0
    is_neg = 1 if atom.formal_charge < 0 else 0
    is_arom = 1 if atom.is_aromatic else 0
    is_hal = 1 if atom.symbol in ("F", "Cl", "Br", "I") else 0
    return _hash_tuple((is_hbd, is_hba, is_pos, is_neg, is_arom, is_hal))


def compute_morgan_counts(
    mol: MoleculeGraph,
    radius: int = 2,
    use_features: bool = False,
) -> Dict[int, int]:
    """Compute sparse Morgan circular environment identifier counts (ECFP/FCFP)."""
    if mol.num_atoms == 0:
        return {}

    identifiers = [_morgan_atom_invariant(a, mol, use_features) for a in mol.atoms]
    counts: Dict[int, int] = {}
    for ident in identifiers:
        counts[ident] = counts.get(ident, 0) + 1

    atom_envs: List[Set[int]] = [{b.index for b in mol.incident_bonds(a.index)} for a in mol.atoms]
    seen_env_signatures: Set[Tuple[int, ...]] = set()

    for r in range(1, radius + 1):
        new_identifiers = list(identifiers)
        new_envs: List[Set[int]] = [set(s) for s in atom_envs]
        for a_idx in range(mol.num_atoms):
            nbr_pairs: List[Tuple[int, int]] = []
            for b in mol.incident_bonds(a_idx):
                nbr = b.other(a_idx)
                bo_code = int(round(b.order.value * 2))
                nbr_pairs.append((bo_code, identifiers[nbr]))
                new_envs[a_idx].update(atom_envs[nbr])
            nbr_pairs.sort()
            flat: List[int] = [r, identifiers[a_idx]]
            for bo_c, nid in nbr_pairs:
                flat.append(bo_c)
                flat.append(nid)
            new_id = _hash_tuple(flat)
            new_identifiers[a_idx] = new_id
            env_sig = tuple(sorted(new_envs[a_idx]))
            if env_sig not in seen_env_signatures:
                seen_env_signatures.add(env_sig)
                counts[new_id] = counts.get(new_id, 0) + 1
        identifiers = new_identifiers
        atom_envs = new_envs

    return counts


def compute_morgan_fingerprint(
    mol: MoleculeGraph,
    radius: int = 2,
    nbits: int = 1024,
    use_features: bool = False,
) -> np.ndarray:
    """Fold sparse Morgan circular identifiers into a binary bitvector of length nbits."""
    bits = np.zeros(nbits, dtype=np.uint8)
    counts = compute_morgan_counts(mol, radius=radius, use_features=use_features)
    for ident in counts:
        bits[ident % nbits] = 1
    return bits


def compute_atom_pair_fingerprint(mol: MoleculeGraph, nbits: int = 512) -> np.ndarray:
    """Compute shortest-path Atom-Pair fingerprint encoding (atom1, dist, atom2) triplets."""
    bits = np.zeros(nbits, dtype=np.uint8)
    n = mol.num_atoms
    if n < 2:
        return bits

    def _atom_code(atom: Atom) -> int:
        return (atom.atomic_number << 6) | (mol.degree(atom.index) << 3) | (1 if atom.is_aromatic else 0)

    codes = [_atom_code(a) for a in mol.atoms]
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in mol.neighbors(u):
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for dst in range(src + 1, n):
            if dist[dst] > 0:
                a, b = min(codes[src], codes[dst]), max(codes[src], codes[dst])
                h = _hash_tuple((a, dist[dst], b))
                bits[h % nbits] = 1
    return bits


_MACCS_SMARTS: Tuple[str, ...] = (
    "[#7]",               # 0: any nitrogen
    "[#8]",               # 1: any oxygen
    "[#16]",              # 2: any sulfur
    "[F,Cl,Br,I]",        # 3: halogen
    "[#6]=[#8]",          # 4: carbonyl
    "[#7]-[C]=[O]",       # 5: amide
    "[#8H1]",             # 6: hydroxyl
    "[#7H1,#7H2]",        # 7: NH or NH2
    "[a]",                # 8: aromatic atom
    "[r5]",               # 9: 5-membered ring
    "[r6]",               # 10: 6-membered ring
    "[#6]#[#7]",          # 11: nitrile
    "[#16](=[#8])=[#8]",  # 12: sulfonyl
    "[#6]=[#6]",          # 13: alkene
    "[#6]#[#6]",          # 14: alkyne
    "[#6]-O-[#6]",        # 15: ether
    "[#7]-[#7]",          # 16: N-N bond
    "[F]",                # 17: fluorine
    "[Cl]",               # 18: chlorine
    "[Br,I]",             # 19: heavy halogen
    "[n]",                # 20: aromatic nitrogen (heterocycle)
    "[o,s]",              # 21: aromatic O or S
    "[#6](=[O])[O]",      # 22: carboxyl/ester
    "[CX4]",              # 23: sp3 carbon
)


def compute_maccs_keys(mol: MoleculeGraph) -> np.ndarray:
    """Evaluate 24 medicinal-chemistry SMARTS structural keys."""
    vec = np.zeros(len(_MACCS_SMARTS), dtype=np.uint8)
    for idx, smarts in enumerate(_MACCS_SMARTS):
        if has_smarts_match(mol, smarts):
            vec[idx] = 1
    return vec


def tanimoto_similarity(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Compute exact Tanimoto (Jaccard) similarity between two binary fingerprints."""
    a_bool = fp_a.astype(bool)
    b_bool = fp_b.astype(bool)
    inter = int(np.logical_and(a_bool, b_bool).sum())
    union = int(np.logical_or(a_bool, b_bool).sum())
    if union == 0:
        return 1.0
    return float(inter / union)


def dice_similarity(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Compute Dice (Sorensen) similarity between two binary fingerprints."""
    a_bool = fp_a.astype(bool)
    b_bool = fp_b.astype(bool)
    inter = int(np.logical_and(a_bool, b_bool).sum())
    total = int(a_bool.sum()) + int(b_bool.sum())
    if total == 0:
        return 1.0
    return float((2.0 * inter) / total)


def tversky_similarity(fp_a: np.ndarray, fp_b: np.ndarray, alpha: float = 0.5, beta: float = 0.5) -> float:
    """Compute asymmetric Tversky index between fp_a (reference) and fp_b (query)."""
    a_bool = fp_a.astype(bool)
    b_bool = fp_b.astype(bool)
    inter = float(np.logical_and(a_bool, b_bool).sum())
    only_a = float(np.logical_and(a_bool, np.logical_not(b_bool)).sum())
    only_b = float(np.logical_and(b_bool, np.logical_not(a_bool)).sum())
    denom = inter + alpha * only_a + beta * only_b
    if denom == 0.0:
        return 1.0
    return float(inter / denom)


def pairwise_tanimoto_matrix(fps: Sequence[np.ndarray]) -> np.ndarray:
    """Compute vectorized N x N Tanimoto similarity matrix for a library of fingerprints."""
    if not fps:
        return np.zeros((0, 0), dtype=np.float64)
    mat = np.asarray(fps, dtype=np.float64)
    inter = mat @ mat.T
    counts = mat.sum(axis=1, keepdims=True)
    union = counts + counts.T - inter
    sim = np.where(union > 0.0, inter / np.maximum(union, 1e-12), 1.0)
    np.fill_diagonal(sim, 1.0)
    return sim


@dataclass(frozen=True)
class ClusterAssignment:
    """Result of Taylor-Butina sphere-exclusion clustering."""

    clusters: List[List[int]]
    centroids: List[int]
    similarity_threshold: float
    singletons: int


def butina_cluster(fps: Sequence[np.ndarray], similarity_threshold: float = 0.65) -> ClusterAssignment:
    """Cluster molecules using Taylor-Butina leader sphere exclusion.

    Molecules are ranked by neighbor density within Tanimoto >= threshold and
    greedily assigned to unclustered centroids.
    """
    n = len(fps)
    if n == 0:
        return ClusterAssignment(clusters=[], centroids=[], similarity_threshold=similarity_threshold, singletons=0)

    sim_mat = pairwise_tanimoto_matrix(fps)
    neighbor_lists: List[List[int]] = []
    for i in range(n):
        nbrs = [j for j in range(n) if j != i and sim_mat[i, j] >= similarity_threshold]
        neighbor_lists.append(nbrs)

    order = sorted(range(n), key=lambda i: (-len(neighbor_lists[i]), i))
    assigned = [False] * n
    clusters: List[List[int]] = []
    centroids: List[int] = []

    for seed in order:
        if assigned[seed]:
            continue
        members = [seed]
        assigned[seed] = True
        for nbr in neighbor_lists[seed]:
            if not assigned[nbr]:
                assigned[nbr] = True
                members.append(nbr)
        clusters.append(members)
        centroids.append(seed)

    singletons = sum(1 for c in clusters if len(c) == 1)
    return ClusterAssignment(
        clusters=clusters,
        centroids=centroids,
        similarity_threshold=similarity_threshold,
        singletons=singletons,
    )


def maxmin_diversity_pick(fps: Sequence[np.ndarray], k: int, seed_idx: int = 0) -> List[int]:
    """Select k maximally diverse compound indices via greedy MaxMin Tanimoto distance."""
    n = len(fps)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    sim_mat = pairwise_tanimoto_matrix(fps)
    dist_mat = 1.0 - sim_mat

    picked = [seed_idx]
    min_dist = dist_mat[seed_idx].copy()
    while len(picked) < k:
        for p in picked:
            min_dist[p] = -1.0
        next_idx = int(np.argmax(min_dist))
        picked.append(next_idx)
        min_dist = np.minimum(min_dist, dist_mat[next_idx])
    return picked


@dataclass(frozen=True)
class MatchedMolecularPair:
    """Single-cut Matched Molecular Pair (MMP) transformation and activity cliff metric."""

    idx_a: int
    idx_b: int
    smiles_a: str
    smiles_b: str
    core_smiles: str
    r_group_a: str
    r_group_b: str
    delta_activity: float
    tanimoto_sim: float
    sali_score: float


def _fragment_single_cut(mol: MoleculeGraph, bond: Bond) -> Tuple[str, str]:
    """Split MoleculeGraph along a non-ring single bond and return sorted (frag1, frag2) SMILES."""
    adj: Dict[int, List[int]] = {a.index: [] for a in mol.atoms}
    for b in mol.bonds:
        if b.index == bond.index:
            continue
        adj[b.begin_idx].append(b.end_idx)
        adj[b.end_idx].append(b.begin_idx)

    def _bfs(start: int) -> Set[int]:
        seen = {start}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return seen

    comp1 = _bfs(bond.begin_idx)
    comp2 = set(range(mol.num_atoms)) - comp1

    def _sub_smiles(indices: Set[int]) -> str:
        sorted_ids = sorted(indices)
        id_map = {old: new for new, old in enumerate(sorted_ids)}
        sub_atoms = [
            Atom(
                index=id_map[old],
                symbol=mol.atoms[old].symbol,
                atomic_number=mol.atoms[old].atomic_number,
                formal_charge=mol.atoms[old].formal_charge,
                explicit_h=mol.atoms[old].explicit_h,
                implicit_h=mol.atoms[old].implicit_h,
                is_aromatic=mol.atoms[old].is_aromatic,
                hybridization=mol.atoms[old].hybridization,
                in_ring=mol.atoms[old].in_ring,
                ring_sizes=mol.atoms[old].ring_sizes,
            )
            for old in sorted_ids
        ]
        sub_bonds = [
            Bond(
                index=len([]),
                begin_idx=id_map[b.begin_idx],
                end_idx=id_map[b.end_idx],
                order=b.order,
                is_aromatic=b.is_aromatic,
                in_ring=b.in_ring,
            )
            for b in mol.bonds
            if b.index != bond.index and b.begin_idx in id_map and b.end_idx in id_map
        ]
        for idx, sb in enumerate(sub_bonds):
            sb.index = idx
        return to_canonical_smiles(MoleculeGraph(atoms=sub_atoms, bonds=sub_bonds))

    s1 = _sub_smiles(comp1)
    s2 = _sub_smiles(comp2)
    return (s1, s2) if len(comp1) >= len(comp2) else (s2, s1)


def extract_mmp_pairs(
    mols: Sequence[MoleculeGraph],
    activities: Sequence[float],
    min_core_heavy_atoms: int = 6,
) -> List[MatchedMolecularPair]:
    """Identify Matched Molecular Pairs via single exocyclic bond fragmentation and compute SALI."""
    if len(mols) != len(activities):
        raise ValueError("mols and activities must have equal length.")

    fps = [compute_morgan_fingerprint(m, radius=2, nbits=1024) for m in mols]
    core_index: Dict[str, List[Tuple[int, str]]] = {}

    for mol_idx, mol in enumerate(mols):
        for bond in mol.bonds:
            if bond.in_ring or bond.is_aromatic or bond.order != BondOrder.SINGLE:
                continue
            core_smi, r_smi = _fragment_single_cut(mol, bond)
            if not core_smi or len(core_smi) < min_core_heavy_atoms:
                continue
            core_index.setdefault(core_smi, []).append((mol_idx, r_smi))

    seen_pairs: Set[Tuple[int, int]] = set()
    results: List[MatchedMolecularPair] = []

    for core_smi, entries in core_index.items():
        if len(entries) < 2:
            continue
        for i in range(len(entries)):
            idx_a, r_a = entries[i]
            for j in range(i + 1, len(entries)):
                idx_b, r_b = entries[j]
                if idx_a == idx_b or r_a == r_b:
                    continue
                pair_key = (min(idx_a, idx_b), max(idx_a, idx_b))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                sim = tanimoto_similarity(fps[idx_a], fps[idx_b])
                delta = float(activities[idx_b] - activities[idx_a])
                sali = abs(delta) / max(0.02, 1.0 - sim)
                results.append(
                    MatchedMolecularPair(
                        idx_a=idx_a,
                        idx_b=idx_b,
                        smiles_a=mols[idx_a].canonical_smiles or mols[idx_a].source_smiles,
                        smiles_b=mols[idx_b].canonical_smiles or mols[idx_b].source_smiles,
                        core_smiles=core_smi,
                        r_group_a=r_a,
                        r_group_b=r_b,
                        delta_activity=delta,
                        tanimoto_sim=sim,
                        sali_score=sali,
                    )
                )

    results.sort(key=lambda p: -p.sali_score)
    return results
