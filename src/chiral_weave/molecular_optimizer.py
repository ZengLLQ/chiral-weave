"""Valence-safe molecular graph mutator, MCTS lead optimizer, and NSGA-II Pareto selector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple
import math
import numpy as np

from chiral_weave.adme_kinetics import AdmePredictor
from chiral_weave.chem_types import (
    ELEMENT_TABLE,
    Atom,
    Bond,
    BondOrder,
    MoleculeGraph,
    OptimizationCandidate,
)
from chiral_weave.conformer_docking import ReceptorPocket, build_kinase_hinge_pocket, dock_molecule
from chiral_weave.descriptors import compute_physicochemical_profile
from chiral_weave.fingerprints import compute_morgan_fingerprint, tanimoto_similarity
from chiral_weave.smiles_parser import (
    SmilesParseError,
    _assign_valences_and_hybridization,
    parse_smiles,
    to_canonical_smiles,
)


_TERMINAL_FRAGMENTS: Tuple[str, ...] = (
    "F",
    "Cl",
    "C",
    "O",
    "N",
    "OC",
    "C(F)(F)F",
    "C#N",
    "C(=O)N",
)


@dataclass
class _MctsNode:
    smiles: str
    depth: int
    parent_smiles: str
    visits: int = 0
    total_value: float = 0.0
    children: List[_MctsNode] = field(default_factory=list)
    untried_smiles: Optional[List[str]] = None

    def ucb1(self, parent_visits: int, c_explore: float = 0.35) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.total_value / self.visits
        explore = c_explore * math.sqrt(math.log(max(1, parent_visits)) / self.visits)
        return exploit + explore


@dataclass(frozen=True)
class OptimizationReport:
    """Summary of an MCTS + Pareto multi-objective de novo lead optimization campaign."""

    seed_smiles: str
    iterations: int
    unique_evaluated: int
    best_candidate: OptimizationCandidate
    pareto_front: List[OptimizationCandidate]
    top_candidates: List[OptimizationCandidate]
    mpo_history: List[float]


def _attach_fragment(base_mol: MoleculeGraph, anchor_idx: int, frag_smiles: str) -> Optional[str]:
    """Attach a parsed fragment root (atom 0) to anchor_idx via a single bond."""
    frag = parse_smiles(frag_smiles)
    combined = base_mol.copy()
    offset = combined.num_atoms

    anchor = combined.atoms[anchor_idx]
    if anchor.implicit_h < 1:
        return None
    anchor.implicit_h -= 1

    for fa in frag.atoms:
        combined.atoms.append(
            Atom(
                index=offset + fa.index,
                symbol=fa.symbol,
                atomic_number=fa.atomic_number,
                formal_charge=fa.formal_charge,
                explicit_h=fa.explicit_h,
                implicit_h=fa.implicit_h - (1 if fa.index == 0 else 0),
                is_aromatic=fa.is_aromatic,
                hybridization=fa.hybridization,
                in_ring=fa.in_ring,
                ring_sizes=fa.ring_sizes,
            )
        )
    for fb in frag.bonds:
        combined.bonds.append(
            Bond(
                index=len(combined.bonds),
                begin_idx=offset + fb.begin_idx,
                end_idx=offset + fb.end_idx,
                order=fb.order,
                is_aromatic=fb.is_aromatic,
                in_ring=fb.in_ring,
            )
        )
    combined.bonds.append(
        Bond(
            index=len(combined.bonds),
            begin_idx=anchor_idx,
            end_idx=offset,
            order=BondOrder.SINGLE,
        )
    )
    combined.rebuild_adjacency()
    smi = to_canonical_smiles(combined)
    try:
        reparsed = parse_smiles(smi)
        return reparsed.canonical_smiles
    except SmilesParseError:
        return None


def _prune_terminal_atom(base_mol: MoleculeGraph, leaf_idx: int) -> Optional[str]:
    """Remove a non-ring degree-1 heavy atom while keeping molecular connectivity."""
    if base_mol.num_heavy_atoms <= 6:
        return None
    if base_mol.degree(leaf_idx) != 1 or base_mol.atoms[leaf_idx].in_ring:
        return None

    kept_ids = [a.index for a in base_mol.atoms if a.index != leaf_idx]
    id_map = {old: new for new, old in enumerate(kept_ids)}
    new_atoms: List[Atom] = []
    for old in kept_ids:
        a = base_mol.atoms[old]
        new_atoms.append(
            Atom(
                index=id_map[old],
                symbol=a.symbol,
                atomic_number=a.atomic_number,
                formal_charge=a.formal_charge,
                explicit_h=a.explicit_h,
                implicit_h=a.implicit_h,
                is_aromatic=a.is_aromatic,
                hybridization=a.hybridization,
                in_ring=a.in_ring,
                ring_sizes=a.ring_sizes,
            )
        )
    new_bonds: List[Bond] = []
    for b in base_mol.bonds:
        if b.begin_idx == leaf_idx or b.end_idx == leaf_idx:
            continue
        new_bonds.append(
            Bond(
                index=len(new_bonds),
                begin_idx=id_map[b.begin_idx],
                end_idx=id_map[b.end_idx],
                order=b.order,
                is_aromatic=b.is_aromatic,
                in_ring=b.in_ring,
            )
        )
    pruned = MoleculeGraph(atoms=new_atoms, bonds=new_bonds)
    _assign_valences_and_hybridization(pruned, {})
    smi = to_canonical_smiles(pruned)
    try:
        return parse_smiles(smi).canonical_smiles
    except SmilesParseError:
        return None


def generate_valence_safe_neighbors(
    mol: MoleculeGraph,
    max_neighbors: int = 16,
    max_heavy_atoms: int = 42,
    rng: Optional[np.random.Generator] = None,
) -> List[str]:
    """Generate chemically valid mutated neighbor SMILES via bioisosteric & fragment transforms."""
    gen = rng if rng is not None else np.random.default_rng(42)
    neighbors: List[str] = []
    seen: Set[str] = {mol.canonical_smiles or to_canonical_smiles(mol)}

    # 1. Aromatic CH <-> N scaffold hop
    for a in mol.atoms:
        if a.is_aromatic and a.symbol == "C" and mol.degree(a.index) == 2 and a.total_h == 1:
            mut = mol.copy()
            mut.atoms[a.index].symbol = "N"
            mut.atoms[a.index].atomic_number = ELEMENT_TABLE["N"].atomic_number
            mut.atoms[a.index].implicit_h = 0
            try:
                cand = parse_smiles(to_canonical_smiles(mut)).canonical_smiles
                if cand not in seen:
                    seen.add(cand)
                    neighbors.append(cand)
            except SmilesParseError:
                pass
        elif a.is_aromatic and a.symbol == "N" and mol.degree(a.index) == 2 and a.total_h == 0:
            mut = mol.copy()
            mut.atoms[a.index].symbol = "C"
            mut.atoms[a.index].atomic_number = ELEMENT_TABLE["C"].atomic_number
            mut.atoms[a.index].implicit_h = 1
            try:
                cand = parse_smiles(to_canonical_smiles(mut)).canonical_smiles
                if cand not in seen:
                    seen.add(cand)
                    neighbors.append(cand)
            except SmilesParseError:
                pass

    # 2. Aliphatic O <-> S / N bioisostere swap on divalent non-aromatic linkers
    for a in mol.atoms:
        if not a.is_aromatic and a.symbol == "O" and mol.degree(a.index) == 2:
            mut = mol.copy()
            mut.atoms[a.index].symbol = "N"
            mut.atoms[a.index].atomic_number = ELEMENT_TABLE["N"].atomic_number
            mut.atoms[a.index].implicit_h = 1
            try:
                cand = parse_smiles(to_canonical_smiles(mut)).canonical_smiles
                if cand not in seen:
                    seen.add(cand)
                    neighbors.append(cand)
            except SmilesParseError:
                pass

    # 3. Attach medicinal-chemistry fragments at open valence positions
    if mol.num_heavy_atoms < max_heavy_atoms:
        open_sites = [
            a.index for a in mol.atoms if a.symbol in ("C", "N") and a.implicit_h >= 1 and a.formal_charge == 0
        ]
        if open_sites:
            gen.shuffle(open_sites)
            for site in open_sites[:5]:
                frag_choices = list(_TERMINAL_FRAGMENTS)
                gen.shuffle(frag_choices)
                for frag in frag_choices[:3]:
                    frag_cand = _attach_fragment(mol, site, frag)
                    if frag_cand is not None and frag_cand not in seen:
                        seen.add(frag_cand)
                        neighbors.append(frag_cand)

    # 4. Terminal substituent pruning
    leaves = [a.index for a in mol.atoms if mol.degree(a.index) == 1 and not a.in_ring]
    for leaf in leaves[:4]:
        prune_cand = _prune_terminal_atom(mol, leaf)
        if prune_cand is not None and prune_cand not in seen:
            seen.add(prune_cand)
            neighbors.append(prune_cand)

    gen.shuffle(neighbors)
    return neighbors[:max_neighbors]


def compute_pareto_front(candidates: Sequence[OptimizationCandidate]) -> List[OptimizationCandidate]:
    """Extract the non-dominated Pareto front (maximization over all objectives)."""
    front: List[OptimizationCandidate] = []
    for i, cand_i in enumerate(candidates):
        dominated = False
        obj_i = cand_i.objectives
        for j, cand_j in enumerate(candidates):
            if i == j:
                continue
            obj_j = cand_j.objectives
            if all(b >= a for a, b in zip(obj_i, obj_j)) and any(b > a for a, b in zip(obj_i, obj_j)):
                dominated = True
                break
        if not dominated:
            front.append(cand_i)
    front.sort(key=lambda c: -c.mpo_score)
    return front


def evaluate_candidate(
    smiles: str,
    seed_fp: np.ndarray,
    adme_predictor: AdmePredictor,
    pocket: ReceptorPocket,
    depth: int,
    parent_smiles: str,
    docking_steps: int = 35,
    seed: int = 42,
) -> OptimizationCandidate:
    """Score a candidate molecule across docking, QED, SA, BBB, and CYP3A4 DDI MPO objectives."""
    mol = parse_smiles(smiles)
    prof = compute_physicochemical_profile(mol)
    adme = adme_predictor.predict(mol, inhibitor_conc_uM=1.0, mc_samples=6, seed=seed)
    pose = dock_molecule(mol, pocket=pocket, steps=docking_steps, seed=seed)
    fp = compute_morgan_fingerprint(mol, radius=2, nbits=1024)
    sim = tanimoto_similarity(seed_fp, fp)

    # Normalized [0, 1] desirability terms
    dock_des = max(0.0, min(1.0, (-pose.total_affinity_kcal - 1.0) / 8.0))
    qed_des = prof.qed
    sa_des = max(0.0, min(1.0, (10.0 - prof.sa_score) / 9.0))
    ddi_des = max(0.0, min(1.0, 1.0 - max(0.0, adme.cyp3a4_ddi_auc_ratio - 1.0) / 2.5))
    alert_pen = 0.85 ** len(prof.alerts)
    lipinski_pen = 0.90 ** prof.lipinski_violations

    mpo = float(
        alert_pen
        * lipinski_pen
        * (0.34 * dock_des + 0.28 * qed_des + 0.18 * sa_des + 0.20 * ddi_des)
    )
    objectives = (
        float(-pose.total_affinity_kcal),
        float(prof.qed),
        float(-prof.sa_score),
        float(-adme.cyp3a4_ddi_auc_ratio),
    )

    return OptimizationCandidate(
        smiles=smiles,
        canonical_smiles=mol.canonical_smiles,
        mpo_score=mpo,
        docking_kcal=pose.total_affinity_kcal,
        qed=prof.qed,
        sa_score=prof.sa_score,
        log_bb=adme.log_bb,
        ddi_ratio=adme.cyp3a4_ddi_auc_ratio,
        tanimoto_to_seed=sim,
        depth=depth,
        parent_smiles=parent_smiles,
        objectives=objectives,
    )


def optimize_lead_mcts(
    seed_smiles: str,
    iterations: int = 30,
    max_depth: int = 4,
    pocket: Optional[ReceptorPocket] = None,
    seed: int = 42,
) -> OptimizationReport:
    """Run MCTS + Pareto multi-objective de novo lead optimization starting from seed_smiles."""
    rng = np.random.default_rng(seed)
    target_pocket = pocket if pocket is not None else build_kinase_hinge_pocket()
    adme_pred = AdmePredictor(hidden_dim=24, seed=seed)

    seed_mol = parse_smiles(seed_smiles)
    seed_canon = seed_mol.canonical_smiles
    seed_fp = compute_morgan_fingerprint(seed_mol, radius=2, nbits=1024)

    cache: Dict[str, OptimizationCandidate] = {}

    def _eval(smi: str, depth: int, parent: str) -> OptimizationCandidate:
        if smi not in cache:
            cache[smi] = evaluate_candidate(
                smi,
                seed_fp=seed_fp,
                adme_predictor=adme_pred,
                pocket=target_pocket,
                depth=depth,
                parent_smiles=parent,
                docking_steps=16,
                seed=seed + len(cache),
            )
        return cache[smi]

    root_cand = _eval(seed_canon, 0, "")
    root = _MctsNode(smiles=seed_canon, depth=0, parent_smiles="", visits=1, total_value=root_cand.mpo_score)
    mpo_history: List[float] = [root_cand.mpo_score]
    best_so_far = root_cand.mpo_score

    for _ in range(iterations):
        # 1. Selection
        node = root
        path: List[_MctsNode] = [node]
        while node.depth < max_depth:
            if node.untried_smiles is None:
                mol_node = parse_smiles(node.smiles)
                node.untried_smiles = generate_valence_safe_neighbors(mol_node, max_neighbors=12, rng=rng)
            if node.untried_smiles:
                break
            if not node.children:
                break
            node = max(node.children, key=lambda ch: ch.ucb1(node.visits))
            path.append(node)

        # 2. Expansion
        if node.depth < max_depth:
            if node.untried_smiles is None:
                node.untried_smiles = generate_valence_safe_neighbors(
                    parse_smiles(node.smiles), max_neighbors=12, rng=rng
                )
            if node.untried_smiles:
                child_smi = node.untried_smiles.pop(0)
                child = _MctsNode(smiles=child_smi, depth=node.depth + 1, parent_smiles=node.smiles)
                node.children.append(child)
                path.append(child)
                node = child

        # 3. Evaluation / Rollout value
        cand = _eval(node.smiles, node.depth, node.parent_smiles)
        reward = cand.mpo_score
        best_so_far = max(best_so_far, reward)
        mpo_history.append(best_so_far)

        # 4. Backpropagation
        for p_node in path:
            p_node.visits += 1
            p_node.total_value += reward

    all_cands = sorted(cache.values(), key=lambda c: -c.mpo_score)
    pareto = compute_pareto_front(all_cands)

    return OptimizationReport(
        seed_smiles=seed_canon,
        iterations=iterations,
        unique_evaluated=len(all_cands),
        best_candidate=all_cands[0],
        pareto_front=pareto,
        top_candidates=all_cands[:10],
        mpo_history=mpo_history,
    )
