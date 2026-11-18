"""Physicochemical descriptors, Gasteiger partial charges, QED, Bemis-Murcko scaffolds, and PAINS alerts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import math
import numpy as np

from chiral_weave.chem_types import (
    ELEMENT_TABLE,
    Atom,
    Bond,
    BondOrder,
    Hybridization,
    MoleculeGraph,
    PharmacophoreAnnotation,
    PharmacophoreKind,
    StereoConfig,
)
from chiral_weave.smiles_parser import (
    _assign_valences_and_hybridization,
    has_smarts_match,
    to_canonical_smiles,
)


# Gasteiger-Marsili (1980) orbital electronegativity coefficients (a, b, c) in eV
_GASTEIGER_PARAMS: Dict[Tuple[str, Hybridization], Tuple[float, float, float]] = {
    ("H", Hybridization.S): (7.17, 6.24, -0.56),
    ("C", Hybridization.SP3): (7.98, 9.18, 1.88),
    ("C", Hybridization.SP2): (8.79, 9.32, 1.51),
    ("C", Hybridization.SP): (10.39, 9.45, 0.73),
    ("N", Hybridization.SP3): (11.54, 10.82, 1.36),
    ("N", Hybridization.SP2): (12.87, 11.15, 0.85),
    ("N", Hybridization.SP): (15.68, 11.70, -0.27),
    ("O", Hybridization.SP3): (14.18, 12.92, 1.39),
    ("O", Hybridization.SP2): (17.07, 13.79, 0.47),
    ("F", Hybridization.SP3): (14.66, 13.85, 2.31),
    ("Cl", Hybridization.SP3): (11.00, 9.69, 1.35),
    ("Br", Hybridization.SP3): (10.08, 8.47, 1.16),
    ("I", Hybridization.SP3): (9.90, 7.96, 0.96),
    ("S", Hybridization.SP3): (10.14, 9.13, 1.38),
    ("S", Hybridization.SP2): (10.88, 9.49, 1.33),
    ("P", Hybridization.SP3): (8.90, 8.32, 1.58),
    ("P", Hybridization.SP2): (9.65, 8.60, 1.40),
    ("B", Hybridization.SP3): (5.98, 6.82, 1.60),
    ("B", Hybridization.SP2): (6.42, 7.10, 1.45),
    ("Si", Hybridization.SP3): (7.30, 6.50, 1.10),
}

# Bickerton et al. (2012) QED asymmetric double-sigmoidal ADS parameters & weights
_QED_PARAMS: Dict[str, Tuple[float, float, float, float, float, float, float]] = {
    # (weight, a, b, c, d, e, f)
    "mw": (0.66, 2.817065973, 392.5754953, 290.7489764, 2.419764353, 49.22325677, 65.37051707),
    "alogp": (0.46, 3.172690585, 137.8624751, 2.534937431, 4.581457802, 0.822739154, 0.576295591),
    "hba": (0.05, 2.948620388, 160.4605972, 3.615294657, 4.435986202, 0.290141953, 1.300669958),
    "hbd": (0.61, 1.618662227, 1010.051101, 0.985094388, 0.000000001, 0.713820843, 0.920922555),
    "psa": (0.06, 1.876861559, 125.2232657, 62.90773554, 87.83366614, 12.01999824, 28.51324732),
    "rotb": (0.65, 0.010000000, 272.4121427, 2.558379970, 1.565547684, 1.271567166, 2.758063707),
    "arom": (0.48, 3.217788970, 957.7374108, 2.274627939, 0.000000001, 1.317690384, 0.375760881),
    "alerts": (0.95, 0.010000000, 1199.094025, -0.09002883, 0.000000001, 0.185904477, 0.875193782),
}

# Curated medicinal chemistry PAINS and reactive structural alert SMARTS
STRUCTURAL_ALERTS_SMARTS: Dict[str, str] = {
    "michael_acceptor": "[C]=[C]-[C]=[O]",
    "epoxide_aziridine": "[O,N]1[C][C]1",
    "acyl_halide": "[C](=[O])[F,Cl,Br,I]",
    "alkyl_halide": "[CX4]-[Cl,Br,I]",
    "nitro_aromatic": "[a]-[N+](=O)[O-]",
    "nitro_group": "[N](=[O])[O]",
    "azo_bond": "[N]=[N]",
    "aldehyde": "[CH1](=[O])",
    "rhodanine_like": "[S]1[C](=[S])[N][C](=[O])[C]1",
    "quinone_like": "[#6]1(=[O])[#6]=[#6][#6](=[O])[#6]=[#6]1",
    "hydrazine": "[N]-[N]",
    "peroxide": "[O]-[O]",
    "thiocarbonyl": "[C]=[S]",
}


@dataclass(frozen=True)
class PhysicochemicalProfile:
    """Comprehensive physicochemical and medicinal-chemistry profile."""

    smiles: str
    formula: str
    molecular_weight: float
    heavy_atom_count: int
    ring_count: int
    aromatic_ring_count: int
    rotatable_bonds: int
    hbd: int
    hba: int
    tpsa: float
    clogp: float
    molar_refractivity: float
    fraction_csp3: float
    formal_charge: int
    qed: float
    sa_score: float
    lipinski_violations: int
    veber_pass: bool
    bemis_murcko_scaffold: str
    alerts: Tuple[str, ...]


def compute_gasteiger_charges(mol: MoleculeGraph, iterations: int = 6) -> np.ndarray:
    """Compute Gasteiger-Marsili partial atomic charges by iterative electronegativity equalization.

    Args:
        mol: Input MoleculeGraph.
        iterations: Number of damped charge-transfer iterations (default 6).

    Returns:
        Array of partial charges per heavy atom whose sum equals total formal charge.
    """
    n = mol.num_atoms
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # Expand implicit hydrogens as virtual attached H nodes so H polarization is captured
    h_parent: List[int] = []
    for a in mol.atoms:
        for _ in range(a.implicit_h):
            h_parent.append(a.index)
    total_nodes = n + len(h_parent)

    a_coef = np.zeros(total_nodes, dtype=np.float64)
    b_coef = np.zeros(total_nodes, dtype=np.float64)
    c_coef = np.zeros(total_nodes, dtype=np.float64)
    chi_plus = np.zeros(total_nodes, dtype=np.float64)
    charges = np.zeros(total_nodes, dtype=np.float64)

    for atom in mol.atoms:
        key = (atom.symbol, atom.hybridization)
        if key not in _GASTEIGER_PARAMS:
            key = (atom.symbol, Hybridization.SP3)
        if key not in _GASTEIGER_PARAMS:
            a_v, b_v, c_v = (8.0, 9.0, 1.5)
        else:
            a_v, b_v, c_v = _GASTEIGER_PARAMS[key]
        a_coef[atom.index] = a_v
        b_coef[atom.index] = b_v
        c_coef[atom.index] = c_v
        chi_plus[atom.index] = max(1e-6, a_v + b_v + c_v)
        charges[atom.index] = float(atom.formal_charge)

    h_a, h_b, h_c = _GASTEIGER_PARAMS[("H", Hybridization.S)]
    for offset in range(len(h_parent)):
        idx = n + offset
        a_coef[idx] = h_a
        b_coef[idx] = h_b
        c_coef[idx] = h_c
        chi_plus[idx] = 20.02

    edges: List[Tuple[int, int]] = [(b.begin_idx, b.end_idx) for b in mol.bonds]
    for offset, parent in enumerate(h_parent):
        edges.append((parent, n + offset))

    damping = 0.5
    for _ in range(iterations):
        chi = a_coef + b_coef * charges + c_coef * (charges ** 2)
        dq = np.zeros(total_nodes, dtype=np.float64)
        for u, v in edges:
            if chi[v] > chi[u]:
                denom = chi_plus[u]
                delta = ((chi[v] - chi[u]) / denom) * damping
                dq[u] += delta
                dq[v] -= delta
            elif chi[u] > chi[v]:
                denom = chi_plus[v]
                delta = ((chi[u] - chi[v]) / denom) * damping
                dq[v] += delta
                dq[u] -= delta
        charges += dq
        damping *= 0.5

    # Fold virtual implicit H charges back onto their heavy parent atoms for heavy-atom conservation
    heavy_charges = charges[:n].copy()
    for offset, parent in enumerate(h_parent):
        heavy_charges[parent] += charges[n + offset]

    for atom in mol.atoms:
        atom.gasteiger_charge = float(heavy_charges[atom.index])

    return heavy_charges


def _is_carbonyl_carbon(atom: Atom, mol: MoleculeGraph) -> bool:
    if atom.symbol != "C":
        return False
    for b in mol.incident_bonds(atom.index):
        other = mol.atoms[b.other(atom.index)]
        if b.order == BondOrder.DOUBLE and other.symbol in ("O", "S", "N"):
            return True
    return False


def compute_crippen_logp_mr(mol: MoleculeGraph) -> Tuple[float, float]:
    """Estimate Wildman-Crippen octanol-water logP and molar refractivity (MR)."""
    logp = 0.0
    mr = 0.0
    h_spec = ELEMENT_TABLE["H"]

    for atom in mol.atoms:
        spec = ELEMENT_TABLE[atom.symbol]
        c_logp = spec.crippen_logp
        c_mr = spec.crippen_mr
        inc = mol.incident_bonds(atom.index)
        hetero_nbrs = sum(
            1 for b in inc if ELEMENT_TABLE[mol.atoms[b.other(atom.index)].symbol].is_heteroatom
        )

        if atom.symbol == "C":
            if atom.is_aromatic:
                c_logp = 0.1581 if hetero_nbrs == 0 else -0.0250
                c_mr = 3.350
            elif _is_carbonyl_carbon(atom, mol):
                c_logp = -0.2780
                c_mr = 2.950
            elif atom.hybridization == Hybridization.SP2:
                c_logp = 0.1551
                c_mr = 3.210
            elif atom.hybridization == Hybridization.SP:
                c_logp = 0.0017
                c_mr = 2.890
            else:
                c_logp = 0.1441 - 0.1950 * hetero_nbrs
                c_mr = 2.503
        elif atom.symbol == "N":
            if atom.formal_charge > 0:
                c_logp = -1.950
                c_mr = 2.200
            elif atom.is_aromatic:
                c_logp = -0.3239 if atom.total_h == 0 else -0.4950
                c_mr = 2.300
            elif any(_is_carbonyl_carbon(mol.atoms[b.other(atom.index)], mol) for b in inc):
                c_logp = -0.7096
                c_mr = 2.262
            elif atom.total_h > 0:
                c_logp = -0.8186
                c_mr = 2.134
            else:
                c_logp = -0.3239
                c_mr = 2.400
        elif atom.symbol == "O":
            if atom.formal_charge < 0:
                c_logp = -1.320
                c_mr = 0.900
            elif atom.total_h > 0:
                c_logp = -0.2893
                c_mr = 0.823
            elif any(b.order == BondOrder.DOUBLE for b in inc):
                c_logp = -0.2470
                c_mr = 0.784
            else:
                c_logp = -0.1526
                c_mr = 1.080
        elif atom.symbol == "S":
            if any(b.order == BondOrder.DOUBLE for b in inc):
                c_logp = -0.050
                c_mr = 7.365
            else:
                c_logp = 0.6482
                c_mr = 7.591

        logp += c_logp + atom.total_h * h_spec.crippen_logp
        mr += c_mr + atom.total_h * h_spec.crippen_mr

    return float(logp), float(mr)


def compute_tpsa(mol: MoleculeGraph, include_sp: bool = False) -> float:
    """Compute Ertl Topological Polar Surface Area (TPSA) in Angstrom^2."""
    tpsa = 0.0
    for atom in mol.atoms:
        if atom.symbol not in ("N", "O", "S", "P"):
            continue
        inc = mol.incident_bonds(atom.index)
        deg = len(inc)
        h_cnt = atom.total_h

        if atom.symbol == "O":
            if atom.formal_charge < 0:
                tpsa += 23.06
            elif h_cnt >= 1:
                tpsa += 20.23
            elif any(b.order == BondOrder.DOUBLE for b in inc):
                tpsa += 17.07
            elif atom.is_aromatic:
                tpsa += 13.14
            else:
                tpsa += 9.23
        elif atom.symbol == "N":
            if atom.is_aromatic:
                tpsa += 15.79 if h_cnt >= 1 else 12.89
            elif any(b.order == BondOrder.TRIPLE for b in inc):
                tpsa += 23.79
            elif atom.formal_charge > 0:
                tpsa += 27.64 if h_cnt >= 3 else 16.61
            elif h_cnt >= 2:
                tpsa += 26.02
            elif h_cnt == 1:
                tpsa += 12.03
            elif deg == 2 and any(b.order == BondOrder.DOUBLE for b in inc):
                tpsa += 12.36
            else:
                tpsa += 3.24
        elif include_sp and atom.symbol == "S":
            tpsa += 38.80 if h_cnt >= 1 else 25.30
        elif include_sp and atom.symbol == "P":
            tpsa += 13.59 if h_cnt == 0 else 23.47
    return float(tpsa)


def count_hbd_hba(mol: MoleculeGraph) -> Tuple[int, int]:
    """Count Lipinski Hydrogen Bond Donors (HBD) and Acceptors (HBA)."""
    hbd = 0
    hba = 0
    for atom in mol.atoms:
        if atom.symbol in ("N", "O"):
            if atom.total_h > 0:
                hbd += 1
            if atom.formal_charge <= 0:
                hba += 1
    return hbd, hba


def count_rotatable_bonds(mol: MoleculeGraph) -> int:
    """Count non-terminal, non-ring single bonds that are not amide C-N linkages."""
    rotb = 0
    for b in mol.bonds:
        if b.in_ring or b.is_aromatic or b.order != BondOrder.SINGLE:
            continue
        u = mol.atoms[b.begin_idx]
        v = mol.atoms[b.end_idx]
        if u.atomic_number == 1 or v.atomic_number == 1:
            continue
        if mol.degree(u.index) <= 1 or mol.degree(v.index) <= 1:
            continue
        if u.hybridization == Hybridization.SP or v.hybridization == Hybridization.SP:
            continue
        # Exclude partial double-bond character amide C(=O)-N
        if (u.symbol == "C" and v.symbol == "N" and _is_carbonyl_carbon(u, mol)) or (
            v.symbol == "C" and u.symbol == "N" and _is_carbonyl_carbon(v, mol)
        ):
            continue
        rotb += 1
    return rotb


def extract_bemis_murcko_scaffold(mol: MoleculeGraph) -> str:
    """Extract the Bemis-Murcko core ring + linker framework SMILES via iterative leaf pruning."""
    if not mol.sssr_rings:
        return ""

    kept: Set[int] = {a.index for a in mol.atoms if a.atomic_number > 1}
    changed = True
    while changed:
        changed = False
        to_remove: List[int] = []
        for idx in kept:
            if mol.atoms[idx].in_ring:
                continue
            deg_in_kept = sum(1 for nbr in mol.neighbors(idx) if nbr in kept)
            if deg_in_kept <= 1:
                to_remove.append(idx)
        if to_remove:
            for idx in to_remove:
                kept.remove(idx)
            changed = True

    if not kept:
        return ""

    # Preserve exocyclic doubly-bonded =O/=S/=N directly attached to a kept ring atom if desired?
    # Pure Bemis-Murcko graph framework strips exocyclic double bonds unless kept; build subgraph cleanly.
    sorted_kept = sorted(kept)
    idx_map = {old: new for new, old in enumerate(sorted_kept)}
    sub_atoms: List[Atom] = []
    for old_idx in sorted_kept:
        orig = mol.atoms[old_idx]
        sub_atoms.append(
            Atom(
                index=idx_map[old_idx],
                symbol=orig.symbol,
                atomic_number=orig.atomic_number,
                formal_charge=orig.formal_charge,
                explicit_h=orig.explicit_h,
                implicit_h=orig.implicit_h,
                is_aromatic=orig.is_aromatic,
                hybridization=orig.hybridization,
                in_ring=orig.in_ring,
                ring_sizes=orig.ring_sizes,
            )
        )
    sub_bonds: List[Bond] = []
    for b in mol.bonds:
        if b.begin_idx in idx_map and b.end_idx in idx_map:
            sub_bonds.append(
                Bond(
                    index=len(sub_bonds),
                    begin_idx=idx_map[b.begin_idx],
                    end_idx=idx_map[b.end_idx],
                    order=b.order,
                    is_aromatic=b.is_aromatic,
                    in_ring=b.in_ring,
                    is_conjugated=b.is_conjugated,
                )
            )
    sub_mol = MoleculeGraph(atoms=sub_atoms, bonds=sub_bonds)
    _assign_valences_and_hybridization(sub_mol, {})
    return to_canonical_smiles(sub_mol)


def compute_sa_score(mol: MoleculeGraph) -> float:
    """Estimate Ertl-style Synthetic Accessibility (SA) score in [1.0 (easy), 10.0 (hard)]."""
    n_heavy = max(1, mol.num_heavy_atoms)
    rings = mol.sssr_rings
    n_rings = len(rings)

    # Ring complexity: spiro + bridgehead atoms + macrocycles
    ring_atom_counts: Dict[int, int] = {}
    for r in rings:
        for a_idx in r:
            ring_atom_counts[a_idx] = ring_atom_counts.get(a_idx, 0) + 1
    spiro_or_bridge = sum(1 for a_idx, cnt in ring_atom_counts.items() if cnt >= 2 and mol.degree(a_idx) >= 3)
    macrocycles = sum(1 for r in rings if len(r) > 7)
    stereo_centers = sum(1 for a in mol.atoms if a.stereo != StereoConfig.NONE)
    hetero_count = sum(1 for a in mol.atoms if ELEMENT_TABLE[a.symbol].is_heteroatom)

    size_penalty = math.pow(n_heavy, 1.005) - n_heavy
    stereo_penalty = math.log10(stereo_centers + 1.0) * 1.4
    spiro_penalty = math.log10(spiro_or_bridge + 1.0) * 1.2
    macro_penalty = 0.8 * macrocycles
    ring_penalty = 0.25 * max(0, n_rings - 2)
    hetero_bonus = 0.15 * min(6, hetero_count)

    raw = 1.65 + size_penalty + stereo_penalty + spiro_penalty + macro_penalty + ring_penalty - 0.1 * hetero_bonus
    return float(max(1.0, min(10.0, raw)))


def _ads_raw(x: float, params: Tuple[float, float, float, float, float, float, float]) -> float:
    _, a, b, c, d, e, f = params
    z1 = max(-60.0, min(60.0, -(x - c + d / 2.0) / max(1e-9, e)))
    z2 = max(-60.0, min(60.0, -(x - c - d / 2.0) / max(1e-9, f)))
    term1 = 1.0 / (1.0 + math.exp(z1))
    term2 = 1.0 - 1.0 / (1.0 + math.exp(z2))
    return a + b * term1 * term2


_QED_DMAX: Dict[str, float] = {
    k: max(_ads_raw(p[3] + (i - 200) * 0.05 * max(1.0, p[5] + p[6]), p) for i in range(401))
    for k, p in _QED_PARAMS.items()
}


def _ads_desirability(
    x: float,
    params: Tuple[float, float, float, float, float, float, float],
    dmax: float,
) -> float:
    """Bickerton Asymmetric Double Sigmoidal (ADS) desirability function normalized by curve peak."""
    val = _ads_raw(x, params)
    return max(1e-6, min(1.0, val / max(1e-9, dmax)))


def compute_qed(
    mw: float,
    alogp: float,
    hba: int,
    hbd: int,
    psa: float,
    rotb: int,
    arom_rings: int,
    num_alerts: int,
) -> float:
    """Compute Bickerton et al. (2012) Quantitative Estimate of Drug-likeness in [0, 1]."""
    vals = {
        "mw": mw,
        "alogp": alogp,
        "hba": float(hba),
        "hbd": float(hbd),
        "psa": psa,
        "rotb": float(rotb),
        "arom": float(arom_rings),
        "alerts": float(num_alerts),
    }
    weighted_log_sum = 0.0
    weight_total = 0.0
    for key, val in vals.items():
        params = _QED_PARAMS[key]
        w = params[0]
        d = _ads_desirability(val, params, _QED_DMAX[key])
        weighted_log_sum += w * math.log(d)
        weight_total += w
    return float(math.exp(weighted_log_sum / weight_total))


def detect_structural_alerts(mol: MoleculeGraph) -> Tuple[str, ...]:
    """Screen MoleculeGraph against PAINS and reactive functional-group SMARTS filters."""
    triggered: List[str] = []
    for name, smarts in STRUCTURAL_ALERTS_SMARTS.items():
        if has_smarts_match(mol, smarts):
            triggered.append(name)
    return tuple(sorted(triggered))


def assign_pharmacophores(mol: MoleculeGraph) -> List[PharmacophoreAnnotation]:
    """Assign medicinal chemistry pharmacophore roles to heavy atoms."""
    annotations: List[PharmacophoreAnnotation] = []
    for atom in mol.atoms:
        if atom.atomic_number == 1:
            continue
        if atom.formal_charge > 0 or (
            atom.symbol == "N" and not atom.is_aromatic and atom.hybridization == Hybridization.SP3
        ):
            if atom.symbol == "N":
                annotations.append(PharmacophoreAnnotation(atom.index, PharmacophoreKind.POS_IONIZABLE, atom.symbol))
        if atom.formal_charge < 0:
            annotations.append(PharmacophoreAnnotation(atom.index, PharmacophoreKind.NEG_IONIZABLE, atom.symbol))
        if atom.symbol in ("N", "O") and atom.total_h > 0:
            annotations.append(PharmacophoreAnnotation(atom.index, PharmacophoreKind.HBD, atom.symbol))
        if atom.symbol in ("N", "O") and atom.formal_charge <= 0:
            annotations.append(PharmacophoreAnnotation(atom.index, PharmacophoreKind.HBA, atom.symbol))
        if atom.is_aromatic:
            annotations.append(PharmacophoreAnnotation(atom.index, PharmacophoreKind.AROMATIC, atom.symbol))
        elif atom.symbol in ("C", "S", "Cl", "Br", "I", "F"):
            has_polar = any(
                mol.atoms[nbr].symbol in ("N", "O") for nbr in mol.neighbors(atom.index)
            )
            if not has_polar:
                annotations.append(PharmacophoreAnnotation(atom.index, PharmacophoreKind.HYDROPHOBIC, atom.symbol))
    return annotations


def compute_physicochemical_profile(mol: MoleculeGraph) -> PhysicochemicalProfile:
    """Compute full physicochemical, scaffold, drug-likeness, and structural-alert profile."""
    compute_gasteiger_charges(mol)
    mw = mol.molecular_weight()
    clogp, mr = compute_crippen_logp_mr(mol)
    tpsa = compute_tpsa(mol)
    hbd, hba = count_hbd_hba(mol)
    rotb = count_rotatable_bonds(mol)

    n_carbons = sum(1 for a in mol.atoms if a.symbol == "C")
    n_sp3_c = sum(1 for a in mol.atoms if a.symbol == "C" and a.hybridization == Hybridization.SP3)
    f_csp3 = float(n_sp3_c / n_carbons) if n_carbons > 0 else 0.0

    arom_rings = 0
    for ring in mol.sssr_rings:
        if all(mol.atoms[idx].is_aromatic for idx in ring):
            arom_rings += 1

    alerts = detect_structural_alerts(mol)
    qed = compute_qed(mw, clogp, hba, hbd, tpsa, rotb, arom_rings, len(alerts))
    sa = compute_sa_score(mol)
    scaffold = extract_bemis_murcko_scaffold(mol)

    lipinski_viol = int(mw > 500.0) + int(clogp > 5.0) + int(hbd > 5) + int(hba > 10)
    veber_pass = rotb <= 10 and tpsa <= 140.0
    formal_q = sum(a.formal_charge for a in mol.atoms)

    return PhysicochemicalProfile(
        smiles=mol.canonical_smiles or mol.source_smiles,
        formula=mol.molecular_formula(),
        molecular_weight=mw,
        heavy_atom_count=mol.num_heavy_atoms,
        ring_count=len(mol.sssr_rings),
        aromatic_ring_count=arom_rings,
        rotatable_bonds=rotb,
        hbd=hbd,
        hba=hba,
        tpsa=tpsa,
        clogp=clogp,
        molar_refractivity=mr,
        fraction_csp3=f_csp3,
        formal_charge=formal_q,
        qed=qed,
        sa_score=sa,
        lipinski_violations=lipinski_viol,
        veber_pass=veber_pass,
        bemis_murcko_scaffold=scaffold,
        alerts=alerts,
    )
