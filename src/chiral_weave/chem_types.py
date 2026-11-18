"""Typed chemical graph data structures, periodic element parameters, and simulation records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class BondOrder(Enum):
    """Chemical bond order classification."""

    SINGLE = 1.0
    DOUBLE = 2.0
    TRIPLE = 3.0
    AROMATIC = 1.5


class Hybridization(Enum):
    """Atomic orbital hybridization state."""

    S = "S"
    SP = "SP"
    SP2 = "SP2"
    SP3 = "SP3"
    OTHER = "OTHER"


class StereoConfig(Enum):
    """Stereochemical descriptor for tetrahedral centers and double bonds."""

    NONE = "NONE"
    TETRAHEDRAL_CCW = "@"
    TETRAHEDRAL_CW = "@@"
    E_TRANS = "E"
    Z_CIS = "Z"


class PharmacophoreKind(IntEnum):
    """Pharmacophoric interaction role assigned to heavy atoms."""

    HYDROPHOBIC = 0
    HBD = 1
    HBA = 2
    AROMATIC = 3
    POS_IONIZABLE = 4
    NEG_IONIZABLE = 5


@dataclass(frozen=True)
class ElementSpec:
    """Physical and cheminformatics parameters for a chemical element.

    Attributes:
        atomic_number: Proton count Z.
        symbol: IUPAC element symbol.
        atomic_mass: Standard atomic weight in Da (g/mol).
        covalent_radius: Single-bond covalent radius in Angstrom.
        vdw_radius: Bondi van der Waals radius in Angstrom.
        electronegativity: Pauling electronegativity scale.
        default_valences: Allowed neutral covalent valences in ascending order.
        crippen_logp: Baseline Wildman-Crippen atomic logP contribution.
        crippen_mr: Baseline Wildman-Crippen molar refractivity contribution.
        is_heteroatom: Whether the atom counts as N/O/S/P/halogen heteroatom.
    """

    atomic_number: int
    symbol: str
    atomic_mass: float
    covalent_radius: float
    vdw_radius: float
    electronegativity: float
    default_valences: Tuple[int, ...]
    crippen_logp: float
    crippen_mr: float
    is_heteroatom: bool = True


ELEMENT_TABLE: Dict[str, ElementSpec] = {
    "H": ElementSpec(1, "H", 1.008, 0.31, 1.20, 2.20, (1,), 0.1230, 1.057, False),
    "B": ElementSpec(5, "B", 10.811, 0.84, 1.92, 2.04, (3,), -0.1500, 3.040, True),
    "C": ElementSpec(6, "C", 12.011, 0.76, 1.70, 2.55, (4,), 0.1441, 2.503, False),
    "N": ElementSpec(7, "N", 14.007, 0.71, 1.55, 3.04, (3, 5), -0.3239, 2.134, True),
    "O": ElementSpec(8, "O", 15.999, 0.66, 1.52, 3.44, (2,), -0.2893, 0.823, True),
    "F": ElementSpec(9, "F", 18.998, 0.57, 1.47, 3.98, (1,), 0.4118, 1.108, True),
    "Si": ElementSpec(14, "Si", 28.085, 1.11, 2.10, 1.90, (4,), 0.3000, 7.200, True),
    "P": ElementSpec(15, "P", 30.974, 1.07, 1.80, 2.19, (3, 5), 0.4210, 6.920, True),
    "S": ElementSpec(16, "S", 32.065, 1.05, 1.80, 2.58, (2, 4, 6), 0.6482, 7.591, True),
    "Cl": ElementSpec(17, "Cl", 35.453, 1.02, 1.75, 3.16, (1,), 0.6895, 5.853, True),
    "Br": ElementSpec(35, "Br", 79.904, 1.20, 1.85, 2.96, (1,), 0.8456, 8.927, True),
    "I": ElementSpec(53, "I", 126.904, 1.39, 1.98, 2.66, (1,), 1.0200, 13.940, True),
}

ATOMIC_NUM_TO_SYMBOL: Dict[int, str] = {spec.atomic_number: sym for sym, spec in ELEMENT_TABLE.items()}
ORGANIC_SUBSET: Set[str] = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}


def get_element(symbol: str) -> ElementSpec:
    """Look up element metadata, normalizing aromatic lowercase symbols."""
    if not symbol:
        raise ValueError("Empty element symbol.")
    normalized = symbol[0].upper() + symbol[1:].lower()
    if normalized not in ELEMENT_TABLE:
        raise ValueError(f"Unsupported element symbol: {symbol!r}")
    return ELEMENT_TABLE[normalized]


@dataclass
class Atom:
    """Vertex in a molecular graph representing a single heavy atom or explicit hydrogen.

    Attributes:
        index: Zero-based atom index in MoleculeGraph.atoms.
        symbol: Canonical IUPAC element symbol.
        atomic_number: Element proton count Z.
        formal_charge: Integer formal charge.
        isotope: Optional isotopic mass number.
        explicit_h: Explicit hydrogens specified in SMILES bracket syntax.
        implicit_h: Inferred hydrogens satisfying neutral/charged valence.
        is_aromatic: Whether atom participates in a Hueckel aromatic ring.
        hybridization: Assigned orbital hybridization state.
        stereo: Chiral configuration marker (@ or @@) when specified.
        in_ring: True if atom belongs to at least one cycle.
        ring_sizes: Sorted distinct SSSR ring sizes containing this atom.
        gasteiger_charge: Computed Gasteiger-Marsili partial charge.
    """

    index: int
    symbol: str
    atomic_number: int
    formal_charge: int = 0
    isotope: Optional[int] = None
    explicit_h: int = 0
    implicit_h: int = 0
    is_aromatic: bool = False
    hybridization: Hybridization = Hybridization.SP3
    stereo: StereoConfig = StereoConfig.NONE
    in_ring: bool = False
    ring_sizes: Tuple[int, ...] = ()
    gasteiger_charge: float = 0.0

    @property
    def total_h(self) -> int:
        """Total attached hydrogens (explicit + implicit)."""
        return self.explicit_h + self.implicit_h


@dataclass
class Bond:
    """Undirected covalent bond connecting two atoms in a molecular graph."""

    index: int
    begin_idx: int
    end_idx: int
    order: BondOrder
    is_aromatic: bool = False
    in_ring: bool = False
    stereo_dir: str = ""
    is_conjugated: bool = False

    def other(self, atom_idx: int) -> int:
        """Return the neighbor index opposite atom_idx."""
        if atom_idx == self.begin_idx:
            return self.end_idx
        if atom_idx == self.end_idx:
            return self.begin_idx
        raise ValueError(f"Atom {atom_idx} is not incident to bond {self.index}.")


@dataclass
class MoleculeGraph:
    """Pure-Python topological and optional 3D molecular representation.

    Maintains atoms, covalent bonds, SSSR ring basis, adjacency lists, and
    optional 3D Cartesian coordinates (in Angstrom).
    """

    atoms: List[Atom]
    bonds: List[Bond]
    source_smiles: str = ""
    canonical_smiles: str = ""
    name: str = ""
    sssr_rings: List[Tuple[int, ...]] = field(default_factory=list)
    coords: Optional[np.ndarray] = None
    metadata: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._adj: Dict[int, List[Tuple[int, int]]] = {a.index: [] for a in self.atoms}
        for bond in self.bonds:
            self._adj[bond.begin_idx].append((bond.end_idx, bond.index))
            self._adj[bond.end_idx].append((bond.begin_idx, bond.index))

    def rebuild_adjacency(self) -> None:
        """Recompute adjacency lists after structural graph modifications."""
        self._adj = {a.index: [] for a in self.atoms}
        for bond in self.bonds:
            self._adj[bond.begin_idx].append((bond.end_idx, bond.index))
            self._adj[bond.end_idx].append((bond.begin_idx, bond.index))

    @property
    def num_atoms(self) -> int:
        """Number of explicit atoms in the graph."""
        return len(self.atoms)

    @property
    def num_heavy_atoms(self) -> int:
        """Number of non-hydrogen atoms."""
        return sum(1 for a in self.atoms if a.atomic_number > 1)

    @property
    def num_bonds(self) -> int:
        """Number of covalent bonds."""
        return len(self.bonds)

    def neighbors(self, atom_idx: int) -> List[int]:
        """Return adjacent atom indices for atom_idx."""
        return [nbr for nbr, _ in self._adj[atom_idx]]

    def incident_bonds(self, atom_idx: int) -> List[Bond]:
        """Return Bond objects incident to atom_idx."""
        return [self.bonds[b_idx] for _, b_idx in self._adj[atom_idx]]

    def degree(self, atom_idx: int) -> int:
        """Explicit heavy-atom/explicit-vertex degree of atom_idx."""
        return len(self._adj[atom_idx])

    def find_bond(self, idx_a: int, idx_b: int) -> Optional[Bond]:
        """Return the Bond connecting idx_a and idx_b, or None if unbonded."""
        for nbr, b_idx in self._adj[idx_a]:
            if nbr == idx_b:
                return self.bonds[b_idx]
        return None

    def heavy_valence(self, atom_idx: int) -> float:
        """Sum of explicit bond orders incident to atom_idx."""
        total = 0.0
        for bond in self.incident_bonds(atom_idx):
            total += bond.order.value
        return total

    def molecular_weight(self) -> float:
        """Exact average molecular weight in Da including implicit hydrogens."""
        mw = 0.0
        h_mass = ELEMENT_TABLE["H"].atomic_mass
        for atom in self.atoms:
            spec = ELEMENT_TABLE[atom.symbol]
            mw += spec.atomic_mass
            mw += atom.implicit_h * h_mass
        return float(mw)

    def molecular_formula(self) -> str:
        """Hill-system molecular formula string."""
        counts: Dict[str, int] = {}
        total_h = 0
        for atom in self.atoms:
            if atom.symbol == "H":
                total_h += 1
            else:
                counts[atom.symbol] = counts.get(atom.symbol, 0) + 1
                total_h += atom.total_h
        if total_h > 0:
            counts["H"] = total_h
        parts: List[str] = []
        if "C" in counts:
            n_c = counts.pop("C")
            parts.append("C" if n_c == 1 else f"C{n_c}")
            if "H" in counts:
                n_h = counts.pop("H")
                parts.append("H" if n_h == 1 else f"H{n_h}")
        for sym in sorted(counts):
            cnt = counts[sym]
            parts.append(sym if cnt == 1 else f"{sym}{cnt}")
        return "".join(parts)

    def copy(self) -> MoleculeGraph:
        """Return a deep topological and coordinate copy of this molecule."""
        new_atoms = [
            Atom(
                index=a.index,
                symbol=a.symbol,
                atomic_number=a.atomic_number,
                formal_charge=a.formal_charge,
                isotope=a.isotope,
                explicit_h=a.explicit_h,
                implicit_h=a.implicit_h,
                is_aromatic=a.is_aromatic,
                hybridization=a.hybridization,
                stereo=a.stereo,
                in_ring=a.in_ring,
                ring_sizes=a.ring_sizes,
                gasteiger_charge=a.gasteiger_charge,
            )
            for a in self.atoms
        ]
        new_bonds = [
            Bond(
                index=b.index,
                begin_idx=b.begin_idx,
                end_idx=b.end_idx,
                order=b.order,
                is_aromatic=b.is_aromatic,
                in_ring=b.in_ring,
                stereo_dir=b.stereo_dir,
                is_conjugated=b.is_conjugated,
            )
            for b in self.bonds
        ]
        new_coords = None if self.coords is None else self.coords.copy()
        return MoleculeGraph(
            atoms=new_atoms,
            bonds=new_bonds,
            source_smiles=self.source_smiles,
            canonical_smiles=self.canonical_smiles,
            name=self.name,
            sssr_rings=list(self.sssr_rings),
            coords=new_coords,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class PharmacophoreAnnotation:
    """Assigned pharmacophore feature at an atom index."""

    atom_index: int
    kind: PharmacophoreKind
    symbol: str


@dataclass(frozen=True)
class AdmeProfile:
    """Predicted ADME/PK and CYP450 DDI risk profile for a candidate molecule."""

    smiles: str
    log_s_esol: float
    log_papp_caco2: float
    log_bb: float
    fu_plasma: float
    cl_int_uL_min_mg: float
    hepatic_extraction: float
    pgp_efflux_prob: float
    cyp3a4_ic50_uM: float
    cyp2d6_ic50_uM: float
    cyp2c9_ic50_uM: float
    cyp3a4_ddi_auc_ratio: float
    epistemic_std: float


@dataclass
class DockingPose:
    """Scored 3D ligand pose inside a receptor pocket."""

    smiles: str
    coords: np.ndarray
    total_affinity_kcal: float
    gauss1_score: float
    gauss2_score: float
    repulsion_score: float
    hydrophobic_score: float
    hbond_score: float
    pi_stacking_score: float
    torsion_penalty: float
    hbond_contacts: List[Tuple[int, str]]
    strain_energy: float


@dataclass
class ProtacOutcome:
    """Equilibrium and kinetic characterization of a PROTAC ternary complex."""

    kd_target_nM: float
    kd_e3_nM: float
    cooperativity_alpha: float
    hook_peak_nM: float
    tf50_nM: float
    dc50_nM: float
    dmax_fraction: float
    peak_ternary_fraction: float
    dose_grid_nM: np.ndarray
    ternary_curve_uM: np.ndarray
    time_grid_h: np.ndarray
    target_protein_fraction: np.ndarray


@dataclass
class OptimizationCandidate:
    """Evaluated molecule discovered during MCTS/Pareto lead optimization."""

    smiles: str
    canonical_smiles: str
    mpo_score: float
    docking_kcal: float
    qed: float
    sa_score: float
    log_bb: float
    ddi_ratio: float
    tanimoto_to_seed: float
    depth: int
    parent_smiles: str
    objectives: Tuple[float, ...]
