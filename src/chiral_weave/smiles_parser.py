"""Pure-Python SMILES/SMARTS parser, Horton GF(2) SSSR ring perception, Hueckel aromaticity, and canonicalizer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from chiral_weave.chem_types import (
    ELEMENT_TABLE,
    ORGANIC_SUBSET,
    Atom,
    Bond,
    BondOrder,
    Hybridization,
    MoleculeGraph,
    StereoConfig,
)


class SmilesParseError(ValueError):
    """Raised when a SMILES or SMARTS string violates lexical or valence syntax."""


@dataclass
class _BracketAtomSpec:
    symbol: str
    is_aromatic: bool
    isotope: Optional[int]
    stereo: StereoConfig
    explicit_h: int
    formal_charge: int


def _parse_bracket_atom(token: str) -> _BracketAtomSpec:
    """Parse the inside of a bracket atom [...] without outer brackets."""
    if not token:
        raise SmilesParseError("Empty bracket atom '[]'.")
    idx = 0
    n = len(token)

    # Optional isotope
    iso_start = idx
    while idx < n and token[idx].isdigit():
        idx += 1
    isotope = int(token[iso_start:idx]) if idx > iso_start else None

    if idx >= n:
        raise SmilesParseError(f"Missing element symbol in bracket atom [{token}].")

    # Element symbol (aromatic lowercase single char or uppercase + optional lowercase)
    ch = token[idx]
    is_aromatic = False
    if ch.islower():
        symbol = ch.upper()
        is_aromatic = True
        idx += 1
    elif ch.isupper():
        if idx + 1 < n and token[idx + 1].islower():
            two = token[idx : idx + 2]
            if two in ELEMENT_TABLE:
                symbol = two
                idx += 2
            else:
                symbol = ch
                idx += 1
        else:
            symbol = ch
            idx += 1
    else:
        raise SmilesParseError(f"Invalid element start '{ch}' in bracket atom [{token}].")

    if symbol not in ELEMENT_TABLE:
        raise SmilesParseError(f"Unknown element '{symbol}' in bracket atom [{token}].")

    # Optional chirality (@ or @@)
    stereo = StereoConfig.NONE
    if idx < n and token[idx] == "@":
        if idx + 1 < n and token[idx + 1] == "@":
            stereo = StereoConfig.TETRAHEDRAL_CW
            idx += 2
        else:
            stereo = StereoConfig.TETRAHEDRAL_CCW
            idx += 1

    # Optional explicit hydrogen count (H or H<digits>)
    explicit_h = 0
    if idx < n and token[idx] == "H":
        idx += 1
        h_start = idx
        while idx < n and token[idx].isdigit():
            idx += 1
        explicit_h = int(token[h_start:idx]) if idx > h_start else 1

    # Optional formal charge (+, -, +n, -n, ++, --)
    formal_charge = 0
    if idx < n and token[idx] in ("+", "-"):
        sign = 1 if token[idx] == "+" else -1
        idx += 1
        if idx < n and token[idx].isdigit():
            c_start = idx
            while idx < n and token[idx].isdigit():
                idx += 1
            formal_charge = sign * int(token[c_start:idx])
        else:
            count = 1
            while idx < n and token[idx] == ("+" if sign > 0 else "-"):
                count += 1
                idx += 1
            formal_charge = sign * count

    # Optional atom-map class :num
    if idx < n and token[idx] == ":":
        idx += 1
        while idx < n and token[idx].isdigit():
            idx += 1

    if idx != n:
        raise SmilesParseError(f"Unexpected trailing characters '{token[idx:]}' in [{token}].")

    return _BracketAtomSpec(
        symbol=symbol,
        is_aromatic=is_aromatic,
        isotope=isotope,
        stereo=stereo,
        explicit_h=explicit_h,
        formal_charge=formal_charge,
    )


def _compute_sssr(num_atoms: int, bonds: Sequence[Bond]) -> List[Tuple[int, ...]]:
    """Compute Smallest Set of Smallest Rings (SSSR) using Horton's shortest-cycle + GF(2) reduction.

    Finds the exact cycle-space basis of minimum total ring size across all
    connected components of the molecular graph.
    """
    if num_atoms < 3 or len(bonds) < 3:
        return []

    adj: List[List[Tuple[int, int]]] = [[] for _ in range(num_atoms)]
    for b in bonds:
        adj[b.begin_idx].append((b.end_idx, b.index))
        adj[b.end_idx].append((b.begin_idx, b.index))

    # Determine connected components and cyclomatic number mu = E - V + C
    visited = [False] * num_atoms
    num_components = 0
    for start in range(num_atoms):
        if visited[start]:
            continue
        num_components += 1
        queue = deque([start])
        visited[start] = True
        while queue:
            u = queue.popleft()
            for v, _ in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)

    mu = len(bonds) - num_atoms + num_components
    if mu <= 0:
        return []

    # Generate candidate shortest cycles passing through (root, edge) via BFS shortest paths
    candidate_cycles: Dict[int, Tuple[int, Tuple[int, ...]]] = {}
    for root in range(num_atoms):
        if len(adj[root]) < 2:
            continue
        dist = [-1] * num_atoms
        parent = [-1] * num_atoms
        parent_edge = [-1] * num_atoms
        dist[root] = 0
        q = deque([root])
        while q:
            u = q.popleft()
            for v, e_idx in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    parent_edge[v] = e_idx
                    q.append(v)

        for b in bonds:
            u, v, e_idx = b.begin_idx, b.end_idx, b.index
            if dist[u] < 0 or dist[v] < 0:
                continue
            if parent[u] == v or parent[v] == u:
                continue
            # Trace paths from u and v back to root to check if cycle is simple and passes root
            path_u: List[int] = []
            edges_mask = 1 << e_idx
            curr = u
            while curr != -1:
                path_u.append(curr)
                if parent_edge[curr] != -1:
                    edges_mask |= 1 << parent_edge[curr]
                curr = parent[curr]
            path_v: List[int] = []
            curr = v
            while curr != -1:
                path_v.append(curr)
                if parent_edge[curr] != -1:
                    edges_mask |= 1 << parent_edge[curr]
                curr = parent[curr]
            # Common vertices other than root invalidate simple Horton cycle
            set_u = set(path_u[:-1])
            set_v = set(path_v[:-1])
            if set_u & set_v:
                continue
            ordered = tuple(path_u[:-1] + [root] + list(reversed(path_v[:-1])))
            if len(ordered) >= 3 and edges_mask not in candidate_cycles:
                candidate_cycles[edges_mask] = (len(ordered), ordered)

    # Sort candidate cycles by length ascending and reduce via GF(2) Gaussian elimination
    sorted_candidates = sorted(candidate_cycles.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    basis_pivots: Dict[int, int] = {}
    sssr: List[Tuple[int, ...]] = []

    for mask, (_, cycle_atoms) in sorted_candidates:
        m = mask
        while m:
            lsb = m & -m
            if lsb in basis_pivots:
                m ^= basis_pivots[lsb]
            else:
                basis_pivots[lsb] = m
                sssr.append(cycle_atoms)
                break
        if len(sssr) == mu:
            break

    return sssr


def _ring_pi_electrons(mol: MoleculeGraph, ring: Tuple[int, ...]) -> Optional[int]:
    """Count Hueckel pi electrons contributed by each atom in a candidate planar ring.

    Returns None if an atom cannot sustain planar sp2 conjugation (e.g., sp3 CH2).
    """
    pi_total = 0
    ring_set = set(ring)
    for idx in ring:
        atom = mol.atoms[idx]
        if atom.symbol not in ("C", "N", "O", "S", "P", "B"):
            return None

        inc = mol.incident_bonds(idx)
        has_ring_double = any(
            b.order == BondOrder.DOUBLE and b.other(idx) in ring_set for b in inc
        )
        has_aromatic = atom.is_aromatic or any(b.order == BondOrder.AROMATIC for b in inc)
        has_exo_double = any(
            b.order == BondOrder.DOUBLE and b.other(idx) not in ring_set for b in inc
        )

        if atom.symbol == "C":
            if has_aromatic or has_ring_double:
                pi_total += 1
            elif has_exo_double:
                # Exocyclic =O on carbonyl carbon contributes 0 pi electrons to ring
                pi_total += 0
            elif atom.formal_charge == 1:
                pi_total += 0
            elif atom.formal_charge == -1:
                pi_total += 2
            else:
                return None
        elif atom.symbol == "B":
            if has_aromatic or has_ring_double or len(inc) <= 3:
                pi_total += 0
            else:
                return None
        elif atom.symbol == "N":
            if has_aromatic:
                if atom.total_h >= 1 or len(inc) == 3:
                    pi_total += 2
                else:
                    pi_total += 1
            elif has_ring_double:
                pi_total += 1
            elif atom.formal_charge == 0 and len(inc) + atom.total_h == 3:
                pi_total += 2
            elif atom.formal_charge == 1 and has_exo_double:
                pi_total += 1
            else:
                return None
        elif atom.symbol in ("O", "S"):
            if has_aromatic or (len(inc) == 2 and not has_exo_double and atom.total_h == 0):
                pi_total += 2
            else:
                return None
        elif atom.symbol == "P":
            if has_ring_double or has_aromatic:
                pi_total += 1
            elif len(inc) + atom.total_h == 3:
                pi_total += 2
            else:
                return None
    return pi_total


def _perceive_aromaticity(mol: MoleculeGraph) -> None:
    """Detect Hueckel 4n+2 aromatic rings and update atom/bond aromatic flags."""
    # First detect rings that satisfy Hueckel 4n + 2 rule
    aromatic_atoms: Set[int] = {a.index for a in mol.atoms if a.is_aromatic}
    aromatic_bonds: Set[int] = {b.index for b in mol.bonds if b.is_aromatic}

    for ring in mol.sssr_rings:
        if len(ring) < 5 or len(ring) > 10:
            continue
        pi_e = _ring_pi_electrons(mol, ring)
        if pi_e is not None and pi_e >= 2 and (pi_e - 2) % 4 == 0:
            aromatic_atoms.update(ring)
            for k in range(len(ring)):
                u = ring[k]
                v = ring[(k + 1) % len(ring)]
                bond = mol.find_bond(u, v)
                if bond is not None:
                    aromatic_bonds.add(bond.index)

    for a in mol.atoms:
        if a.index in aromatic_atoms:
            a.is_aromatic = True
    for b in mol.bonds:
        if b.index in aromatic_bonds or (
            b.begin_idx in aromatic_atoms and b.end_idx in aromatic_atoms and b.in_ring
        ):
            b.is_aromatic = True
            b.order = BondOrder.AROMATIC
            b.is_conjugated = True


def _kekulize_aromatic_valence(mol: MoleculeGraph) -> Dict[int, float]:
    """Compute effective Kekule bond-order sum for each atom via maximum matching on aromatic subgraphs.

    Allows accurate implicit hydrogen calculation even when input SMILES uses
    aromatic lowercase tokens (e.g., c1ccccc1, c1ccncc1, c1cc[nH]c1).
    """
    kekule_sum: Dict[int, float] = {a.index: 0.0 for a in mol.atoms}
    # Start with non-aromatic bonds
    for b in mol.bonds:
        if b.order != BondOrder.AROMATIC:
            kekule_sum[b.begin_idx] += b.order.value
            kekule_sum[b.end_idx] += b.order.value
        else:
            kekule_sum[b.begin_idx] += 1.0
            kekule_sum[b.end_idx] += 1.0

    # Collect atoms needing a pi-bond (unsaturated sp2 aromatic atoms without lone pair)
    need_double: Set[int] = set()
    for a in mol.atoms:
        if not a.is_aromatic:
            continue
        deg = mol.degree(a.index) + a.explicit_h
        if a.symbol == "C":
            # Carbon in aromatic ring needs total Kekule valence 4; with 3 bonds needs 1 double bond
            if a.formal_charge == 0:
                need_double.add(a.index)
        elif a.symbol == "N":
            # Pyridine-like N (degree 2, 0 explicit H, neutral) needs a double bond
            if deg == 2 and a.explicit_h == 0 and a.formal_charge == 0:
                need_double.add(a.index)
            elif deg == 3 and a.formal_charge == 1:
                need_double.add(a.index)
        elif a.symbol == "P":
            if deg == 2 and a.explicit_h == 0:
                need_double.add(a.index)

    if not need_double:
        return kekule_sum

    # Build aromatic subgraph over need_double and find maximum matching via augmenting paths
    sub_adj: Dict[int, List[int]] = {u: [] for u in need_double}
    for b in mol.bonds:
        if b.order == BondOrder.AROMATIC and b.begin_idx in need_double and b.end_idx in need_double:
            sub_adj[b.begin_idx].append(b.end_idx)
            sub_adj[b.end_idx].append(b.begin_idx)

    match: Dict[int, int] = {}

    def _augment(u: int, seen: Set[int]) -> bool:
        for v in sub_adj[u]:
            if v in seen:
                continue
            seen.add(v)
            if v not in match or _augment(match[v], seen):
                match[v] = u
                match[u] = v
                return True
        return False

    for u in sorted(need_double):
        if u not in match:
            _augment(u, set())

    for u in match:
        kekule_sum[u] += 1.0

    return kekule_sum


def _assign_valences_and_hybridization(mol: MoleculeGraph, bracket_flags: Dict[int, bool]) -> None:
    """Compute implicit hydrogens, conjugation, and orbital hybridization for all atoms."""
    kekule_sum = _kekulize_aromatic_valence(mol)

    for b in mol.bonds:
        if b.order in (BondOrder.DOUBLE, BondOrder.TRIPLE, BondOrder.AROMATIC):
            b.is_conjugated = True

    for atom in mol.atoms:
        spec = ELEMENT_TABLE[atom.symbol]
        inc = mol.incident_bonds(atom.index)
        k_val = int(round(kekule_sum[atom.index])) + atom.explicit_h

        if bracket_flags.get(atom.index, False):
            # Bracket atoms specify explicit_h literally; never invent extra implicit H
            atom.implicit_h = 0
        else:
            # Organic subset atom infers implicit H to reach lowest compatible valence
            target_valence = spec.default_valences[-1]
            for v in spec.default_valences:
                if v >= k_val:
                    target_valence = v
                    break
            atom.implicit_h = max(0, target_valence - k_val)

        # Assign hybridization
        if atom.symbol == "H":
            atom.hybridization = Hybridization.S
        elif any(b.order == BondOrder.TRIPLE for b in inc):
            atom.hybridization = Hybridization.SP
        elif sum(1 for b in inc if b.order == BondOrder.DOUBLE) >= 2:
            atom.hybridization = Hybridization.SP
        elif atom.is_aromatic or any(b.order == BondOrder.DOUBLE for b in inc):
            atom.hybridization = Hybridization.SP2
        else:
            atom.hybridization = Hybridization.SP3


def parse_smiles(smiles: str, name: str = "") -> MoleculeGraph:
    """Parse a SMILES string into a fully perceived MoleculeGraph.

    Performs lexical tokenization, ring-closure linking, Horton GF(2) SSSR ring
    perception, Hueckel 4n+2 aromaticity perception, Kekule valence resolution,
    implicit hydrogen assignment, and canonical SMILES generation.
    """
    text = smiles.strip()
    if not text:
        raise SmilesParseError("Empty SMILES string.")

    atoms: List[Atom] = []
    bonds: List[Bond] = []
    bracket_flags: Dict[int, bool] = {}

    branch_stack: List[Optional[int]] = []
    ring_opens: Dict[int, Tuple[int, Optional[BondOrder], str]] = {}
    prev_atom: Optional[int] = None
    pending_order: Optional[BondOrder] = None
    pending_dir: str = ""

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        if ch == "(":
            branch_stack.append(prev_atom)
            i += 1
            continue

        if ch == ")":
            if not branch_stack:
                raise SmilesParseError(f"Unmatched ')' at position {i} in {smiles!r}.")
            prev_atom = branch_stack.pop()
            pending_order = None
            pending_dir = ""
            i += 1
            continue

        if ch == ".":
            prev_atom = None
            pending_order = None
            pending_dir = ""
            i += 1
            continue

        if ch in ("-", "=", "#", ":", "/", "\\"):
            order_map = {
                "-": BondOrder.SINGLE,
                "=": BondOrder.DOUBLE,
                "#": BondOrder.TRIPLE,
                ":": BondOrder.AROMATIC,
                "/": BondOrder.SINGLE,
                "\\": BondOrder.SINGLE,
            }
            pending_order = order_map[ch]
            pending_dir = ch if ch in ("/", "\\") else ""
            i += 1
            continue

        # Ring closure digit or %nn
        if ch.isdigit() or ch == "%":
            if prev_atom is None:
                raise SmilesParseError(f"Ring closure without preceding atom at position {i}.")
            if ch == "%":
                if i + 2 >= n or not text[i + 1 : i + 3].isdigit():
                    raise SmilesParseError(f"Invalid '%nn' ring closure at position {i}.")
                ring_id = int(text[i + 1 : i + 3])
                i += 3
            else:
                ring_id = int(ch)
                i += 1

            if ring_id in ring_opens:
                other_idx, open_order, open_dir = ring_opens.pop(ring_id)
                if other_idx == prev_atom:
                    raise SmilesParseError(f"Self-loop ring closure {ring_id} on atom {prev_atom}.")
                if any(
                    (b.begin_idx == other_idx and b.end_idx == prev_atom)
                    or (b.begin_idx == prev_atom and b.end_idx == other_idx)
                    for b in bonds
                ):
                    raise SmilesParseError(f"Duplicate bond created by ring closure {ring_id}.")
                if open_order is not None and pending_order is not None and open_order != pending_order:
                    raise SmilesParseError(f"Conflicting bond orders at ring closure {ring_id}.")
                chosen_order = pending_order or open_order
                if chosen_order is None:
                    if atoms[other_idx].is_aromatic and atoms[prev_atom].is_aromatic:
                        chosen_order = BondOrder.AROMATIC
                    else:
                        chosen_order = BondOrder.SINGLE
                bonds.append(
                    Bond(
                        index=len(bonds),
                        begin_idx=other_idx,
                        end_idx=prev_atom,
                        order=chosen_order,
                        is_aromatic=(chosen_order == BondOrder.AROMATIC),
                        stereo_dir=pending_dir or open_dir,
                    )
                )
                pending_order = None
                pending_dir = ""
            else:
                ring_opens[ring_id] = (prev_atom, pending_order, pending_dir)
                pending_order = None
                pending_dir = ""
            continue

        # Atom token: bracket or organic subset
        if ch == "[":
            close_idx = text.find("]", i + 1)
            if close_idx == -1:
                raise SmilesParseError(f"Unclosed '[' at position {i} in {smiles!r}.")
            spec = _parse_bracket_atom(text[i + 1 : close_idx])
            atom_idx = len(atoms)
            atoms.append(
                Atom(
                    index=atom_idx,
                    symbol=spec.symbol,
                    atomic_number=ELEMENT_TABLE[spec.symbol].atomic_number,
                    formal_charge=spec.formal_charge,
                    isotope=spec.isotope,
                    explicit_h=spec.explicit_h,
                    is_aromatic=spec.is_aromatic,
                    stereo=spec.stereo,
                )
            )
            bracket_flags[atom_idx] = True
            i = close_idx + 1
        elif ch.isalpha():
            if ch.islower():
                sym = ch.upper()
                if sym not in ("B", "C", "N", "O", "P", "S"):
                    raise SmilesParseError(f"Invalid aromatic organic-subset atom '{ch}' at {i}.")
                is_arom = True
                i += 1
            else:
                if i + 1 < n and text[i : i + 2] in ("Cl", "Br"):
                    sym = text[i : i + 2]
                    i += 2
                else:
                    sym = ch
                    i += 1
                if sym not in ORGANIC_SUBSET:
                    raise SmilesParseError(f"Atom '{sym}' outside organic subset requires brackets.")
                is_arom = False

            atom_idx = len(atoms)
            atoms.append(
                Atom(
                    index=atom_idx,
                    symbol=sym,
                    atomic_number=ELEMENT_TABLE[sym].atomic_number,
                    is_aromatic=is_arom,
                )
            )
            bracket_flags[atom_idx] = False
        else:
            raise SmilesParseError(f"Unexpected character '{ch}' at position {i} in {smiles!r}.")

        # Connect new atom to prev_atom
        if prev_atom is not None:
            chosen_order = pending_order
            if chosen_order is None:
                if atoms[prev_atom].is_aromatic and atoms[atom_idx].is_aromatic:
                    chosen_order = BondOrder.AROMATIC
                else:
                    chosen_order = BondOrder.SINGLE
            bonds.append(
                Bond(
                    index=len(bonds),
                    begin_idx=prev_atom,
                    end_idx=atom_idx,
                    order=chosen_order,
                    is_aromatic=(chosen_order == BondOrder.AROMATIC),
                    stereo_dir=pending_dir,
                )
            )
        prev_atom = atom_idx
        pending_order = None
        pending_dir = ""

    if branch_stack:
        raise SmilesParseError(f"Unclosed branch '(' in {smiles!r}.")
    if ring_opens:
        raise SmilesParseError(f"Unclosed ring closure(s) {sorted(ring_opens)} in {smiles!r}.")

    mol = MoleculeGraph(atoms=atoms, bonds=bonds, source_smiles=smiles, name=name)
    sssr = _compute_sssr(mol.num_atoms, mol.bonds)
    mol.sssr_rings = sssr

    ring_membership: Dict[int, List[int]] = {a.index: [] for a in mol.atoms}
    ring_edges: Set[Tuple[int, int]] = set()
    for ring in sssr:
        for k, a_idx in enumerate(ring):
            ring_membership[a_idx].append(len(ring))
            u, v = a_idx, ring[(k + 1) % len(ring)]
            ring_edges.add((min(u, v), max(u, v)))

    for a in mol.atoms:
        sizes = tuple(sorted(set(ring_membership[a.index])))
        a.ring_sizes = sizes
        a.in_ring = len(sizes) > 0

    for b in mol.bonds:
        key = (min(b.begin_idx, b.end_idx), max(b.begin_idx, b.end_idx))
        b.in_ring = key in ring_edges

    _perceive_aromaticity(mol)
    _assign_valences_and_hybridization(mol, bracket_flags)
    mol.canonical_smiles = to_canonical_smiles(mol)
    return mol


def _morgan_canonical_ranks(mol: MoleculeGraph) -> List[int]:
    """Compute deterministic canonical atom ranks via iterative Morgan-style refinement."""
    n = mol.num_atoms
    if n == 0:
        return []

    # Initial invariant tuple per atom
    invariants: List[Tuple[int, ...]] = []
    for a in mol.atoms:
        invariants.append(
            (
                a.atomic_number,
                mol.degree(a.index),
                a.total_h,
                a.formal_charge + 10,
                1 if a.is_aromatic else 0,
                len(a.ring_sizes),
                a.ring_sizes[0] if a.ring_sizes else 0,
            )
        )

    def _compress(keys: Sequence[Tuple[int, ...]]) -> List[int]:
        uniq = sorted(set(keys))
        mapping = {k: idx for idx, k in enumerate(uniq)}
        return [mapping[k] for k in keys]

    ranks = _compress(invariants)
    prev_classes = -1
    for _ in range(max(4, n)):
        num_classes = len(set(ranks))
        if num_classes == n or num_classes == prev_classes:
            break
        prev_classes = num_classes
        refined: List[Tuple[int, ...]] = []
        for idx in range(n):
            nbr_sigs = sorted(
                (int(round(b.order.value * 2)), ranks[b.other(idx)])
                for b in mol.incident_bonds(idx)
            )
            flat: List[int] = [ranks[idx]]
            for bo_code, r in nbr_sigs:
                flat.append(bo_code)
                flat.append(r)
            refined.append(tuple(flat))
        ranks = _compress(refined)

    # Break any remaining symmetry ties deterministically via BFS tie-break order
    order = sorted(range(n), key=lambda i: (ranks[i], i))
    final_ranks = [0] * n
    for pos, atom_idx in enumerate(order):
        final_ranks[atom_idx] = pos
    return final_ranks


def _format_atom_smiles(atom: Atom, mol: MoleculeGraph) -> str:
    """Format a single atom in canonical SMILES syntax."""
    needs_brackets = (
        atom.symbol not in ORGANIC_SUBSET
        or atom.formal_charge != 0
        or atom.isotope is not None
        or atom.stereo != StereoConfig.NONE
        or (atom.is_aromatic and atom.symbol == "N" and atom.total_h > 0)
        or (atom.is_aromatic and atom.symbol not in ("B", "C", "N", "O", "P", "S"))
    )
    if not needs_brackets:
        return atom.symbol.lower() if atom.is_aromatic else atom.symbol

    parts: List[str] = ["["]
    if atom.isotope is not None:
        parts.append(str(atom.isotope))
    parts.append(atom.symbol.lower() if atom.is_aromatic else atom.symbol)
    if atom.stereo in (StereoConfig.TETRAHEDRAL_CCW, StereoConfig.TETRAHEDRAL_CW):
        parts.append(atom.stereo.value)
    if atom.total_h == 1:
        parts.append("H")
    elif atom.total_h > 1:
        parts.append(f"H{atom.total_h}")
    if atom.formal_charge > 0:
        parts.append("+" if atom.formal_charge == 1 else f"+{atom.formal_charge}")
    elif atom.formal_charge < 0:
        parts.append("-" if atom.formal_charge == -1 else f"{atom.formal_charge}")
    parts.append("]")
    return "".join(parts)


def _format_bond_smiles(bond: Bond, a_from: Atom, a_to: Atom) -> str:
    """Format bond symbol for canonical SMILES emission."""
    if bond.order == BondOrder.AROMATIC or (a_from.is_aromatic and a_to.is_aromatic and bond.is_aromatic):
        return ""
    if bond.order == BondOrder.SINGLE:
        return "-" if (a_from.is_aromatic and a_to.is_aromatic) else ""
    if bond.order == BondOrder.DOUBLE:
        return "="
    if bond.order == BondOrder.TRIPLE:
        return "#"
    return ""


def to_canonical_smiles(mol: MoleculeGraph) -> str:
    """Serialize MoleculeGraph to a deterministic canonical SMILES string."""
    if mol.num_atoms == 0:
        return ""

    ranks = _morgan_canonical_ranks(mol)
    visited = [False] * mol.num_atoms
    ring_closures_at: Dict[int, List[Tuple[int, str]]] = {i: [] for i in range(mol.num_atoms)}
    parent_map: Dict[int, int] = {}

    # Pre-scan spanning tree in canonical rank order to identify ring closure edges
    components_trees: List[int] = []
    seen_scan = [False] * mol.num_atoms
    in_stack_or_done = [False] * mol.num_atoms
    next_ring_num = 1
    closure_edges: Set[int] = set()

    def _scan_dfs(u: int, p: int) -> None:
        nonlocal next_ring_num
        seen_scan[u] = True
        in_stack_or_done[u] = True
        nbrs = sorted(mol.neighbors(u), key=lambda v: ranks[v])
        for v in nbrs:
            if v == p:
                continue
            bond = mol.find_bond(u, v)
            assert bond is not None
            if not seen_scan[v]:
                parent_map[v] = u
                _scan_dfs(v, u)
            elif bond.index not in closure_edges and parent_map.get(u) != v:
                closure_edges.add(bond.index)
                rnum = next_ring_num
                next_ring_num += 1
                rtok = str(rnum) if rnum < 10 else f"%{rnum}"
                bsym = _format_bond_smiles(bond, mol.atoms[u], mol.atoms[v])
                ring_closures_at[u].append((rnum, bsym + rtok))
                ring_closures_at[v].append((rnum, rtok))

    for root in sorted(range(mol.num_atoms), key=lambda idx: ranks[idx]):
        if not seen_scan[root]:
            components_trees.append(root)
            _scan_dfs(root, -1)

    def _emit_dfs(u: int, p: int) -> str:
        visited[u] = True
        tokens: List[str] = [_format_atom_smiles(mol.atoms[u], mol)]
        for _, rstr in sorted(ring_closures_at[u], key=lambda item: item[0]):
            tokens.append(rstr)

        children: List[Tuple[int, Bond]] = []
        for v in sorted(mol.neighbors(u), key=lambda idx: ranks[idx]):
            if v == p or visited[v]:
                continue
            bond = mol.find_bond(u, v)
            if bond is not None and bond.index not in closure_edges:
                children.append((v, bond))

        for pos, (v, bond) in enumerate(children):
            if visited[v]:
                continue
            bsym = _format_bond_smiles(bond, mol.atoms[u], mol.atoms[v])
            sub = bsym + _emit_dfs(v, u)
            if pos < len(children) - 1:
                tokens.append(f"({sub})")
            else:
                tokens.append(sub)
        return "".join(tokens)

    comp_strings = [_emit_dfs(root, -1) for root in components_trees]
    return ".".join(sorted(comp_strings))


# ---------------------------------------------------------------------------
# SMARTS Substructure Parser and Subgraph Isomorphism Matcher
# ---------------------------------------------------------------------------


@dataclass
class _SmartsAtomQuery:
    expr: str


@dataclass
class _SmartsBondQuery:
    expr: str


@dataclass
class _SmartsPattern:
    atoms: List[_SmartsAtomQuery]
    bonds: List[Tuple[int, int, _SmartsBondQuery]]
    adj: Dict[int, List[Tuple[int, _SmartsBondQuery]]]


def _eval_smarts_atom_primitive(prim: str, atom: Atom, mol: MoleculeGraph) -> bool:
    """Evaluate a single SMARTS atom primitive against an Atom in MoleculeGraph."""
    if prim == "*":
        return True
    if prim == "a":
        return atom.is_aromatic
    if prim == "A":
        return not atom.is_aromatic
    if prim == "R":
        return atom.in_ring
    if prim.startswith("R") and prim[1:].isdigit():
        return len(atom.ring_sizes) >= int(prim[1:]) if int(prim[1:]) > 0 else not atom.in_ring
    if prim == "r":
        return atom.in_ring
    if prim.startswith("r") and prim[1:].isdigit():
        return int(prim[1:]) in atom.ring_sizes
    if prim.startswith("#") and prim[1:].isdigit():
        return atom.atomic_number == int(prim[1:])
    if prim == "H":
        return atom.total_h >= 1
    if prim.startswith("H") and prim[1:].isdigit():
        return atom.total_h == int(prim[1:])
    if prim.startswith("X") and prim[1:].isdigit():
        return (mol.degree(atom.index) + atom.total_h) == int(prim[1:])
    if prim.startswith("D") and prim[1:].isdigit():
        return mol.degree(atom.index) == int(prim[1:])
    if prim == "+":
        return atom.formal_charge > 0
    if prim.startswith("+") and prim[1:].isdigit():
        return atom.formal_charge == int(prim[1:])
    if prim == "-":
        return atom.formal_charge < 0
    if prim.startswith("-") and prim[1:].isdigit():
        return atom.formal_charge == -int(prim[1:])
    if prim in ("c", "n", "o", "s", "p", "b"):
        return atom.symbol == prim.upper() and atom.is_aromatic
    if prim in ELEMENT_TABLE:
        return atom.symbol == prim and not atom.is_aromatic
    return False


def _match_smarts_atom_expr(expr: str, atom: Atom, mol: MoleculeGraph) -> bool:
    """Evaluate boolean SMARTS atom expression supporting ',', '&', ';', and '!'."""
    # Split OR clauses on ','
    for or_clause in expr.split(","):
        # Split AND clauses on ';' or '&'
        and_tokens: List[str] = []
        for semi in or_clause.split(";"):
            and_tokens.extend(semi.split("&"))
        ok = True
        for tok in and_tokens:
            tok = tok.strip()
            if not tok:
                continue
            negated = tok.startswith("!")
            core_tok = tok[1:] if negated else tok
            res = _eval_smarts_atom_primitive(core_tok, atom, mol)
            if negated:
                res = not res
            if not res:
                ok = False
                break
        if ok:
            return True
    return False


def _split_smarts_and_tokens(body: str) -> str:
    """Insert explicit '&' between adjacent SMARTS atom primitives inside [...]."""
    out: List[str] = []
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch in (",", ";", "&"):
            out.append(ch)
            i += 1
            continue
        neg = ""
        if ch == "!":
            neg = "!"
            i += 1
            if i >= n:
                break
            ch = body[i]
        if ch in ("#", "R", "r", "H", "X", "D", "+", "-"):
            j = i + 1
            while j < n and body[j].isdigit():
                j += 1
            tok = neg + body[i:j]
            i = j
        elif ch.isupper() and i + 1 < n and body[i : i + 2] in ("Cl", "Br", "Si"):
            tok = neg + body[i : i + 2]
            i += 2
        else:
            tok = neg + ch
            i += 1
        if out and out[-1] not in (",", ";", "&"):
            out.append("&")
        out.append(tok)
    return "".join(out)


def _match_smarts_bond_expr(expr: str, bond: Bond) -> bool:
    """Evaluate SMARTS bond query expression against Bond."""
    for or_clause in expr.split(","):
        ok = True
        for tok in or_clause.split("&"):
            tok = tok.strip()
            if not tok:
                continue
            neg = tok.startswith("!")
            p = tok[1:] if neg else tok
            if p == "~":
                res = True
            elif p == "-":
                res = bond.order == BondOrder.SINGLE and not bond.is_aromatic
            elif p == "=":
                res = bond.order == BondOrder.DOUBLE
            elif p == "#":
                res = bond.order == BondOrder.TRIPLE
            elif p == ":":
                res = bond.is_aromatic or bond.order == BondOrder.AROMATIC
            elif p == "@":
                res = bond.in_ring
            else:
                res = False
            if neg:
                res = not res
            if not res:
                ok = False
                break
        if ok:
            return True
    return False


def _parse_smarts(smarts: str) -> _SmartsPattern:
    """Parse a SMARTS pattern string into a query graph."""
    text = smarts.strip()
    if not text:
        raise SmilesParseError("Empty SMARTS pattern.")
    q_atoms: List[_SmartsAtomQuery] = []
    q_bonds: List[Tuple[int, int, _SmartsBondQuery]] = []
    stack: List[Optional[int]] = []
    rings: Dict[int, Tuple[int, str]] = {}
    prev: Optional[int] = None
    pending_bond = ""
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch == "(":
            stack.append(prev)
            i += 1
        elif ch == ")":
            prev = stack.pop()
            pending_bond = ""
            i += 1
        elif ch in ("-", "=", "#", ":", "~", "@", "!", "&", ","):
            # Bond token accumulation
            j = i
            while j < n and text[j] in ("-", "=", "#", ":", "~", "@", "!", "&", ","):
                j += 1
            pending_bond = text[i:j]
            i = j
        elif ch.isdigit():
            rnum = int(ch)
            i += 1
            assert prev is not None
            if rnum in rings:
                other, open_b = rings.pop(rnum)
                b_expr = pending_bond or open_b or "-,:"
                q_bonds.append((other, prev, _SmartsBondQuery(b_expr)))
            else:
                rings[rnum] = (prev, pending_bond)
            pending_bond = ""
        elif ch == "[":
            end = text.find("]", i + 1)
            if end == -1:
                raise SmilesParseError(f"Unclosed '[' in SMARTS {smarts!r}.")
            expr = _split_smarts_and_tokens(text[i + 1 : end])
            q_idx = len(q_atoms)
            q_atoms.append(_SmartsAtomQuery(expr))
            if prev is not None:
                b_expr = pending_bond or "-,:"
                q_bonds.append((prev, q_idx, _SmartsBondQuery(b_expr)))
            prev = q_idx
            pending_bond = ""
            i = end + 1
        else:
            if ch.isupper() and i + 1 < n and text[i : i + 2] in ("Cl", "Br"):
                sym = text[i : i + 2]
                i += 2
            else:
                sym = ch
                i += 1
            q_idx = len(q_atoms)
            q_atoms.append(_SmartsAtomQuery(sym))
            if prev is not None:
                if not pending_bond:
                    if sym.islower() and q_atoms[prev].expr.islower():
                        b_expr = ":"
                    else:
                        b_expr = "-,:"
                else:
                    b_expr = pending_bond
                q_bonds.append((prev, q_idx, _SmartsBondQuery(b_expr)))
            prev = q_idx
            pending_bond = ""

    adj: Dict[int, List[Tuple[int, _SmartsBondQuery]]] = {k: [] for k in range(len(q_atoms))}
    for u, v, bq in q_bonds:
        adj[u].append((v, bq))
        adj[v].append((u, bq))
    return _SmartsPattern(atoms=q_atoms, bonds=q_bonds, adj=adj)


def find_smarts_matches(mol: MoleculeGraph, smarts: str, unique: bool = True) -> List[Tuple[int, ...]]:
    """Find all subgraph isomorphism matches of a SMARTS pattern in MoleculeGraph."""
    pat = _parse_smarts(smarts)
    m = len(pat.atoms)
    if m == 0 or mol.num_atoms < m:
        return []

    candidates: List[List[int]] = []
    for q_atom in pat.atoms:
        cands = [
            a.index
            for a in mol.atoms
            if _match_smarts_atom_expr(q_atom.expr, a, mol)
        ]
        if not cands:
            return []
        candidates.append(cands)

    # Order query vertices by smallest candidate list for fast pruning
    order = sorted(range(m), key=lambda q: (len(candidates[q]), -len(pat.adj[q])))
    pos_of = {q: pos for pos, q in enumerate(order)}
    mapping: Dict[int, int] = {}
    used_targets: Set[int] = set()
    matches: List[Tuple[int, ...]] = []
    seen_sets: Set[frozenset[int]] = set()

    def _backtrack(step: int) -> None:
        if step == m:
            tup = tuple(mapping[q] for q in range(m))
            if unique:
                fs = frozenset(tup)
                if fs in seen_sets:
                    return
                seen_sets.add(fs)
            matches.append(tup)
            return

        q = order[step]
        for t in candidates[q]:
            if t in used_targets:
                continue
            # Check all bonds to already-mapped query neighbors
            ok = True
            for q_nbr, bq in pat.adj[q]:
                if pos_of[q_nbr] < step:
                    t_nbr = mapping[q_nbr]
                    bond = mol.find_bond(t, t_nbr)
                    if bond is None or not _match_smarts_bond_expr(bq.expr, bond):
                        ok = False
                        break
            if not ok:
                continue
            mapping[q] = t
            used_targets.add(t)
            _backtrack(step + 1)
            used_targets.remove(t)
            del mapping[q]

    _backtrack(0)
    return matches


def has_smarts_match(mol: MoleculeGraph, smarts: str) -> bool:
    """Return True if MoleculeGraph contains at least one SMARTS substructure match."""
    return len(find_smarts_matches(mol, smarts, unique=False)) > 0
