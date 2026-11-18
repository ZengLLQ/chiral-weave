"""Tests for SMILES/SMARTS parsing, Horton SSSR rings, Hueckel aromaticity, and canonicalization."""

from __future__ import annotations

import pytest

from chiral_weave.chem_types import BondOrder, StereoConfig
from chiral_weave.smiles_parser import (
    SmilesParseError,
    find_smarts_matches,
    has_smarts_match,
    parse_smiles,
)


def test_kekule_and_aromatic_benzene_canonical_equivalence() -> None:
    """Kekule C1=CC=CC=C1 and aromatic c1ccccc1 must produce identical canonical SMILES."""
    mol_kek = parse_smiles("C1=CC=CC=C1")
    mol_arom = parse_smiles("c1ccccc1")
    assert mol_kek.canonical_smiles == mol_arom.canonical_smiles
    assert len(mol_kek.sssr_rings) == 1
    assert all(a.is_aromatic for a in mol_kek.atoms)
    assert all(b.order == BondOrder.AROMATIC for b in mol_kek.bonds)


def test_indole_pyrrole_nitrogen_aromaticity_and_h_count() -> None:
    """Indole [nH] must carry 1 total H and fuse a 5- and 6-membered aromatic SSSR basis."""
    indole = parse_smiles("c1ccc2[nH]ccc2c1")
    assert len(indole.sssr_rings) == 2
    ring_sizes = sorted(len(r) for r in indole.sssr_rings)
    assert ring_sizes == [5, 6]
    n_atoms = [a for a in indole.atoms if a.symbol == "N"]
    assert len(n_atoms) == 1
    assert n_atoms[0].is_aromatic
    assert n_atoms[0].total_h == 1
    assert indole.molecular_formula() == "C8H7N"


def test_polycyclic_sssr_cyclomatic_basis_counts() -> None:
    """Verify exact mu = E - V + C SSSR ring counts for naphthalene, adamantane, and cubane."""
    naphthalene = parse_smiles("c1ccc2ccccc2c1")
    assert len(naphthalene.sssr_rings) == 2

    adamantane = parse_smiles("C1C2CC3CC1CC(C2)C3")
    assert len(adamantane.sssr_rings) == 3

    cubane = parse_smiles("C12C3C4C1C5C4C3C25")
    assert len(cubane.sssr_rings) == 5


def test_canonical_smiles_permutation_invariance() -> None:
    """Different atom-traversal orderings of aspirin and ibuprofen must canonicalize identically."""
    asp1 = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    asp2 = parse_smiles("O=C(O)c1ccccc1OC(=O)C")
    asp3 = parse_smiles("c1ccc(OC(=O)C)c(C(=O)O)c1")
    assert asp1.canonical_smiles == asp2.canonical_smiles == asp3.canonical_smiles

    ibu1 = parse_smiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1")
    ibu2 = parse_smiles("O=C(O)C(C)c1ccc(CC(C)C)cc1")
    assert ibu1.canonical_smiles == ibu2.canonical_smiles


def test_bracket_stereochemistry_and_charged_species() -> None:
    """Bracket chiral markers @ / @@ and formal charges must be preserved accurately."""
    l_ala = parse_smiles("C[C@H](N)C(=O)O")
    d_ala = parse_smiles("C[C@@H](N)C(=O)O")
    assert l_ala.atoms[1].stereo == StereoConfig.TETRAHEDRAL_CCW
    assert d_ala.atoms[1].stereo == StereoConfig.TETRAHEDRAL_CW
    assert l_ala.atoms[1].total_h == 1

    zwitterion = parse_smiles("C[C@H]([NH3+])C(=O)[O-]")
    assert sum(a.formal_charge for a in zwitterion.atoms) == 0
    assert zwitterion.atoms[2].formal_charge == 1
    assert zwitterion.atoms[5].formal_charge == -1


def test_smarts_substructure_matching() -> None:
    """Verify SMARTS pattern matching for carboxylic acids, amides, and aromatic rings."""
    aspirin = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    assert has_smarts_match(aspirin, "[#6](=[#8])[#8H1]")
    assert has_smarts_match(aspirin, "[a]")
    assert not has_smarts_match(aspirin, "[#7]")

    imatinib = parse_smiles("Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1")
    amide_hits = find_smarts_matches(imatinib, "[#7H1]-[C](=[O])")
    assert len(amide_hits) == 1
    arom_n_hits = find_smarts_matches(imatinib, "[n]")
    assert len(arom_n_hits) == 3


def test_invalid_smiles_raises_parse_error() -> None:
    """Malformed SMILES strings must raise SmilesParseError deterministically."""
    for bad in ["", "CC((C)", "C1CCC", "C1=C-1", "C1=1", "[C", "[Xx]"]:
        with pytest.raises(SmilesParseError):
            parse_smiles(bad)


def test_multidigit_ring_closure_and_salt_fragments() -> None:
    """Verify %10 ring closure tokens and multi-component salt '.' parsing."""
    ring10 = parse_smiles("C%10CCCCC%10")
    cyc6 = parse_smiles("C1CCCCC1")
    assert ring10.canonical_smiles == cyc6.canonical_smiles
    assert len(ring10.sssr_rings) == 1

    salt = parse_smiles("CC(=O)[O-].C[NH3+]")
    assert salt.num_atoms == 6
    assert sum(a.formal_charge for a in salt.atoms) == 0


def test_hybridization_sp_sp2_sp3_and_smarts_boolean_ops() -> None:
    """Verify SP/SP2/SP3 hybridization assignment and SMARTS OR/NOT boolean logic."""
    from chiral_weave.chem_types import Hybridization

    erlotinib = parse_smiles("COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC")
    hybs = {a.hybridization for a in erlotinib.atoms}
    assert Hybridization.SP in hybs
    assert Hybridization.SP2 in hybs
    assert Hybridization.SP3 in hybs

    hetero_ring = find_smarts_matches(erlotinib, "[!#6;R]")
    assert len(hetero_ring) == 2  # Two quinazoline ring nitrogens
    oxygen_or_nitrogen = find_smarts_matches(erlotinib, "[#7,#8]")
    assert len(oxygen_or_nitrogen) == 7
