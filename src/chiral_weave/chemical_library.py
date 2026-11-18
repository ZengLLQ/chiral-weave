"""Curated medicinal-chemistry benchmark compounds, congeneric SAR series, and library container."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import hashlib
import json
import numpy as np

from chiral_weave.chem_types import MoleculeGraph
from chiral_weave.descriptors import PhysicochemicalProfile, compute_physicochemical_profile
from chiral_weave.fingerprints import compute_morgan_fingerprint
from chiral_weave.smiles_parser import parse_smiles


@dataclass(frozen=True)
class LibraryEntry:
    """Single molecule record inside a CompoundLibrary."""

    compound_id: str
    name: str
    smiles: str
    series: str
    measured_pIC50: float


# 24 canonical medicinal-chemistry drugs & kinase/PROTAC reference scaffolds
CURATED_DRUG_PANEL: Tuple[Tuple[str, str, str, str, float], ...] = (
    ("CW-001", "Aspirin", "CC(=O)Oc1ccccc1C(=O)O", "NSAID", 4.60),
    ("CW-002", "Ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "NSAID", 5.10),
    ("CW-003", "Acetaminophen", "CC(=O)Nc1ccc(O)cc1", "Analgesic", 4.30),
    ("CW-004", "Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "Xanthine", 4.85),
    ("CW-005", "Gefitinib", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", "Quinazoline_Kinase", 8.65),
    ("CW-006", "Erlotinib", "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC", "Quinazoline_Kinase", 8.40),
    ("CW-007", "Lapatinib_Core",
     "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
     "Quinazoline_Kinase", 8.80),
    ("CW-008", "Imatinib",
     "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
     "Aminopyrimidine_Kinase", 8.55),
    ("CW-009", "Sildenafil", "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12", "PDE5_Inhibitor", 8.35),
    ("CW-010", "Warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "Coumarin", 6.20),
    ("CW-011", "Thalidomide", "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1", "CRBN_Binder", 6.75),
    ("CW-012", "Lenalidomide", "Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O", "CRBN_Binder", 7.15),
    ("CW-013", "Pomalidomide", "Nc1cccc2c1C(=O)N(C1CCC(=O)NC1=O)C2=O", "CRBN_Binder", 7.45),
    ("CW-014", "Celecoxib", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1", "Sulfonamide", 7.30),
    ("CW-015", "Fluoxetine", "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1", "SSRI", 6.90),
    ("CW-016", "Atorvastatin_Frag",
     "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
     "Statin", 8.10),
    ("CW-017", "Tamoxifen", "CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1", "SERM", 7.40),
    ("CW-018", "Quinazoline_Hit1", "c1ccc(Nc2ncnc3ccccc23)cc1", "Quinazoline_SAR", 6.50),
    ("CW-019", "Quinazoline_F", "Fc1cccc(Nc2ncnc3ccccc23)c1", "Quinazoline_SAR", 7.10),
    ("CW-020", "Quinazoline_Cl", "Clc1cccc(Nc2ncnc3ccccc23)c1", "Quinazoline_SAR", 7.35),
    ("CW-021", "Quinazoline_CF3", "FC(F)(F)c1cccc(Nc2ncnc3ccccc23)c1", "Quinazoline_SAR", 7.55),
    ("CW-022", "Quinazoline_OH", "Oc1cccc(Nc2ncnc3ccccc23)c1", "Quinazoline_SAR", 6.80),
    ("CW-023", "Quinazoline_OMe", "COc1cc2ncnc(Nc3ccc(Cl)cc3)c2cc1OC", "Quinazoline_SAR", 8.05),
    ("CW-024", "Quinazoline_CN", "N#Cc1cccc(Nc2ncnc3ccccc23)c1", "Quinazoline_SAR", 7.05),
)


class CompoundLibrary:
    """Indexed chemical screening library with SHA-256 provenance and fingerprint cache."""

    def __init__(self, entries: Sequence[LibraryEntry]) -> None:
        self.entries: List[LibraryEntry] = list(entries)
        self.molecules: List[MoleculeGraph] = []
        self.profiles: List[PhysicochemicalProfile] = []
        self.fingerprints: List[np.ndarray] = []

        for entry in self.entries:
            mol = parse_smiles(entry.smiles, name=entry.name)
            self.molecules.append(mol)
            self.profiles.append(compute_physicochemical_profile(mol))
            self.fingerprints.append(compute_morgan_fingerprint(mol, radius=2, nbits=1024))

    @classmethod
    def default_benchmark_library(cls) -> CompoundLibrary:
        """Load the built-in 24-compound medicinal chemistry and kinase SAR library."""
        entries = [
            LibraryEntry(cid, name, smi, series, pic50)
            for cid, name, smi, series, pic50 in CURATED_DRUG_PANEL
        ]
        return cls(entries)

    @classmethod
    def synthetic_sar_cohort(cls, seed: int = 42) -> CompoundLibrary:
        """Generate a 36-compound multi-scaffold SAR dataset with deterministic pIC50 labels."""
        rng = np.random.default_rng(seed)
        entries: List[LibraryEntry] = [
            LibraryEntry(cid, name, smi, series, pic50)
            for cid, name, smi, series, pic50 in CURATED_DRUG_PANEL
        ]

        # Add 12 aryl-sulfonamide and benzimidazole analogs for scaffold-split CV
        r_groups = ["", "F", "Cl", "C", "OC", "C(F)(F)F"]
        templates = [
            ("Sulfonamide_SAR", "NS(=O)(=O)c1ccc(Nc2ccc({R})cc2)cc1", 6.2),
            ("Benzimidazole_SAR", "c1ccc2nc(N{R_alt})[nH]c2c1", 6.0),
        ]
        idx = 25
        for series_name, tmpl, base_val in templates:
            for r in r_groups:
                if "{R}" in tmpl:
                    smi = tmpl.replace("({R})", "" if not r else f"({r})")
                else:
                    alt = "C" if not r else f"c1ccc({r})cc1"
                    smi = tmpl.replace("{R_alt}", alt)
                mol = parse_smiles(smi)
                prof = compute_physicochemical_profile(mol)
                # Deterministic structure-activity relationship: lipophilic + HBA boost minus excess TPSA
                pic50 = float(
                    base_val
                    + 0.42 * prof.clogp
                    + 0.18 * prof.hba
                    - 0.012 * prof.tpsa
                    + rng.normal(0.0, 0.08)
                )
                entries.append(
                    LibraryEntry(
                        compound_id=f"CW-{idx:03d}",
                        name=f"{series_name}_{idx}",
                        smiles=mol.canonical_smiles,
                        series=series_name,
                        measured_pIC50=pic50,
                    )
                )
                idx += 1
        return cls(entries)

    def provenance_sha256(self) -> str:
        """Compute deterministic SHA-256 checksum over canonical SMILES and measured labels."""
        h = hashlib.sha256()
        for entry, mol in zip(self.entries, self.molecules):
            line = f"{entry.compound_id}|{mol.canonical_smiles}|{entry.measured_pIC50:.6f}\n"
            h.update(line.encode("utf-8"))
        return h.hexdigest()

    def save_json(self, path: Path) -> None:
        """Serialize library metadata, canonical SMILES, and provenance checksum to JSON."""
        payload = {
            "provenance_sha256": self.provenance_sha256(),
            "count": len(self.entries),
            "compounds": [
                {
                    "compound_id": e.compound_id,
                    "name": e.name,
                    "smiles": m.canonical_smiles,
                    "series": e.series,
                    "measured_pIC50": e.measured_pIC50,
                    "mw": p.molecular_weight,
                    "clogp": p.clogp,
                    "tpsa": p.tpsa,
                    "qed": p.qed,
                    "scaffold": p.bemis_murcko_scaffold,
                }
                for e, m, p in zip(self.entries, self.molecules, self.profiles)
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def labels_array(self) -> np.ndarray:
        """Return measured pIC50 targets as a 1D float64 array."""
        return np.asarray([e.measured_pIC50 for e in self.entries], dtype=np.float64)

    def series_map(self) -> Dict[str, List[int]]:
        """Group compound indices by chemical series."""
        groups: Dict[str, List[int]] = {}
        for idx, e in enumerate(self.entries):
            groups.setdefault(e.series, []).append(idx)
        return groups
