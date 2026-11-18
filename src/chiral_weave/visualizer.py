"""Pure 7-bit ASCII terminal visualizers for molecular graphs, docking, PROTAC curves, and Pareto fronts."""

from __future__ import annotations

from typing import Sequence
import math
import numpy as np

from chiral_weave.chem_types import AdmeProfile, DockingPose, MoleculeGraph, OptimizationCandidate, ProtacOutcome
from chiral_weave.descriptors import PhysicochemicalProfile


def assert_ascii(text: str) -> str:
    """Verify strictly that every character in text is 7-bit ASCII (codes 0-127)."""
    for idx, ch in enumerate(text):
        if ord(ch) >= 128:
            raise ValueError(f"Non-ASCII character {ch!r} (ord={ord(ch)}) at index {idx}.")
    return text


def render_molecule_ascii_summary(
    mol: MoleculeGraph,
    profile: PhysicochemicalProfile,
    adme: AdmeProfile | None = None,
) -> str:
    """Render an ASCII summary card with topological ring inventory and Gasteiger partial charges."""
    lines = [
        "+" + "=" * 74 + "+",
        f"| ChiralWeave Molecule Profile : {mol.name or 'Unnamed':<41} |",
        "+" + "-" * 74 + "+",
        f"| Canonical SMILES : {profile.smiles[:53]:<53} |",
        f"| Formula / MW     : {profile.formula:<16}  MW={profile.molecular_weight:>7.2f} Da   "
        f"Heavy={profile.heavy_atom_count:>3}       |",
        f"| SSSR Rings / Arom: {profile.ring_count:<2} rings ({profile.aromatic_ring_count} aromatic)   "
        f"RotB={profile.rotatable_bonds:<2}   Fsp3={profile.fraction_csp3:>5.2f}   |",
        f"| Polar / Lipo     : TPSA={profile.tpsa:>6.1f} A^2   clogP={profile.clogp:>6.2f}   "
        f"HBD={profile.hbd:<2} HBA={profile.hba:<2}    |",
        f"| Drug-likeness    : QED={profile.qed:>5.3f}   SA={profile.sa_score:>4.2f}   "
        f"Ro5_Viol={profile.lipinski_violations}   Veber={str(profile.veber_pass):<5}  |",
        f"| Murcko Scaffold  : {(profile.bemis_murcko_scaffold or '<acyclic>')[:53]:<53} |",
        f"| PAINS/Alerts     : {(', '.join(profile.alerts) if profile.alerts else 'None')[:53]:<53} |",
    ]
    if adme is not None:
        lines.extend(
            [
                "+" + "-" * 74 + "+",
                f"| ADME/PK (GNN+ESOL): logS={adme.log_s_esol:>6.2f}  logPapp={adme.log_papp_caco2:>5.2f}  "
                f"logBB={adme.log_bb:>6.2f}  fu={adme.fu_plasma:>5.3f} |",
                f"| Clearance / DDI  : CLint={adme.cl_int_uL_min_mg:>6.1f} uL/min/mg  "
                f"EH={adme.hepatic_extraction:>5.3f}  CYP3A4_AUCr={adme.cyp3a4_ddi_auc_ratio:>5.2f} |",
            ]
        )

    lines.append("+" + "-" * 74 + "+")
    lines.append("| Gasteiger Partial Charge Distribution (first 8 heavy atoms):             |")
    for atom in mol.atoms[:8]:
        q = atom.gasteiger_charge
        bar_len = min(14, max(0, int(round(abs(q) * 32.0))))
        sign = "+" if q >= 0 else "-"
        bar = (sign * bar_len).ljust(14, ".")
        arom_tag = "arom" if atom.is_aromatic else "alip"
        lines.append(
            f"|   [{atom.index:>2}] {atom.symbol:<2} ({atom.hybridization.value:<3},{arom_tag})  "
            f"q={q:>+7.4f}  |{bar}|          |"
        )
    lines.append("+" + "=" * 74 + "+")
    return assert_ascii("\n".join(lines))


def render_docking_ascii_breakdown(pose: DockingPose, pocket_name: str = "Kinase_ATP_Hinge") -> str:
    """Render an ASCII bar breakdown of Vina empirical docking affinity terms."""
    terms = [
        ("Gauss1 Steric", pose.gauss1_score),
        ("Gauss2 Steric", pose.gauss2_score),
        ("Repulsion", pose.repulsion_score),
        ("Hydrophobic", pose.hydrophobic_score),
        ("H-Bonding", pose.hbond_score),
        ("Pi-Stacking", pose.pi_stacking_score),
        ("Torsion Entr.", pose.torsion_penalty),
    ]
    max_abs = max([abs(v) for _, v in terms] + [0.5])
    lines = [
        "+" + "=" * 70 + "+",
        f"| Empirical Docking Pose ({pocket_name:<20}) dG={pose.total_affinity_kcal:>7.3f} kcal/mol  |",
        "+" + "-" * 70 + "+",
    ]
    for label, val in terms:
        width = min(28, int(round((abs(val) / max_abs) * 28.0)))
        ch = "#" if val <= 0 else "!"
        bar = (ch * width).ljust(28, ".")
        lines.append(f"| {label:<14} {val:>+8.4f} kcal/mol  [{bar}]     |")

    contact_str = ", ".join(f"{idx}:{res}" for idx, res in pose.hbond_contacts[:4]) or "None"
    lines.append("+" + "-" * 70 + "+")
    lines.append(f"| H-Bond Contacts: {contact_str[:51]:<51} |")
    lines.append("+" + "=" * 70 + "+")
    return assert_ascii("\n".join(lines))


def render_protac_ascii_curves(outcome: ProtacOutcome, width: int = 58, height: int = 12) -> str:
    """Render 7-bit ASCII PROTAC ternary hook-effect curve and 24h degradation trajectory."""
    lines = [
        "+" + "=" * 74 + "+",
        f"| PROTAC Ternary Hook Effect (alpha={outcome.cooperativity_alpha:>5.2f}, "
        f"Kd_P={outcome.kd_target_nM:>5.1f} nM, Kd_E={outcome.kd_e3_nM:>5.1f} nM) |",
        f"| Hook Peak={outcome.hook_peak_nM:>8.2f} nM | TF50={outcome.tf50_nM:>7.2f} nM | "
        f"DC50={outcome.dc50_nM:>7.2f} nM | Dmax={outcome.dmax_fraction*100:>5.1f}%  |",
        "+" + "-" * 74 + "+",
    ]

    # Plot hook curve on log10 dose axis
    grid = [[" " for _ in range(width)] for _ in range(height)]
    log_doses = np.log10(outcome.dose_grid_nM)
    d_min, d_max = float(log_doses[0]), float(log_doses[-1])
    y_max = max(1e-9, float(np.max(outcome.ternary_curve_uM)))

    for ld, val in zip(log_doses, outcome.ternary_curve_uM):
        col = min(width - 1, max(0, int(round((ld - d_min) / max(1e-9, d_max - d_min) * (width - 1)))))
        row = height - 1 - min(height - 1, max(0, int(round((val / y_max) * (height - 1)))))
        grid[row][col] = "*"

    peak_col = min(
        width - 1,
        max(0, int(round((math.log10(outcome.hook_peak_nM) - d_min) / max(1e-9, d_max - d_min) * (width - 1)))),
    )
    for r in range(height):
        if grid[r][peak_col] == " ":
            grid[r][peak_col] = ":"

    for r in range(height):
        frac = 1.0 - r / max(1, height - 1)
        y_label = f"{frac * y_max * 1000.0:6.1f}nM"
        lines.append(f"| {y_label} |{''.join(grid[r])}|  |")
    lines.append("|          +" + "-" * width + "+  |")
    lines.append(f"|          10^{d_min:+.0f} nM" + " " * (width - 18) + f"10^{d_max:+.0f} nM     |")

    # Degradation kinetic mini-bars
    lines.append("+" + "-" * 74 + "+")
    lines.append("| Target Protein Degradation Time-Course (Fraction Remaining vs Time):     |")
    sample_idx = np.linspace(0, len(outcome.time_grid_h) - 1, 6, dtype=int)
    for idx in sample_idx:
        t_h = float(outcome.time_grid_h[int(idx)])
        rem = float(outcome.target_protein_fraction[int(idx)])
        bar = ("#" * int(round(rem * 36.0))).ljust(36, ".")
        lines.append(f"|   t={t_h:>5.1f} h  [{bar}]  {rem * 100:>5.1f}% remaining    |")
    lines.append("+" + "=" * 74 + "+")
    return assert_ascii("\n".join(lines))


def render_similarity_heatmap_ascii(sim_matrix: np.ndarray, names: Sequence[str], max_items: int = 10) -> str:
    """Render an ASCII Tanimoto similarity matrix heatmap using ramp characters."""
    ramp = " .:-=+*#%@"
    k = min(max_items, len(names), sim_matrix.shape[0])
    lines = [
        "+" + "=" * 66 + "+",
        "| Pairwise Morgan ECFP4 Tanimoto Similarity Matrix (ASCII ramp)    |",
        "+" + "-" * 66 + "+",
    ]
    header = "  " + " ".join(f"{i:>2}" for i in range(k))
    lines.append(f"| {header:<64} |")
    for i in range(k):
        cells = []
        for j in range(k):
            val = float(max(0.0, min(1.0, sim_matrix[i, j])))
            ch = ramp[min(len(ramp) - 1, int(round(val * (len(ramp) - 1))))]
            cells.append(f" {ch}")
        row_str = f"{i:>2} " + " ".join(cells) + f"  ({names[i][:20]})"
        lines.append(f"| {row_str:<64} |")
    lines.append("+" + "=" * 66 + "+")
    return assert_ascii("\n".join(lines))


def render_pareto_ascii_plot(
    candidates: Sequence[OptimizationCandidate],
    pareto: Sequence[OptimizationCandidate],
    width: int = 52,
    height: int = 12,
) -> str:
    """Render a 2D ASCII scatter plot of Docking Affinity (-dG) vs QED showing Pareto front."""
    if not candidates:
        return assert_ascii("No candidates evaluated.")
    pareto_set = {c.canonical_smiles for c in pareto}
    x_vals = [-c.docking_kcal for c in candidates]
    y_vals = [c.qed for c in candidates]
    x_min, x_max = min(x_vals), max(x_vals) + 1e-6
    y_min, y_max = max(0.0, min(y_vals) - 0.05), min(1.0, max(y_vals) + 0.05)

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for c in candidates:
        xv = -c.docking_kcal
        yv = c.qed
        col = min(width - 1, max(0, int(round((xv - x_min) / max(1e-9, x_max - x_min) * (width - 1)))))
        row = height - 1 - min(height - 1, max(0, int(round((yv - y_min) / max(1e-9, y_max - y_min) * (height - 1)))))
        sym = "*" if c.canonical_smiles in pareto_set else "+"
        if grid[row][col] != "*":
            grid[row][col] = sym

    lines = [
        "+" + "=" * 68 + "+",
        "| MCTS Lead Optimization Pareto Plot (* = Pareto Front, + = Eval)  |",
        "+" + "-" * 68 + "+",
    ]
    for r in range(height):
        q_tick = y_max - (r / max(1, height - 1)) * (y_max - y_min)
        lines.append(f"| QED {q_tick:>4.2f} |{''.join(grid[r])}|   |")
    lines.append("|          +" + "-" * width + "+   |")
    lines.append(f"|       -dG={x_min:>4.2f} kcal/mol" + " " * (width - 26) + f"{x_max:>4.2f} kcal/mol  |")
    lines.append("+" + "=" * 68 + "+")
    return assert_ascii("\n".join(lines))
