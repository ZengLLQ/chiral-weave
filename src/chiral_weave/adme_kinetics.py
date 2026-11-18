"""Multitask ADME/PK predictors, CYP450 MBI ODE kinetics, hepatic extraction, and SoM ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import math
import numpy as np
from scipy.integrate import solve_ivp

from chiral_weave.chem_types import AdmeProfile, BondOrder, Hybridization, MoleculeGraph
from chiral_weave.descriptors import (
    PhysicochemicalProfile,
    compute_physicochemical_profile,
)
from chiral_weave.graph_neural import MolecularMPNN, featurize_molecule


@dataclass(frozen=True)
class CypSimulationResult:
    """Output of coupled CYP450 substrate-inhibitor mechanism-based inactivation ODEs."""

    isoform: str
    time_min: np.ndarray
    substrate_uM: np.ndarray
    active_enzyme_fraction: np.ndarray
    auc_inhibited: float
    auc_control: float
    ddi_auc_ratio: float
    residual_active_enzyme: float


@dataclass(frozen=True)
class MetabolismHotspot:
    """Predicted CYP450 Site-of-Metabolism (SoM) vulnerability at a heavy atom."""

    atom_index: int
    symbol: str
    reaction_type: str
    lability_score: float


def compute_esol_log_s(profile: PhysicochemicalProfile) -> float:
    """Compute Delaney ESOL aqueous solubility logS (log10 mol/L)."""
    rb = float(profile.rotatable_bonds)
    ap = (
        0.0
        if profile.heavy_atom_count == 0
        else float(profile.aromatic_ring_count * 6) / max(1.0, float(profile.heavy_atom_count))
    )
    ap = min(1.0, ap)
    return float(
        0.16
        - 0.63 * profile.clogp
        - 0.0062 * profile.molecular_weight
        + 0.066 * rb
        - 0.74 * ap
    )


def compute_hepatic_extraction(
    cl_int_uL_min_mg: float,
    fu_plasma: float,
    qh_mL_min_kg: float = 20.7,
    mppgl_mg_g: float = 40.0,
    liver_g_kg: float = 25.7,
) -> float:
    """Compute human well-stirred hepatic extraction ratio E_H in [0, 1).

    Scales in vitro hepatocyte/microsomal CL_int (uL/min/mg) to whole-liver
    clearance (mL/min/kg) using standard IVIVE physiological scalars.
    """
    cl_scaled_mL_min_kg = max(0.0, cl_int_uL_min_mg) * 1e-3 * mppgl_mg_g * liver_g_kg
    fu = max(1e-4, min(1.0, fu_plasma))
    intrinsic = fu * cl_scaled_mL_min_kg
    return float(intrinsic / (qh_mL_min_kg + intrinsic))


def simulate_cyp_mbi_kinetics(
    isoform: str,
    inhibitor_uM: float,
    ic50_uM: float,
    k_inact_per_min: float = 0.035,
    k_deg_per_min: float = 0.00032,
    substrate_init_uM: float = 5.0,
    km_uM: float = 12.0,
    vmax_uM_min: float = 0.45,
    duration_min: float = 120.0,
) -> CypSimulationResult:
    """Integrate coupled CYP450 substrate clearance and mechanism-based inactivation ODEs.

    Solves:
      dE/dt = k_deg * (1 - E) - (k_inact * I / (K_I + I)) * E
      dS/dt = -V_max * E * S / (K_m * (1 + I / K_i) + S)
    in comparison with an uninhibited control (I = 0) to determine DDI AUC ratio.
    """
    ki_uM = max(0.05, ic50_uM * 0.5)
    k_i_mbi = max(0.1, ic50_uM * 1.4)
    t_eval = np.linspace(0.0, duration_min, 121)

    def _rhs(_t: float, y: np.ndarray, inh: float) -> List[float]:
        enz = max(0.0, min(1.5, float(y[0])))
        sub = max(0.0, float(y[1]))
        inact_rate = (k_inact_per_min * inh / (k_i_mbi + inh)) if inh > 0.0 else 0.0
        d_enz = k_deg_per_min * (1.0 - enz) - inact_rate * enz
        km_app = km_uM * (1.0 + inh / ki_uM)
        d_sub = -vmax_uM_min * enz * sub / (km_app + sub + 1e-12)
        return [d_enz, d_sub]

    sol_inh = solve_ivp(
        lambda t, y: _rhs(t, y, max(0.0, inhibitor_uM)),
        (0.0, duration_min),
        [1.0, substrate_init_uM],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )
    sol_ctrl = solve_ivp(
        lambda t, y: _rhs(t, y, 0.0),
        (0.0, duration_min),
        [1.0, substrate_init_uM],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )

    sub_inh = np.maximum(0.0, sol_inh.y[1])
    sub_ctrl = np.maximum(0.0, sol_ctrl.y[1])
    enz_inh = np.clip(sol_inh.y[0], 0.0, 1.0)

    auc_inh = float(np.trapezoid(sub_inh, t_eval))
    auc_ctrl = float(np.trapezoid(sub_ctrl, t_eval))
    ratio = auc_inh / max(auc_ctrl, 1e-12)

    return CypSimulationResult(
        isoform=isoform,
        time_min=t_eval,
        substrate_uM=sub_inh,
        active_enzyme_fraction=enz_inh,
        auc_inhibited=auc_inh,
        auc_control=auc_ctrl,
        ddi_auc_ratio=float(ratio),
        residual_active_enzyme=float(enz_inh[-1]),
    )


def rank_metabolism_sites(mol: MoleculeGraph) -> List[MetabolismHotspot]:
    """Rank heavy atoms as soft spots for CYP450 oxidative metabolism."""
    sites: List[MetabolismHotspot] = []
    for atom in mol.atoms:
        if atom.symbol not in ("C", "N", "S"):
            continue
        inc = mol.incident_bonds(atom.index)
        nbrs = [mol.atoms[b.other(atom.index)] for b in inc]

        score = 0.0
        rxn = ""
        if atom.symbol == "C" and atom.hybridization == Hybridization.SP3 and atom.total_h > 0:
            # Benzylic or allylic C-H stabilization
            if any(nbr.is_aromatic for nbr in nbrs):
                score = 0.88
                rxn = "benzylic_hydroxylation"
            elif any(nbr.symbol in ("N", "O") for nbr in nbrs):
                score = 0.82 if atom.total_h == 3 else 0.74
                rxn = "N_or_O_dealkylation"
            elif atom.total_h == 1:
                score = 0.65
                rxn = "tertiary_aliphatic_hydroxylation"
            else:
                score = 0.42
                rxn = "aliphatic_hydroxylation"
        elif atom.symbol == "C" and atom.is_aromatic and atom.total_h > 0:
            has_ewg = any(nbr.symbol in ("F", "Cl", "N") for nbr in nbrs)
            score = 0.28 if has_ewg else 0.54
            rxn = "aromatic_hydroxylation"
        elif atom.symbol == "N" and not atom.is_aromatic and atom.formal_charge == 0:
            score = 0.68
            rxn = "N_oxidation"
        elif atom.symbol == "S" and all(b.order == BondOrder.SINGLE for b in inc):
            score = 0.79
            rxn = "S_oxidation"

        if score > 0.0:
            sites.append(
                MetabolismHotspot(
                    atom_index=atom.index,
                    symbol=atom.symbol,
                    reaction_type=rxn,
                    lability_score=float(score),
                )
            )

    sites.sort(key=lambda s: -s.lability_score)
    return sites


class AdmePredictor:
    """Hybrid descriptor + MolecularMPNN multitask ADME and CYP450 DDI predictor."""

    def __init__(self, hidden_dim: int = 32, seed: int = 17) -> None:
        self.gnn = MolecularMPNN(hidden_dim=hidden_dim, num_layers=3, out_dim=5, dropout_rate=0.12, seed=seed)

    def predict(
        self,
        mol: MoleculeGraph,
        inhibitor_conc_uM: float = 1.0,
        mc_samples: int = 10,
        seed: int = 42,
    ) -> AdmeProfile:
        """Predict full ADME/PK endpoints and CYP3A4 DDI AUC ratio with epistemic uncertainty."""
        prof = compute_physicochemical_profile(mol)
        tensor = featurize_molecule(mol)
        gnn_mean, gnn_std = self.gnn.predict_with_uncertainty(tensor, mc_samples=mc_samples, seed=seed)

        # Combine mechanistic biophysics priors with bounded GNN structural residual
        res = np.tanh(gnn_mean) * np.array([0.45, 0.25, 0.20, 0.35, 0.40], dtype=np.float64)

        log_s = compute_esol_log_s(prof) + float(res[0])
        # Caco-2 permeability: log Papp (10^-6 cm/s) drops sharply at high TPSA or extreme MW
        log_papp = (
            1.45
            - 0.014 * prof.tpsa
            + 0.18 * prof.clogp
            - 0.0012 * max(0.0, prof.molecular_weight - 350.0)
            + float(res[1])
        )
        # Clark BBB model: logBB = 0.152*clogP - 0.0148*TPSA + 0.139
        log_bb = 0.152 * prof.clogp - 0.0148 * prof.tpsa + 0.139 + float(res[2])

        # Plasma fraction unbound (fu): decreases with lipophilicity
        logit_fu = 0.8 - 0.85 * prof.clogp - 0.002 * prof.molecular_weight + float(res[3])
        fu_plasma = float(max(0.002, min(0.98, 1.0 / (1.0 + math.exp(-logit_fu)))))

        # Intrinsic hepatic clearance (uL/min/mg) modulated by lipophilicity & metabolic soft spots
        som_list = rank_metabolism_sites(mol)
        som_burden = sum(s.lability_score for s in som_list[:4])
        cl_int = float(
            max(
                1.5,
                8.0 * math.exp(0.28 * max(-1.0, prof.clogp)) * (1.0 + 0.35 * som_burden)
                + 12.0 * float(res[4]),
            )
        )
        eh = compute_hepatic_extraction(cl_int, fu_plasma)

        # P-gp efflux risk rises for high MW + high TPSA + HBD >= 2
        pgp_logit = -2.2 + 0.005 * prof.molecular_weight + 0.018 * prof.tpsa + 0.35 * prof.hbd - 0.25 * prof.clogp
        pgp_prob = float(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, pgp_logit)))))

        # CYP isoform IC50 approximations (uM): lipophilic/basic amines drive CYP2D6; bulky lipophiles CYP3A4
        base_3a4 = 25.0 * math.exp(-0.35 * max(0.0, prof.clogp - 1.5)) * (500.0 / max(250.0, prof.molecular_weight))
        cyp3a4_ic50 = float(max(0.15, min(100.0, base_3a4 * math.exp(0.3 * float(res[0])))))
        has_basic_n = any(
            a.symbol == "N" and not a.is_aromatic and a.hybridization == Hybridization.SP3
            for a in mol.atoms
        )
        cyp2d6_ic50 = float(
            max(0.20, min(100.0, (6.5 if has_basic_n else 38.0) * math.exp(-0.22 * max(0.0, prof.clogp - 1.5))))
        )
        cyp2c9_ic50 = float(
            max(0.25, min(100.0, 28.0 * math.exp(-0.28 * max(0.0, prof.clogp - 1.8))))
        )

        cyp_sim = simulate_cyp_mbi_kinetics(
            isoform="CYP3A4",
            inhibitor_uM=inhibitor_conc_uM,
            ic50_uM=cyp3a4_ic50,
            duration_min=90.0,
        )

        return AdmeProfile(
            smiles=prof.smiles,
            log_s_esol=log_s,
            log_papp_caco2=log_papp,
            log_bb=log_bb,
            fu_plasma=fu_plasma,
            cl_int_uL_min_mg=cl_int,
            hepatic_extraction=eh,
            pgp_efflux_prob=pgp_prob,
            cyp3a4_ic50_uM=cyp3a4_ic50,
            cyp2d6_ic50_uM=cyp2d6_ic50,
            cyp2c9_ic50_uM=cyp2c9_ic50,
            cyp3a4_ddi_auc_ratio=cyp_sim.ddi_auc_ratio,
            epistemic_std=float(np.mean(gnn_std)),
        )
