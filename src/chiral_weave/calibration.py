"""Scaffold-split cross-validation, GNN QSAR calibration metrics, and sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import math
import numpy as np
from scipy.stats import pearsonr, spearmanr

from chiral_weave.chemical_library import CompoundLibrary
from chiral_weave.conformer_docking import generate_3d_conformer, signed_chiral_volume
from chiral_weave.descriptors import compute_gasteiger_charges
from chiral_weave.graph_neural import MolecularMPNN, featurize_molecule
from chiral_weave.protac_cooperativity import simulate_protac_campaign
from chiral_weave.smiles_parser import parse_smiles


@dataclass(frozen=True)
class ScaffoldSplitMetrics:
    """Out-of-fold validation metrics on Bemis-Murcko scaffold splits."""

    num_train: int
    num_test: int
    train_initial_loss: float
    train_final_loss: float
    test_rmse: float
    test_mae: float
    pearson_r: float
    spearman_rho: float
    enrichment_factor_20pct: float
    mean_epistemic_std: float


@dataclass(frozen=True)
class ChiralityVerification:
    """Result of 3D Distance Geometry chiral inversion test (@ vs @@)."""

    smiles_ccw: str
    smiles_cw: str
    volume_ccw: float
    volume_cw: float
    opposite_handedness_preserved: bool


@dataclass(frozen=True)
class SensitivityReport:
    """Local finite-difference sensitivity of PROTAC hook-peak and Dmax to K_d and alpha."""

    baseline_hook_peak_nM: float
    baseline_dmax: float
    sensitivities: Dict[str, Tuple[float, float]]  # param -> (d_log_peak / d_log_param, d_dmax / d_log_param)


def scaffold_train_test_split(
    library: CompoundLibrary,
    test_fraction: float = 0.25,
) -> Tuple[List[int], List[int]]:
    """Partition compounds by Bemis-Murcko scaffold so identical cores never leak across splits."""
    groups: Dict[str, List[int]] = {}
    for idx, prof in enumerate(library.profiles):
        key = prof.bemis_murcko_scaffold or f"ACYCLIC_{idx}"
        groups.setdefault(key, []).append(idx)

    # Sort scaffold buckets largest-first and assign smaller/diverse buckets to test set
    buckets = sorted(groups.values(), key=lambda ids: (len(ids), ids[0]))
    n_total = len(library.entries)
    target_test = max(2, int(round(n_total * test_fraction)))

    test_idx: List[int] = []
    train_idx: List[int] = []
    for bucket in buckets:
        if len(test_idx) < target_test:
            test_idx.extend(bucket)
        else:
            train_idx.extend(bucket)

    if not train_idx and len(test_idx) > 2:
        train_idx = test_idx[:-2]
        test_idx = test_idx[-2:]
    return sorted(train_idx), sorted(test_idx)


def compute_enrichment_factor(y_true: np.ndarray, y_pred: np.ndarray, top_fraction: float = 0.20) -> float:
    """Compute Enrichment Factor EF_x for recovering top-quartile actives by predicted rank."""
    n = len(y_true)
    if n < 4:
        return 1.0
    cutoff = float(np.percentile(y_true, 75.0))
    actives = y_true >= cutoff
    total_actives = int(actives.sum())
    if total_actives == 0:
        return 1.0

    k = max(1, int(math.ceil(n * top_fraction)))
    top_k_idx = np.argsort(y_pred)[::-1][:k]
    hits_in_top_k = int(actives[top_k_idx].sum())
    expected = (k / n) * total_actives
    return float(hits_in_top_k / max(1e-9, expected))


def run_scaffold_gnn_benchmark(
    epochs: int = 20,
    hidden_dim: int = 24,
    seed: int = 42,
) -> ScaffoldSplitMetrics:
    """Train and evaluate MolecularMPNN on a scaffold-split synthetic + curated SAR dataset."""
    lib = CompoundLibrary.synthetic_sar_cohort(seed=seed)
    train_idx, test_idx = scaffold_train_test_split(lib, test_fraction=0.25)

    tensors = [featurize_molecule(m) for m in lib.molecules]
    y_all = lib.labels_array()

    y_mean = float(y_all[train_idx].mean())
    y_std = max(1e-6, float(y_all[train_idx].std()))
    y_norm = (y_all - y_mean) / y_std

    train_tensors = [tensors[i] for i in train_idx]
    train_y = y_norm[train_idx]

    model = MolecularMPNN(hidden_dim=hidden_dim, num_layers=2, out_dim=1, dropout_rate=0.08, seed=seed)
    report = model.fit(train_tensors, train_y, epochs=epochs, lr=6e-3, batch_size=6, seed=seed)

    preds_norm: List[float] = []
    stds: List[float] = []
    for i in test_idx:
        mean_arr, std_arr = model.predict_with_uncertainty(tensors[i], mc_samples=8, seed=seed + i)
        preds_norm.append(float(mean_arr[0]))
        stds.append(float(std_arr[0] * y_std))

    y_test = y_all[test_idx]
    y_pred = np.asarray(preds_norm, dtype=np.float64) * y_std + y_mean
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_test)))

    if len(y_test) >= 3 and np.std(y_pred) > 1e-9 and np.std(y_test) > 1e-9:
        r_val = float(pearsonr(y_test, y_pred).statistic)
        rho_val = float(spearmanr(y_test, y_pred).statistic)
    else:
        r_val, rho_val = 0.0, 0.0

    ef20 = compute_enrichment_factor(y_test, y_pred, top_fraction=0.20)

    return ScaffoldSplitMetrics(
        num_train=len(train_idx),
        num_test=len(test_idx),
        train_initial_loss=report.initial_loss,
        train_final_loss=report.final_loss,
        test_rmse=rmse,
        test_mae=mae,
        pearson_r=r_val,
        spearman_rho=rho_val,
        enrichment_factor_20pct=ef20,
        mean_epistemic_std=float(np.mean(stds)),
    )


def verify_chiral_3d_inversion(seed: int = 42) -> ChiralityVerification:
    """Verify that 3D Distance Geometry preserves opposite signed tetrahedral volumes for @ vs @@."""
    smi_ccw = "C[C@H](N)C(=O)O"
    smi_cw = "C[C@@H](N)C(=O)O"
    mol_ccw = parse_smiles(smi_ccw)
    mol_cw = parse_smiles(smi_cw)
    coords_ccw = generate_3d_conformer(mol_ccw, seed=seed)
    coords_cw = generate_3d_conformer(mol_cw, seed=seed)

    # Atom 1 is the chiral alpha-carbon
    nbrs_ccw = sorted(mol_ccw.neighbors(1))[:3]
    nbrs_cw = sorted(mol_cw.neighbors(1))[:3]
    vol_ccw = signed_chiral_volume(coords_ccw, 1, nbrs_ccw)
    vol_cw = signed_chiral_volume(coords_cw, 1, nbrs_cw)

    preserved = (vol_ccw < -0.1) and (vol_cw > 0.1)
    return ChiralityVerification(
        smiles_ccw=smi_ccw,
        smiles_cw=smi_cw,
        volume_ccw=vol_ccw,
        volume_cw=vol_cw,
        opposite_handedness_preserved=preserved,
    )


def verify_charge_conservation(smiles_list: Sequence[str]) -> float:
    """Return maximum absolute Gasteiger charge conservation error across molecules."""
    max_err = 0.0
    for smi in smiles_list:
        mol = parse_smiles(smi)
        q = compute_gasteiger_charges(mol)
        expected = float(sum(a.formal_charge for a in mol.atoms))
        max_err = max(max_err, abs(float(q.sum()) - expected))
    return max_err


def run_protac_sensitivity_analysis(
    kd_target_nM: float = 15.0,
    kd_e3_nM: float = 45.0,
    alpha: float = 6.5,
) -> SensitivityReport:
    """Compute dimensionless log-elasticities of PROTAC hook peak and Dmax."""
    base = simulate_protac_campaign(kd_target_nM=kd_target_nM, kd_e3_nM=kd_e3_nM, cooperativity_alpha=alpha, hours=24.0)
    delta = 0.10
    sens: Dict[str, Tuple[float, float]] = {}

    for pname, kwargs_plus, kwargs_minus in (
        (
            "kd_target_nM",
            {"kd_target_nM": kd_target_nM * (1.0 + delta), "kd_e3_nM": kd_e3_nM, "cooperativity_alpha": alpha},
            {"kd_target_nM": kd_target_nM * (1.0 - delta), "kd_e3_nM": kd_e3_nM, "cooperativity_alpha": alpha},
        ),
        (
            "kd_e3_nM",
            {"kd_target_nM": kd_target_nM, "kd_e3_nM": kd_e3_nM * (1.0 + delta), "cooperativity_alpha": alpha},
            {"kd_target_nM": kd_target_nM, "kd_e3_nM": kd_e3_nM * (1.0 - delta), "cooperativity_alpha": alpha},
        ),
        (
            "cooperativity_alpha",
            {"kd_target_nM": kd_target_nM, "kd_e3_nM": kd_e3_nM, "cooperativity_alpha": alpha * (1.0 + delta)},
            {"kd_target_nM": kd_target_nM, "kd_e3_nM": kd_e3_nM, "cooperativity_alpha": alpha * (1.0 - delta)},
        ),
    ):
        r_p = simulate_protac_campaign(**kwargs_plus, hours=24.0)
        r_m = simulate_protac_campaign(**kwargs_minus, hours=24.0)
        d_log_param = math.log((1.0 + delta) / (1.0 - delta))
        e_peak = math.log(max(1e-9, r_p.hook_peak_nM) / max(1e-9, r_m.hook_peak_nM)) / d_log_param
        e_dmax = (r_p.dmax_fraction - r_m.dmax_fraction) / d_log_param
        sens[pname] = (float(e_peak), float(e_dmax))

    return SensitivityReport(
        baseline_hook_peak_nM=base.hook_peak_nM,
        baseline_dmax=base.dmax_fraction,
        sensitivities=sens,
    )
