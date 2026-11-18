"""Tests for strict 7-bit ASCII visualizers, scaffold benchmarks, and all 6 CLI subcommands."""

from __future__ import annotations

from chiral_weave.adme_kinetics import AdmePredictor
from chiral_weave.calibration import run_protac_sensitivity_analysis, run_scaffold_gnn_benchmark
from chiral_weave.chemical_library import CompoundLibrary
from chiral_weave.cli import main as cli_main
from chiral_weave.conformer_docking import dock_molecule
from chiral_weave.descriptors import compute_physicochemical_profile
from chiral_weave.fingerprints import pairwise_tanimoto_matrix
from chiral_weave.molecular_optimizer import optimize_lead_mcts
from chiral_weave.protac_cooperativity import simulate_protac_campaign
from chiral_weave.smiles_parser import parse_smiles
from chiral_weave.visualizer import (
    assert_ascii,
    render_docking_ascii_breakdown,
    render_molecule_ascii_summary,
    render_pareto_ascii_plot,
    render_protac_ascii_curves,
    render_similarity_heatmap_ascii,
)


def test_all_visualizers_emit_strict_7bit_ascii() -> None:
    """Every terminal visualization string must contain exclusively ASCII codes 0-127."""
    mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O", name="Aspirin")
    prof = compute_physicochemical_profile(mol)
    adme = AdmePredictor(hidden_dim=16, seed=42).predict(mol)
    pose = dock_molecule(mol, steps=25, seed=42)
    protac = simulate_protac_campaign()
    lib = CompoundLibrary.default_benchmark_library()
    sim = pairwise_tanimoto_matrix(lib.fingerprints[:6])
    opt = optimize_lead_mcts("c1ccc(Nc2ncnc3ccccc23)cc1", iterations=8, max_depth=2, seed=42)

    outputs = [
        render_molecule_ascii_summary(mol, prof, adme),
        render_docking_ascii_breakdown(pose),
        render_protac_ascii_curves(protac),
        render_similarity_heatmap_ascii(sim, [e.name for e in lib.entries[:6]], max_items=6),
        render_pareto_ascii_plot(opt.top_candidates, opt.pareto_front),
    ]
    for out in outputs:
        assert_ascii(out)
        assert len(out.splitlines()) >= 6


def test_library_provenance_and_calibration_benchmarks() -> None:
    """CompoundLibrary provenance SHA-256 must be deterministic and scaffold benchmark must train."""
    lib1 = CompoundLibrary.default_benchmark_library()
    lib2 = CompoundLibrary.default_benchmark_library()
    assert lib1.provenance_sha256() == lib2.provenance_sha256()

    bm = run_scaffold_gnn_benchmark(epochs=10, hidden_dim=16, seed=42)
    assert bm.train_final_loss <= bm.train_initial_loss
    assert bm.num_train > 0 and bm.num_test > 0

    sens = run_protac_sensitivity_analysis()
    assert "cooperativity_alpha" in sens.sensitivities


def test_all_six_cli_subcommands_run_cleanly(capsys) -> None:
    """All 6 CLI subcommands must return exit code 0 and emit valid 7-bit ASCII output."""
    commands = [
        ["inspect", "--smiles", "CC(=O)Oc1ccccc1C(=O)O", "--name", "Aspirin"],
        ["dock", "--smiles", "c1ccc(Nc2ncnc3ccccc23)cc1", "--steps", "25", "--seed", "42"],
        ["protac", "--kd-target", "15.0", "--kd-e3", "45.0", "--alpha", "6.5", "--hours", "24.0"],
        ["screen", "--threshold", "0.4", "--clusters", "3", "--seed", "42"],
        ["optimize", "--smiles", "c1ccc(Nc2ncnc3ccccc23)cc1", "--iterations", "10", "--seed", "42"],
        ["benchmark", "--epochs", "8", "--seed", "42"],
    ]
    for argv in commands:
        rc = cli_main(argv)
        assert rc == 0
        captured = capsys.readouterr()
        assert_ascii(captured.out)
        assert len(captured.out.strip()) > 40


def test_assert_ascii_rejects_non_ascii_and_scaffold_split_disjoint() -> None:
    """assert_ascii must reject >127 codepoints and scaffold split must share zero Murcko scaffolds."""
    import pytest
    from chiral_weave.calibration import scaffold_train_test_split

    with pytest.raises(ValueError):
        assert_ascii("delta G = -5.2 \u0394")

    lib = CompoundLibrary.synthetic_sar_cohort(seed=42)
    tr, te = scaffold_train_test_split(lib, test_fraction=0.25)
    assert set(tr).isdisjoint(set(te))
    tr_scaffs = {lib.profiles[i].bemis_murcko_scaffold for i in tr if lib.profiles[i].bemis_murcko_scaffold}
    te_scaffs = {lib.profiles[i].bemis_murcko_scaffold for i in te if lib.profiles[i].bemis_murcko_scaffold}
    assert tr_scaffs.isdisjoint(te_scaffs)


def test_default_library_series_and_charge_conservation_gate() -> None:
    """Default 24-compound library must cover multiple series and pass Gasteiger charge conservation."""
    from chiral_weave.calibration import verify_charge_conservation

    lib = CompoundLibrary.default_benchmark_library()
    assert len(lib.entries) == 24
    assert len(lib.series_map()) >= 5
    max_err = verify_charge_conservation([e.smiles for e in lib.entries])
    assert max_err < 1e-10
