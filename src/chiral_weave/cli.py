"""Command-line interface for ChiralWeave cheminformatics and drug-discovery workflows."""

from __future__ import annotations

import argparse
from typing import List, Optional

from chiral_weave.adme_kinetics import AdmePredictor, rank_metabolism_sites, simulate_cyp_mbi_kinetics
from chiral_weave.calibration import (
    run_protac_sensitivity_analysis,
    run_scaffold_gnn_benchmark,
    verify_charge_conservation,
    verify_chiral_3d_inversion,
)
from chiral_weave.chemical_library import CompoundLibrary
from chiral_weave.conformer_docking import build_kinase_hinge_pocket, dock_molecule
from chiral_weave.descriptors import compute_physicochemical_profile
from chiral_weave.fingerprints import butina_cluster, extract_mmp_pairs, pairwise_tanimoto_matrix
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


def _cmd_inspect(args: argparse.Namespace) -> int:
    mol = parse_smiles(args.smiles, name=args.name)
    prof = compute_physicochemical_profile(mol)
    adme = AdmePredictor(hidden_dim=24, seed=args.seed).predict(mol, inhibitor_conc_uM=1.0, seed=args.seed)
    cyp = simulate_cyp_mbi_kinetics("CYP3A4", inhibitor_uM=1.0, ic50_uM=adme.cyp3a4_ic50_uM)
    hotspots = rank_metabolism_sites(mol)

    print(render_molecule_ascii_summary(mol, prof, adme))
    print(f"CYP3A4 MBI residual active enzyme (120 min @ 1 uM): {cyp.residual_active_enzyme:.3f}")
    if hotspots:
        top_som = ", ".join(f"atom[{s.atom_index}]({s.reaction_type}:{s.lability_score:.2f})" for s in hotspots[:3])
        print(assert_ascii(f"Top Metabolism Soft Spots: {top_som}"))
    return 0


def _cmd_dock(args: argparse.Namespace) -> int:
    mol = parse_smiles(args.smiles, name="DockingLigand")
    pocket = build_kinase_hinge_pocket()
    pose = dock_molecule(mol, pocket=pocket, steps=args.steps, seed=args.seed)
    print(render_docking_ascii_breakdown(pose, pocket_name=pocket.name))
    return 0


def _cmd_protac(args: argparse.Namespace) -> int:
    outcome = simulate_protac_campaign(
        kd_target_nM=args.kd_target,
        kd_e3_nM=args.kd_e3,
        cooperativity_alpha=args.alpha,
        p_tot_nM=args.p_tot,
        e_tot_nM=args.e_tot,
        hours=args.hours,
    )
    print(render_protac_ascii_curves(outcome))
    return 0


def _cmd_screen(args: argparse.Namespace) -> int:
    lib = CompoundLibrary.default_benchmark_library()
    adme_pred = AdmePredictor(hidden_dim=24, seed=args.seed)
    clust = butina_cluster(lib.fingerprints, similarity_threshold=args.threshold)
    sim_mat = pairwise_tanimoto_matrix(lib.fingerprints)
    mmps = extract_mmp_pairs(lib.molecules, lib.labels_array().tolist())

    print(assert_ascii(f"Library SHA-256 Provenance: {lib.provenance_sha256()[:24]}... ({len(lib.entries)} compounds)"))
    print(
        assert_ascii(
            f"Butina Clusters (Tanimoto >= {args.threshold:.2f}): "
            f"{len(clust.clusters)} clusters ({clust.singletons} singletons)"
        )
    )
    for cid, members in enumerate(clust.clusters[: args.clusters]):
        names = ", ".join(lib.entries[m].name for m in members[:4])
        print(assert_ascii(f"  Cluster {cid}: size={len(members)} | {names}"))

    print(render_similarity_heatmap_ascii(sim_mat, [e.name for e in lib.entries], max_items=8))
    if mmps:
        print(assert_ascii("Top Matched Molecular Pair Activity Cliffs (SALI):"))
        for pair in mmps[:3]:
            print(
                assert_ascii(
                    f"  {pair.r_group_a} -> {pair.r_group_b} | dAct={pair.delta_activity:+.2f} | "
                    f"Sim={pair.tanimoto_sim:.3f} | SALI={pair.sali_score:.2f}"
                )
            )
    sample_adme = adme_pred.predict(lib.molecules[4], seed=args.seed)
    print(
        assert_ascii(
            f"Reference Kinase Inhibitor ({lib.entries[4].name}) ADME: logS={sample_adme.log_s_esol:.2f}, "
            f"EH={sample_adme.hepatic_extraction:.3f}, DDI_AUCr={sample_adme.cyp3a4_ddi_auc_ratio:.2f}"
        )
    )
    return 0


def _cmd_optimize(args: argparse.Namespace) -> int:
    report = optimize_lead_mcts(
        seed_smiles=args.smiles,
        iterations=args.iterations,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    print(
        assert_ascii(
            f"MCTS Lead Optimization: evaluated={report.unique_evaluated} unique molecules | "
            f"Pareto front={len(report.pareto_front)}"
        )
    )
    print(
        assert_ascii(
            f"Best MPO Candidate: {report.best_candidate.canonical_smiles} | "
            f"MPO={report.best_candidate.mpo_score:.3f} | dG={report.best_candidate.docking_kcal:.2f} kcal/mol | "
            f"QED={report.best_candidate.qed:.3f} | SA={report.best_candidate.sa_score:.2f}"
        )
    )
    print(render_pareto_ascii_plot(report.top_candidates, report.pareto_front))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    gnn_metrics = run_scaffold_gnn_benchmark(epochs=args.epochs, seed=args.seed)
    chiral_check = verify_chiral_3d_inversion(seed=args.seed)
    lib = CompoundLibrary.default_benchmark_library()
    max_q_err = verify_charge_conservation([e.smiles for e in lib.entries[:10]])
    sens = run_protac_sensitivity_analysis()

    print(assert_ascii("=== ChiralWeave Scientific Calibration & Conservation Benchmarks ==="))
    print(
        assert_ascii(
            f"1. Scaffold-Split GNN (train={gnn_metrics.num_train}, test={gnn_metrics.num_test}): "
            f"Loss {gnn_metrics.train_initial_loss:.4f}->{gnn_metrics.train_final_loss:.4f} | "
            f"RMSE={gnn_metrics.test_rmse:.3f} | Pearson r={gnn_metrics.pearson_r:.3f} | "
            f"EF20={gnn_metrics.enrichment_factor_20pct:.2f}x"
        )
    )
    print(
        assert_ascii(
            f"2. 3D Distance Geometry Chiral Volume Check: "
            f"V(@)={chiral_check.volume_ccw:+.3f}, V(@@)={chiral_check.volume_cw:+.3f} | "
            f"Preserved={chiral_check.opposite_handedness_preserved}"
        )
    )
    print(assert_ascii(f"3. Gasteiger Charge Conservation Max Error: {max_q_err:.3e}"))
    print(
        assert_ascii(
            f"4. PROTAC Sensitivity (Hook Peak={sens.baseline_hook_peak_nM:.2f} nM, "
            f"Dmax={sens.baseline_dmax*100:.1f}%): "
            + ", ".join(f"{k}:(peak_e={v[0]:+.2f},dmax_e={v[1]:+.2f})" for k, v in sens.sensitivities.items())
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the root argparse CLI parser with all 6 scientific subcommands."""
    parser = argparse.ArgumentParser(
        prog="chiral-weave",
        description="ChiralWeave computational cheminformatics, GNN ADME, docking, and PROTAC simulator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ins = sub.add_parser("inspect", help="Parse SMILES, descriptors, Gasteiger charges, and ADME/CYP profile.")
    p_ins.add_argument("--smiles", required=True, help="Input SMILES string.")
    p_ins.add_argument("--name", default="QueryMolecule", help="Molecule display name.")
    p_ins.add_argument("--seed", type=int, default=42, help="PRNG seed.")
    p_ins.set_defaults(func=_cmd_inspect)

    p_dock = sub.add_parser("dock", help="Generate 3D conformer and dock into ATP kinase hinge pocket.")
    p_dock.add_argument("--smiles", required=True, help="Ligand SMILES string.")
    p_dock.add_argument("--steps", type=int, default=60, help="Pose optimization effort steps.")
    p_dock.add_argument("--seed", type=int, default=42, help="PRNG seed.")
    p_dock.set_defaults(func=_cmd_dock)

    p_prot = sub.add_parser("protac", help="Simulate PROTAC ternary equilibrium hook effect and degradation ODEs.")
    p_prot.add_argument("--kd-target", type=float, default=15.0, help="Target binary Kd (nM).")
    p_prot.add_argument("--kd-e3", type=float, default=45.0, help="E3 ligase binary Kd (nM).")
    p_prot.add_argument("--alpha", type=float, default=6.5, help="Ternary cooperativity factor alpha.")
    p_prot.add_argument("--p-tot", type=float, default=100.0, help="Total target concentration (nM).")
    p_prot.add_argument("--e-tot", type=float, default=50.0, help="Total E3 concentration (nM).")
    p_prot.add_argument("--hours", type=float, default=24.0, help="Degradation ODE horizon (hours).")
    p_prot.set_defaults(func=_cmd_protac)

    p_scr = sub.add_parser("screen", help="Cluster benchmark library, detect MMP activity cliffs, and predict ADME.")
    p_scr.add_argument("--threshold", type=float, default=0.40, help="Butina Tanimoto clustering threshold.")
    p_scr.add_argument("--clusters", type=int, default=5, help="Number of top clusters to print.")
    p_scr.add_argument("--seed", type=int, default=42, help="PRNG seed.")
    p_scr.set_defaults(func=_cmd_screen)

    p_opt = sub.add_parser("optimize", help="Run MCTS + Pareto multi-objective de novo lead optimization.")
    p_opt.add_argument("--smiles", required=True, help="Seed scaffold SMILES.")
    p_opt.add_argument("--iterations", type=int, default=25, help="MCTS iterations.")
    p_opt.add_argument("--max-depth", type=int, default=3, help="Maximum MCTS tree depth.")
    p_opt.add_argument("--seed", type=int, default=42, help="PRNG seed.")
    p_opt.set_defaults(func=_cmd_optimize)

    p_bm = sub.add_parser("benchmark", help="Run scaffold-split GNN, chirality, and thermodynamic conservation checks.")
    p_bm.add_argument("--epochs", type=int, default=15, help="GNN training epochs.")
    p_bm.add_argument("--seed", type=int, default=42, help="PRNG seed.")
    p_bm.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for CLI execution."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
