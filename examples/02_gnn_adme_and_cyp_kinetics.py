"""Example 02: Graph Neural Network training on scaffold splits and CYP450 MBI DDI kinetics."""

from __future__ import annotations

from chiral_weave import (
    AdmePredictor,
    parse_smiles,
    rank_metabolism_sites,
    run_scaffold_gnn_benchmark,
    simulate_cyp_mbi_kinetics,
)


def main() -> None:
    metrics = run_scaffold_gnn_benchmark(epochs=15, hidden_dim=24, seed=42)
    print("Scaffold-Split MolecularMPNN Calibration:")
    print(f"  Train={metrics.num_train}, Test={metrics.num_test}")
    print(f"  Loss: {metrics.train_initial_loss:.4f} -> {metrics.train_final_loss:.4f}")
    print(
        f"  Test RMSE={metrics.test_rmse:.3f} | Pearson r={metrics.pearson_r:.3f} | "
        f"EF20={metrics.enrichment_factor_20pct:.2f}x"
    )

    pred = AdmePredictor(hidden_dim=24, seed=42)
    imatinib = parse_smiles("Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", name="Imatinib")
    adme = pred.predict(imatinib, inhibitor_conc_uM=2.0, seed=42)
    print(f"\nImatinib ADME: logS={adme.log_s_esol:.2f}, logBB={adme.log_bb:.2f}, EH={adme.hepatic_extraction:.3f}")
    print(f"  CYP3A4 IC50={adme.cyp3a4_ic50_uM:.2f} uM | DDI AUC Ratio={adme.cyp3a4_ddi_auc_ratio:.2f}")

    cyp = simulate_cyp_mbi_kinetics("CYP3A4", inhibitor_uM=2.0, ic50_uM=adme.cyp3a4_ic50_uM)
    print(f"  Residual Active CYP3A4 (120 min): {cyp.residual_active_enzyme * 100:.1f}%")
    sites = rank_metabolism_sites(imatinib)
    print("  Top SoM Soft Spots:", [(s.atom_index, s.reaction_type, round(s.lability_score, 2)) for s in sites[:3]])


if __name__ == "__main__":
    main()
