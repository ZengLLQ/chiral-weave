"""Example 01: SMILES parsing, SSSR rings, Gasteiger charges, Murcko scaffolds, and MMP cliffs."""

from __future__ import annotations

from chiral_weave import (
    CompoundLibrary,
    butina_cluster,
    compute_physicochemical_profile,
    extract_mmp_pairs,
    parse_smiles,
)
from chiral_weave.visualizer import render_molecule_ascii_summary


def main() -> None:
    gefitinib = parse_smiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", name="Gefitinib")
    prof = compute_physicochemical_profile(gefitinib)
    print(render_molecule_ascii_summary(gefitinib, prof))

    lib = CompoundLibrary.default_benchmark_library()
    clust = butina_cluster(lib.fingerprints, similarity_threshold=0.45)
    print(f"\nBenchmark library ({len(lib.entries)} drugs) clustered into {len(clust.clusters)} Butina groups.")
    mmps = extract_mmp_pairs(lib.molecules, lib.labels_array())
    print(f"Detected {len(mmps)} single-cut Matched Molecular Pairs:")
    for pair in mmps[:5]:
        print(
            f"  {pair.r_group_a:<10} -> {pair.r_group_b:<10} | "
            f"dAct={pair.delta_activity:+.2f} | SALI={pair.sali_score:.2f}"
        )


if __name__ == "__main__":
    main()
