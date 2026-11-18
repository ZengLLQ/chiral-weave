"""Example 04: MCTS + NSGA-II Pareto de novo lead optimization from a quinazoline hit."""

from __future__ import annotations

from chiral_weave import optimize_lead_mcts
from chiral_weave.visualizer import render_pareto_ascii_plot


def main() -> None:
    seed_smiles = "c1ccc(Nc2ncnc3ccccc23)cc1"
    report = optimize_lead_mcts(seed_smiles=seed_smiles, iterations=20, max_depth=3, seed=42)
    print(f"Evaluated {report.unique_evaluated} unique molecules | Pareto Front size={len(report.pareto_front)}")
    print(
        f"Best Candidate: {report.best_candidate.canonical_smiles} | "
        f"MPO={report.best_candidate.mpo_score:.3f} | dG={report.best_candidate.docking_kcal:.2f} kcal/mol | "
        f"QED={report.best_candidate.qed:.3f} | SA={report.best_candidate.sa_score:.2f}"
    )
    print(render_pareto_ascii_plot(report.top_candidates, report.pareto_front))


if __name__ == "__main__":
    main()
