"""Example 03: 3D Distance Geometry docking and PROTAC ternary complex hook-effect simulation."""

from __future__ import annotations

from chiral_weave import (
    build_kinase_hinge_pocket,
    dock_molecule,
    parse_smiles,
    simulate_protac_campaign,
    verify_chiral_3d_inversion,
)
from chiral_weave.visualizer import render_docking_ascii_breakdown, render_protac_ascii_curves


def main() -> None:
    chiral = verify_chiral_3d_inversion(seed=42)
    print(
        f"Chiral 3D Check: V(@)={chiral.volume_ccw:+.3f}, V(@@)={chiral.volume_cw:+.3f}, "
        f"Preserved={chiral.opposite_handedness_preserved}"
    )

    pocket = build_kinase_hinge_pocket()
    erlotinib = parse_smiles("COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC", name="Erlotinib")
    pose = dock_molecule(erlotinib, pocket=pocket, steps=50, seed=42)
    print(render_docking_ascii_breakdown(pose, pocket_name=pocket.name))

    protac = simulate_protac_campaign(kd_target_nM=12.0, kd_e3_nM=40.0, cooperativity_alpha=8.0, hours=24.0)
    print(render_protac_ascii_curves(protac))


if __name__ == "__main__":
    main()
