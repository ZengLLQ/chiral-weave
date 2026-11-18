"""Finite-difference gradient verification and convergence tests for MolecularMPNN."""

from __future__ import annotations

import numpy as np

from chiral_weave.graph_neural import (
    MolecularMPNN,
    export_parameters_dict,
    featurize_molecule,
    has_torch_adapter,
)
from chiral_weave.smiles_parser import parse_smiles


def test_gnn_analytical_backprop_matches_finite_differences() -> None:
    """Check every parameter tensor in MolecularMPNN against central finite differences."""
    mol = parse_smiles("COc1ccc(NC(=O)c2ccccc2)cc1")
    tensor = featurize_molecule(mol)
    model = MolecularMPNN(hidden_dim=8, num_layers=2, out_dim=2, dropout_rate=0.0, seed=7)

    target = np.array([1.25, -0.45], dtype=np.float64)
    y_hat, cache = model.forward_with_cache(tensor, training=False)
    diff = y_hat - target
    grad_y = 2.0 * diff / len(target)
    ana_grads = model.backward(grad_y, cache)

    eps = 1e-6
    rng = np.random.default_rng(123)
    for name, param in model.params.items():
        # Sample up to 6 entries per parameter tensor for exhaustive fast check
        flat = param.ravel()
        indices = rng.choice(len(flat), size=min(6, len(flat)), replace=False)
        ana_flat = ana_grads[name].ravel()

        for idx in indices:
            orig = float(flat[idx])
            flat[idx] = orig + eps
            y_plus, _ = model.forward_with_cache(tensor, training=False)
            loss_plus = float(np.mean((y_plus - target) ** 2))

            flat[idx] = orig - eps
            y_minus, _ = model.forward_with_cache(tensor, training=False)
            loss_minus = float(np.mean((y_minus - target) ** 2))

            flat[idx] = orig
            num_grad = (loss_plus - loss_minus) / (2.0 * eps)
            ana_grad = float(ana_flat[idx])

            abs_err = abs(num_grad - ana_grad)
            rel_err = abs_err / max(1e-7, abs(num_grad) + abs(ana_grad))
            assert abs_err < 1e-6 or rel_err < 1e-5, (
                f"Gradient mismatch in {name}[{idx}]: analytic={ana_grad}, numeric={num_grad}"
            )


def test_gnn_training_loss_decreases_and_uncertainty_works() -> None:
    """MolecularMPNN.fit must strictly reduce MSE loss and MC-dropout must yield positive std."""
    smiles = [
        "c1ccccc1",
        "Cc1ccccc1",
        "Oc1ccccc1",
        "Nc1ccccc1",
        "Fc1ccccc1",
        "Clc1ccccc1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "COc1ccccc1OC",
    ]
    tensors = [featurize_molecule(parse_smiles(s)) for s in smiles]
    targets = np.array([0.2, 0.4, -0.3, -0.5, 0.35, 0.65, -0.2, 0.1], dtype=np.float64)

    model = MolecularMPNN(hidden_dim=16, num_layers=2, out_dim=1, dropout_rate=0.15, seed=42)
    report = model.fit(tensors, targets, epochs=18, lr=8e-3, batch_size=4, seed=42)
    assert report.final_loss < report.initial_loss * 0.65

    mean_p, std_p = model.predict_with_uncertainty(tensors[0], mc_samples=10, seed=42)
    assert mean_p.shape == (1,)
    assert float(std_p[0]) > 0.0

    emb = model.predict_embedding(tensors[0])
    assert emb.shape == (16,)
    assert isinstance(has_torch_adapter(), bool)
    exported = export_parameters_dict(model)
    assert "W_in" in exported and exported["W_in"].shape == (24, 16)


def test_gnn_permutation_invariance_under_smiles_reordering() -> None:
    """MolecularMPNN prediction must be strictly invariant to SMILES atom traversal order."""
    m1 = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    m2 = parse_smiles("O=C(O)c1ccccc1OC(=O)C")
    model = MolecularMPNN(hidden_dim=16, num_layers=2, out_dim=2, dropout_rate=0.0, seed=99)

    p1 = model.predict(featurize_molecule(m1))
    p2 = model.predict(featurize_molecule(m2))
    assert np.allclose(p1, p2, atol=1e-8)


def test_gnn_multitask_convergence_and_single_atom_graph() -> None:
    """Verify multi-output training convergence and edge-less single-atom self-loop handling."""
    single_atom = parse_smiles("O")
    t_single = featurize_molecule(single_atom)
    assert t_single.num_atoms == 1
    assert t_single.num_directed_edges == 1

    mols = [parse_smiles(s) for s in ("c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1")]
    tensors = [featurize_molecule(m) for m in mols]
    targets = np.array([[0.1, -0.2, 0.5], [0.3, -0.1, 0.4], [-0.4, 0.3, -0.2], [-0.5, 0.4, -0.3]])
    model = MolecularMPNN(hidden_dim=12, num_layers=2, out_dim=3, dropout_rate=0.0, seed=11)
    rep = model.fit(tensors, targets, epochs=15, lr=1e-2, batch_size=2, seed=11)
    assert rep.final_loss < rep.initial_loss * 0.6
