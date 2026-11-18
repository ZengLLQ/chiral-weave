"""Vectorized NumPy Graph Attention/Message-Passing Neural Network with exact analytical backprop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import importlib.util
import numpy as np

from chiral_weave.chem_types import (
    ELEMENT_TABLE,
    BondOrder,
    Hybridization,
    MoleculeGraph,
)
from chiral_weave.descriptors import compute_gasteiger_charges


ATOM_FEATURE_DIM = 24
EDGE_FEATURE_DIM = 8
_ELEM_LIST: Tuple[str, ...] = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si")
_HYB_LIST: Tuple[Hybridization, ...] = (
    Hybridization.SP,
    Hybridization.SP2,
    Hybridization.SP3,
)


@dataclass(frozen=True)
class MolecularGraphTensor:
    """Dense tensor representation of a single MoleculeGraph for GNN inference and training."""

    atom_features: np.ndarray  # (N, 24)
    edge_features: np.ndarray  # (E_dir, 8)
    edge_src: np.ndarray       # (E_dir,)
    edge_dst: np.ndarray       # (E_dir,)
    num_atoms: int
    num_directed_edges: int


def featurize_molecule(mol: MoleculeGraph) -> MolecularGraphTensor:
    """Convert MoleculeGraph into normalized atom and directed-edge feature tensors."""
    compute_gasteiger_charges(mol)
    n = mol.num_atoms
    x = np.zeros((n, ATOM_FEATURE_DIM), dtype=np.float64)

    for a in mol.atoms:
        idx = a.index
        if a.symbol in _ELEM_LIST:
            x[idx, _ELEM_LIST.index(a.symbol)] = 1.0
        else:
            x[idx, 11] = 1.0

        x[idx, 12] = mol.degree(idx) / 4.0
        x[idx, 13] = a.total_h / 4.0
        x[idx, 14] = a.formal_charge / 2.0
        x[idx, 15] = a.gasteiger_charge

        if a.hybridization in _HYB_LIST:
            x[idx, 16 + _HYB_LIST.index(a.hybridization)] = 1.0
        else:
            x[idx, 19] = 1.0

        x[idx, 20] = 1.0 if a.is_aromatic else 0.0
        x[idx, 21] = 1.0 if a.in_ring else 0.0
        x[idx, 22] = (a.ring_sizes[0] / 6.0) if a.ring_sizes else 0.0
        x[idx, 23] = ELEMENT_TABLE[a.symbol].electronegativity / 4.0

    src_list: List[int] = []
    dst_list: List[int] = []
    e_rows: List[np.ndarray] = []
    bo_orders = (BondOrder.SINGLE, BondOrder.DOUBLE, BondOrder.TRIPLE, BondOrder.AROMATIC)

    for b in mol.bonds:
        feat = np.zeros(EDGE_FEATURE_DIM, dtype=np.float64)
        if b.order in bo_orders:
            feat[bo_orders.index(b.order)] = 1.0
        feat[4] = 1.0 if b.is_aromatic else 0.0
        feat[5] = 1.0 if b.in_ring else 0.0
        feat[6] = 1.0 if b.is_conjugated else 0.0
        feat[7] = b.order.value / 3.0

        # Bidirectional message passing
        src_list.extend([b.begin_idx, b.end_idx])
        dst_list.extend([b.end_idx, b.begin_idx])
        e_rows.append(feat)
        e_rows.append(feat.copy())

    # Add self-loops if molecule has no bonds (e.g. single heavy atom)
    if not e_rows and n > 0:
        for i in range(n):
            feat = np.zeros(EDGE_FEATURE_DIM, dtype=np.float64)
            feat[0] = 1.0
            src_list.append(i)
            dst_list.append(i)
            e_rows.append(feat)

    e_mat = np.vstack(e_rows) if e_rows else np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float64)
    return MolecularGraphTensor(
        atom_features=x,
        edge_features=e_mat,
        edge_src=np.asarray(src_list, dtype=np.int64),
        edge_dst=np.asarray(dst_list, dtype=np.int64),
        num_atoms=n,
        num_directed_edges=len(src_list),
    )


def _leaky_relu(z: np.ndarray, slope: float = 0.1) -> np.ndarray:
    return np.where(z >= 0.0, z, slope * z)


def _leaky_relu_grad(z: np.ndarray, grad_out: np.ndarray, slope: float = 0.1) -> np.ndarray:
    return np.where(z >= 0.0, grad_out, slope * grad_out)


def _segment_softmax(logits: np.ndarray, dst: np.ndarray, num_nodes: int) -> np.ndarray:
    """Numerically stable per-destination-node softmax over directed edges."""
    if len(logits) == 0:
        return np.zeros(0, dtype=np.float64)
    max_per_node = np.full(num_nodes, -np.inf, dtype=np.float64)
    np.maximum.at(max_per_node, dst, logits)
    shifted = logits - max_per_node[dst]
    exp_vals = np.exp(shifted)
    sum_per_node = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(sum_per_node, dst, exp_vals)
    return np.asarray(exp_vals / np.maximum(sum_per_node[dst], 1e-12), dtype=np.float64)


def _segment_softmax_grad(
    alpha: np.ndarray,
    grad_alpha: np.ndarray,
    dst: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """Exact vector-Jacobian product for segmented softmax over destination nodes."""
    if len(alpha) == 0:
        return np.zeros(0, dtype=np.float64)
    dot = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(dot, dst, alpha * grad_alpha)
    return np.asarray(alpha * (grad_alpha - dot[dst]), dtype=np.float64)


def _layernorm_forward(
    x: np.ndarray,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Row-wise LayerNorm without affine parameters (zero mean, unit variance)."""
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    var = np.mean(centered ** 2, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    y = centered * inv_std
    return y, (centered, inv_std, y)


def _layernorm_backward(
    grad_y: np.ndarray,
    cache: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Exact backward pass through row-wise LayerNorm."""
    _, inv_std, y = cache
    mean_g = grad_y.mean(axis=-1, keepdims=True)
    mean_gy = np.mean(grad_y * y, axis=-1, keepdims=True)
    return np.asarray(inv_std * (grad_y - mean_g - y * mean_gy), dtype=np.float64)


@dataclass
class _LayerCache:
    h_in: np.ndarray
    z_edge: np.ndarray
    a_edge: np.ndarray
    logits: np.ndarray
    alpha: np.ndarray
    u_pre_ln: np.ndarray
    ln_cache: Tuple[np.ndarray, np.ndarray, np.ndarray]
    h_ln: np.ndarray
    drop_mask: np.ndarray


@dataclass
class _ForwardCache:
    tensor: MolecularGraphTensor
    z_in: np.ndarray
    h0: np.ndarray
    layer_caches: List[_LayerCache]
    h_final: np.ndarray
    z_gate: np.ndarray
    a_gate: np.ndarray
    beta: np.ndarray
    z_mol: np.ndarray
    z_mlp: np.ndarray
    h_mlp: np.ndarray


@dataclass
class GnnTrainingReport:
    """Summary of MolecularMPNN training and convergence."""

    epochs: int
    initial_loss: float
    final_loss: float
    loss_history: List[float]


class MolecularMPNN:
    """Multi-layer GATv2/MPNN Graph Neural Network implemented in pure NumPy.

    Supports exact analytical backpropagation, AdamW optimization, and
    Monte Carlo Dropout epistemic uncertainty estimation.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 3,
        out_dim: int = 1,
        dropout_rate: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.params: Dict[str, np.ndarray] = {}
        self._adam_m: Dict[str, np.ndarray] = {}
        self._adam_v: Dict[str, np.ndarray] = {}
        self._adam_step = 0
        self._init_weights(np.random.default_rng(seed))

    def _init_weights(self, rng: np.random.Generator) -> None:
        d = self.hidden_dim

        def _glorot(rows: int, cols: int) -> np.ndarray:
            scale = np.sqrt(2.0 / (rows + cols))
            return rng.normal(0.0, scale, size=(rows, cols)).astype(np.float64)

        self.params["W_in"] = _glorot(ATOM_FEATURE_DIM, d)
        self.params["b_in"] = np.zeros(d, dtype=np.float64)

        for layer in range(self.num_layers):
            self.params[f"W_src_{layer}"] = _glorot(d, d)
            self.params[f"W_dst_{layer}"] = _glorot(d, d)
            self.params[f"W_edge_{layer}"] = _glorot(EDGE_FEATURE_DIM, d)
            self.params[f"b_attn_{layer}"] = np.zeros(d, dtype=np.float64)
            self.params[f"v_attn_{layer}"] = rng.normal(0.0, 1.0 / np.sqrt(d), size=d).astype(np.float64)
            self.params[f"W_out_{layer}"] = _glorot(d, d)
            self.params[f"b_out_{layer}"] = np.zeros(d, dtype=np.float64)

        self.params["W_gate"] = _glorot(d, d)
        self.params["b_gate"] = np.zeros(d, dtype=np.float64)
        self.params["v_gate"] = rng.normal(0.0, 1.0 / np.sqrt(d), size=d).astype(np.float64)

        self.params["W_mlp1"] = _glorot(d, d)
        self.params["b_mlp1"] = np.zeros(d, dtype=np.float64)
        self.params["W_mlp2"] = _glorot(d, self.out_dim)
        self.params["b_mlp2"] = np.zeros(self.out_dim, dtype=np.float64)

        for k, v in self.params.items():
            self._adam_m[k] = np.zeros_like(v)
            self._adam_v[k] = np.zeros_like(v)

    def forward_with_cache(
        self,
        tensor: MolecularGraphTensor,
        training: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, _ForwardCache]:
        """Run forward pass and return prediction vector and backprop cache."""
        x = tensor.atom_features
        e = tensor.edge_features
        src = tensor.edge_src
        dst = tensor.edge_dst
        n = tensor.num_atoms

        z_in = x @ self.params["W_in"] + self.params["b_in"]
        h = _leaky_relu(z_in)
        h0 = h

        layer_caches: List[_LayerCache] = []
        for layer in range(self.num_layers):
            h_in = h
            if tensor.num_directed_edges > 0:
                z_edge = (
                    h_in[src] @ self.params[f"W_src_{layer}"]
                    + h_in[dst] @ self.params[f"W_dst_{layer}"]
                    + e @ self.params[f"W_edge_{layer}"]
                    + self.params[f"b_attn_{layer}"]
                )
                a_edge = _leaky_relu(z_edge)
                logits = a_edge @ self.params[f"v_attn_{layer}"]
                alpha = _segment_softmax(logits, dst, n)

                weighted_msg = alpha[:, None] * a_edge
                agg = np.zeros((n, self.hidden_dim), dtype=np.float64)
                np.add.at(agg, dst, weighted_msg)
            else:
                z_edge = np.zeros((0, self.hidden_dim), dtype=np.float64)
                a_edge = np.zeros((0, self.hidden_dim), dtype=np.float64)
                logits = np.zeros(0, dtype=np.float64)
                alpha = np.zeros(0, dtype=np.float64)
                agg = np.zeros((n, self.hidden_dim), dtype=np.float64)

            u_pre_ln = h_in + agg @ self.params[f"W_out_{layer}"] + self.params[f"b_out_{layer}"]
            h_ln, ln_cache = _layernorm_forward(u_pre_ln)

            if training and self.dropout_rate > 0.0:
                gen = rng if rng is not None else np.random.default_rng(0)
                keep_prob = 1.0 - self.dropout_rate
                drop_mask = (gen.random(h_ln.shape) < keep_prob).astype(np.float64) / keep_prob
            else:
                drop_mask = np.ones_like(h_ln)

            h = h_ln * drop_mask
            layer_caches.append(
                _LayerCache(
                    h_in=h_in,
                    z_edge=z_edge,
                    a_edge=a_edge,
                    logits=logits,
                    alpha=alpha,
                    u_pre_ln=u_pre_ln,
                    ln_cache=ln_cache,
                    h_ln=h_ln,
                    drop_mask=drop_mask,
                )
            )

        # Attentive global pooling
        z_gate = h @ self.params["W_gate"] + self.params["b_gate"]
        a_gate = _leaky_relu(z_gate)
        gate_logits = a_gate @ self.params["v_gate"]
        gate_shift = gate_logits - np.max(gate_logits)
        exp_g = np.exp(gate_shift)
        beta = exp_g / np.maximum(np.sum(exp_g), 1e-12)
        z_mol = (beta[:, None] * h).sum(axis=0)

        # Readout MLP
        z_mlp = z_mol @ self.params["W_mlp1"] + self.params["b_mlp1"]
        h_mlp = _leaky_relu(z_mlp)
        y_hat = h_mlp @ self.params["W_mlp2"] + self.params["b_mlp2"]

        cache = _ForwardCache(
            tensor=tensor,
            z_in=z_in,
            h0=h0,
            layer_caches=layer_caches,
            h_final=h,
            z_gate=z_gate,
            a_gate=a_gate,
            beta=beta,
            z_mol=z_mol,
            z_mlp=z_mlp,
            h_mlp=h_mlp,
        )
        return y_hat, cache

    def backward(self, grad_y: np.ndarray, cache: _ForwardCache) -> Dict[str, np.ndarray]:
        """Compute exact analytical gradients for all model parameters."""
        grads: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.params.items()}
        tensor = cache.tensor
        n = tensor.num_atoms
        src = tensor.edge_src
        dst = tensor.edge_dst

        # Readout MLP backward
        grads["W_mlp2"] = np.outer(cache.h_mlp, grad_y)
        grads["b_mlp2"] = grad_y.copy()
        grad_h_mlp = grad_y @ self.params["W_mlp2"].T

        grad_z_mlp = _leaky_relu_grad(cache.z_mlp, grad_h_mlp)
        grads["W_mlp1"] = np.outer(cache.z_mol, grad_z_mlp)
        grads["b_mlp1"] = grad_z_mlp.copy()
        grad_z_mol = grad_z_mlp @ self.params["W_mlp1"].T

        # Attentive global pooling backward
        grad_h = cache.beta[:, None] * grad_z_mol[None, :]
        grad_beta = (cache.h_final * grad_z_mol[None, :]).sum(axis=1)
        beta_dot = float(np.sum(cache.beta * grad_beta))
        grad_gate_logits = cache.beta * (grad_beta - beta_dot)

        grads["v_gate"] = (cache.a_gate * grad_gate_logits[:, None]).sum(axis=0)
        grad_a_gate = grad_gate_logits[:, None] * self.params["v_gate"][None, :]
        grad_z_gate = _leaky_relu_grad(cache.z_gate, grad_a_gate)
        grads["W_gate"] = cache.h_final.T @ grad_z_gate
        grads["b_gate"] = grad_z_gate.sum(axis=0)
        grad_h = grad_h + grad_z_gate @ self.params["W_gate"].T

        # Message passing layers backward
        for layer in range(self.num_layers - 1, -1, -1):
            lc = cache.layer_caches[layer]
            grad_h_ln = grad_h * lc.drop_mask
            grad_u = _layernorm_backward(grad_h_ln, lc.ln_cache)

            grad_h_res = grad_u
            if tensor.num_directed_edges > 0:
                agg = np.zeros((n, self.hidden_dim), dtype=np.float64)
                np.add.at(agg, dst, lc.alpha[:, None] * lc.a_edge)
                grads[f"W_out_{layer}"] = agg.T @ grad_u
                grads[f"b_out_{layer}"] = grad_u.sum(axis=0)
                grad_agg = grad_u @ self.params[f"W_out_{layer}"].T

                grad_agg_dst = grad_agg[dst]
                grad_alpha = np.sum(grad_agg_dst * lc.a_edge, axis=1)
                grad_a_edge = lc.alpha[:, None] * grad_agg_dst

                grad_logits = _segment_softmax_grad(lc.alpha, grad_alpha, dst, n)
                grads[f"v_attn_{layer}"] = (lc.a_edge * grad_logits[:, None]).sum(axis=0)
                grad_a_edge = grad_a_edge + grad_logits[:, None] * self.params[f"v_attn_{layer}"][None, :]

                grad_z_edge = _leaky_relu_grad(lc.z_edge, grad_a_edge)
                grads[f"W_src_{layer}"] = lc.h_in[src].T @ grad_z_edge
                grads[f"W_dst_{layer}"] = lc.h_in[dst].T @ grad_z_edge
                grads[f"W_edge_{layer}"] = tensor.edge_features.T @ grad_z_edge
                grads[f"b_attn_{layer}"] = grad_z_edge.sum(axis=0)

                grad_h_from_src = grad_z_edge @ self.params[f"W_src_{layer}"].T
                grad_h_from_dst = grad_z_edge @ self.params[f"W_dst_{layer}"].T
                grad_h_in = grad_h_res.copy()
                np.add.at(grad_h_in, src, grad_h_from_src)
                np.add.at(grad_h_in, dst, grad_h_from_dst)
            else:
                grads[f"b_out_{layer}"] = grad_u.sum(axis=0)
                grad_h_in = grad_h_res

            grad_h = grad_h_in

        # Input projection backward
        grad_z_in = _leaky_relu_grad(cache.z_in, grad_h)
        grads["W_in"] = tensor.atom_features.T @ grad_z_in
        grads["b_in"] = grad_z_in.sum(axis=0)
        return grads

    def apply_adamw(
        self,
        grads: Dict[str, np.ndarray],
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        clip_norm: float = 5.0,
    ) -> None:
        """Apply a decoupled AdamW update step with global gradient norm clipping."""
        total_sq = sum(float(np.sum(g ** 2)) for g in grads.values())
        norm = float(np.sqrt(total_sq))
        scale = min(1.0, clip_norm / max(norm, 1e-12))

        self._adam_step += 1
        t = self._adam_step
        bc1 = 1.0 - (beta1 ** t)
        bc2 = 1.0 - (beta2 ** t)

        for name, param in self.params.items():
            g = grads[name] * scale
            self._adam_m[name] = beta1 * self._adam_m[name] + (1.0 - beta1) * g
            self._adam_v[name] = beta2 * self._adam_v[name] + (1.0 - beta2) * (g ** 2)
            m_hat = self._adam_m[name] / bc1
            v_hat = self._adam_v[name] / bc2
            if param.ndim >= 2:
                param -= lr * weight_decay * param
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(
        self,
        tensors: Sequence[MolecularGraphTensor],
        targets: np.ndarray,
        epochs: int = 25,
        lr: float = 4e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 8,
        seed: int = 42,
    ) -> GnnTrainingReport:
        """Train MolecularMPNN on molecular graph targets using mini-batch MSE + AdamW."""
        if len(tensors) != len(targets):
            raise ValueError("tensors and targets must have matching lengths.")
        n_samples = len(tensors)
        targets_2d = np.asarray(targets, dtype=np.float64)
        if targets_2d.ndim == 1:
            targets_2d = targets_2d[:, None]

        rng = np.random.default_rng(seed)
        loss_history: List[float] = []

        for _ in range(epochs):
            perm = rng.permutation(n_samples)
            epoch_loss = 0.0
            for start in range(0, n_samples, batch_size):
                batch_ids = perm[start : start + batch_size]
                accum_grads: Optional[Dict[str, np.ndarray]] = None
                bs = len(batch_ids)
                for idx in batch_ids:
                    y_hat, cache = self.forward_with_cache(tensors[int(idx)], training=True, rng=rng)
                    diff = y_hat - targets_2d[int(idx)]
                    epoch_loss += float(np.mean(diff ** 2))
                    grad_y = (2.0 / (self.out_dim * bs)) * diff
                    sample_grads = self.backward(grad_y, cache)
                    if accum_grads is None:
                        accum_grads = sample_grads
                    else:
                        for k in accum_grads:
                            accum_grads[k] += sample_grads[k]
                assert accum_grads is not None
                self.apply_adamw(accum_grads, lr=lr, weight_decay=weight_decay)
            loss_history.append(epoch_loss / max(1, n_samples))

        return GnnTrainingReport(
            epochs=epochs,
            initial_loss=loss_history[0] if loss_history else 0.0,
            final_loss=loss_history[-1] if loss_history else 0.0,
            loss_history=loss_history,
        )

    def predict(self, tensor: MolecularGraphTensor) -> np.ndarray:
        """Deterministic inference without dropout."""
        y_hat, _ = self.forward_with_cache(tensor, training=False)
        return y_hat

    def predict_embedding(self, tensor: MolecularGraphTensor) -> np.ndarray:
        """Extract learned graph-level embedding z_mol."""
        _, cache = self.forward_with_cache(tensor, training=False)
        return cache.z_mol

    def predict_with_uncertainty(
        self,
        tensor: MolecularGraphTensor,
        mc_samples: int = 12,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate predictive mean and epistemic uncertainty via MC-Dropout passes."""
        rng = np.random.default_rng(seed)
        preds: List[np.ndarray] = []
        for _ in range(max(2, mc_samples)):
            y_hat, _ = self.forward_with_cache(tensor, training=True, rng=rng)
            preds.append(y_hat)
        stacked = np.vstack(preds)
        return stacked.mean(axis=0), stacked.std(axis=0)


def has_torch_adapter() -> bool:
    """Return True if optional PyTorch backend is importable in the environment."""
    return importlib.util.find_spec("torch") is not None


def export_parameters_dict(model: MolecularMPNN) -> Dict[str, np.ndarray]:
    """Export deep copies of trained NumPy GNN parameters for serialization or Torch transfer."""
    return {k: v.copy() for k, v in model.params.items()}
