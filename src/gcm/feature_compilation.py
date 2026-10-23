"""Compile feature-nearby graph nodes into the same butterfly wire groups.

The compiler uses only the supplied, fixed input features. It never reads
labels, graph splits, model weights, or intermediate learned representations.
The resulting order can be cached and reused by every forward pass.
"""

from __future__ import annotations

import numpy as np
import torch


def _split_size(count: int, group_size: int) -> int:
    """Keep group boundaries aligned even when the last group is incomplete."""
    if count <= group_size:
        return count // 2
    groups = (count + group_size - 1) // group_size
    return (groups // 2) * group_size


def _feature_order(features: np.ndarray, priority: np.ndarray, group_size: int) -> np.ndarray:
    """Balanced recursive principal-direction splits, with linear-time medians."""
    iterations = 4

    def visit(ids: np.ndarray) -> np.ndarray:
        count = len(ids)
        if count <= 1:
            return ids
        left_size = _split_size(count, group_size)
        local = features[ids]
        centered = local - local.mean(axis=0)
        energy = np.einsum("ij,ij->j", centered, centered)
        anchor = int(np.argmax(energy))
        if energy[anchor] <= 1e-20:
            scores = np.zeros(count, dtype=np.float64)
        else:
            # A coordinate with maximal variance is a deterministic, nonzero
            # starting direction for a few covariance power iterations.
            direction = np.zeros(features.shape[1], dtype=np.float64)
            direction[anchor] = 1.0
            for _ in range(iterations):
                next_direction = centered.T @ (centered @ direction)
                length = np.linalg.norm(next_direction)
                if length <= 1e-20:
                    break
                direction = next_direction / length
            if direction[anchor] < 0:
                direction = -direction
            scores = centered @ direction

        threshold = np.partition(scores, left_size - 1)[left_size - 1]
        before = scores < threshold
        left = ids[before]
        tied = ids[scores == threshold]
        if tied.size:
            tied = tied[np.argsort(priority[tied], kind="stable")]
        needed = left_size - len(left)
        left = np.concatenate((left, tied[:needed]))
        right = np.concatenate((ids[scores > threshold], tied[needed:]))
        return np.concatenate((visit(left), visit(right)))

    return visit(np.arange(features.shape[0], dtype=np.int64))


def compile_feature_wire_order(
    x: torch.Tensor,
    node_to_wire: torch.Tensor,
    padded: int,
    group_size: int = 32,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(order, inverse)`` for one feature-compiled circuit block.

    ``order[new_wire]`` is the old wire and ``inverse[old_wire]`` is the new
    wire. The real nodes occupy consecutive groups of ``group_size`` wires;
    any unused wires follow the real nodes. The final group may contain both
    real and dummy wires. Inputs are copied to CPU and outputs are CPU int64
    tensors so compilation does not depend on the training device.

    The features are expected to have been standardized by the caller. Each
    node vector is L2-normalized here, so the split geometry emphasizes
    feature direction. Ties are resolved by a fixed permutation from ``seed``.
    """
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("x must have shape [N, D] with D >= 1")
    if node_to_wire.ndim != 1 or node_to_wire.numel() != x.shape[0]:
        raise ValueError("node_to_wire must have one entry per node")
    if isinstance(padded, bool) or not isinstance(padded, int) or padded < 1 or padded & (padded - 1):
        raise ValueError("padded must be a positive power of two")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1 or group_size & (group_size - 1):
        raise ValueError("group_size must be a positive power of two")
    if x.shape[0] < 1 or x.shape[0] > padded:
        raise ValueError("the number of nodes must lie in [1, padded]")
    if node_to_wire.dtype != torch.long:
        raise TypeError("node_to_wire must have dtype int64")

    mapping = node_to_wire.detach().cpu().numpy()
    if (mapping < 0).any() or (mapping >= padded).any() or len(np.unique(mapping)) != len(mapping):
        raise ValueError("node_to_wire must inject nodes into padded wires")
    # The conversion may otherwise alias an already-CPU float64 input.
    features = x.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
    if not np.isfinite(features).all():
        raise ValueError("x must contain only finite features")
    lengths = np.linalg.norm(features, axis=1, keepdims=True)
    np.divide(features, lengths, out=features, where=lengths > 0)

    rng = np.random.default_rng(seed)
    priority = np.empty(len(mapping), dtype=np.int64)
    priority[rng.permutation(len(mapping))] = np.arange(len(mapping))
    nodes = _feature_order(features, priority, group_size)
    real_old_wires = mapping[nodes]

    occupied = np.zeros(padded, dtype=bool)
    occupied[mapping] = True
    dummy_old_wires = np.flatnonzero(~occupied)
    order_np = np.concatenate((real_old_wires, dummy_old_wires))
    inverse_np = np.empty(padded, dtype=np.int64)
    inverse_np[order_np] = np.arange(padded, dtype=np.int64)
    return torch.from_numpy(order_np.copy()), torch.from_numpy(inverse_np)


__all__ = ["compile_feature_wire_order"]
