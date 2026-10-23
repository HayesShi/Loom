"""CPU probe of pair-channel diversity in the frozen LRIM original-J circuit.

Uses learned base angles from an existing no-shuffle checkpoint but sets all
content gates to zero. It trains nothing and reads no labels. The reported
complex spatial kernels describe the linear J carrier only, not trained GCM
predictions or a bound on achievable error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from gcm.multigraph import (compile_headwise_pair_angle_scales,
                            compile_pair_distance_angle_scales,
                            _VariableSizeCircuit)


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (REPO / "results/lrim_gcm/"
                      "feature_gray16hard_distance_nodrop_b256_e100/"
                      "lrim_16_0.6_10k/seed_0/scale_0p25/best.pt")


def _torus_edges(side: int) -> torch.Tensor:
    nodes = side * side
    source = torch.arange(nodes)
    right = side * (source // side) + (source + 1) % side
    down = (source + side) % nodes
    return torch.stack((torch.cat((source, right, source, down)),
                        torch.cat((right, source, down, source))))


def _torus_gray_mapping(side: int) -> torch.Tensor:
    nodes = side * side
    ids = torch.arange(nodes)
    row, col = ids // side, ids % side
    return side * (row ^ (row >> 1)) + (col ^ (col >> 1))


def _frozen_kernels(base_angles: np.ndarray, angle_scales: np.ndarray,
                    pair_order: np.ndarray, shuffled: bool,
                    reverse: bool) -> tuple[np.ndarray, list[list[int]]]:
    stages, heads = base_angles.shape
    nodes = angle_scales.shape[1] * 2
    channel_pairs = len(pair_order)
    pairs_per_head = channel_pairs // heads
    operators = np.broadcast_to(np.eye(nodes, dtype=np.complex128),
                                (channel_pairs, nodes, nodes)).copy()
    original_pair = np.arange(channel_pairs)
    trajectories = [[] for _ in range(channel_pairs)]
    levels = range(stages - 1, -1, -1) if reverse else range(stages)
    for stage_index, level in enumerate(levels):
        head_at_slot = np.arange(channel_pairs) // pairs_per_head
        for slot, original in enumerate(original_pair):
            trajectories[int(original)].append(int(head_at_slot[slot]))
        wires = np.arange(nodes)
        left = wires[(wires & (1 << level)) == 0]
        right = left ^ (1 << level)
        scales = (angle_scales[level, :, None] if angle_scales.ndim == 2
                  else angle_scales[level])
        if scales.shape[-1] == 1:
            theta = base_angles[level, head_at_slot, None] * scales[:, 0]
        else:
            theta = base_angles[level, head_at_slot, None] * (
                scales[:, head_at_slot].T)
        cosine = np.cos(theta)[:, :, None]
        sine = np.sin(theta)[:, :, None]
        old_left, old_right = operators[:, left, :].copy(), operators[:, right, :].copy()
        operators[:, left, :] = cosine * old_left + 1j * sine * old_right
        operators[:, right, :] = cosine * old_right + 1j * sine * old_left
        if shuffled and stage_index + 1 < stages:
            operators = operators[pair_order]
            original_pair = original_pair[pair_order]
    aligned = np.empty_like(operators)
    aligned[original_pair] = operators
    return aligned, trajectories


def _kernel_summary(operators: np.ndarray,
                    trajectories: list[list[int]]) -> dict[str, object]:
    count, nodes, _ = operators.shape
    representatives: list[int] = []
    for index in range(count):
        if not any(np.max(np.abs(operators[index] - operators[other])) < 1e-10
                   for other in representatives):
            representatives.append(index)
    flat = operators.reshape(count, -1) / math.sqrt(nodes)
    gram = flat @ flat.conj().T
    distances = np.sqrt(np.maximum(0.,
                                   gram.diagonal().real[:, None]
                                   + gram.diagonal().real[None, :]
                                   - 2 * gram.real))
    upper = distances[np.triu_indices(count, k=1)]
    return {
        "pair_channels": count,
        "distinct_head_trajectories": len({tuple(path) for path in trajectories}),
        "distinct_spatial_kernels_at_max_abs_tolerance_1e-10": len(representatives),
        "duplicate_channel_pairs_at_tolerance_1e-10": int(sum(
            np.max(np.abs(operators[left] - operators[right])) < 1e-10
            for left in range(count) for right in range(left + 1, count))),
        "relative_frobenius_distance_between_channel_kernels": {
            "p10": float(np.quantile(upper, .1)),
            "median": float(np.median(upper)),
            "p90": float(np.quantile(upper, .9)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=(
        REPO / "results/diagnostics/lrim_interstage_head_shuffle_probe_20260926.json"))
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]["model"]
    if config["mode"] != "original" or not config.get("pair_distance_angle"):
        raise ValueError("Probe requires an original-J LRIM distance-angle checkpoint")
    dim, heads = int(config["dim"]), int(config["heads"])
    stages = int(config["max_stages"])
    nodes = 1 << stages
    side = math.isqrt(nodes)
    if side * side != nodes or dim % (2 * heads):
        raise ValueError("Expected a square power-of-two torus and even head width")
    distance_scales = compile_pair_distance_angle_scales(
        _torus_edges(side), _torus_gray_mapping(side))
    angle_table = compile_headwise_pair_angle_scales(
        distance_scales, heads, int(config.get("global_angle_heads", 0)),
        config.get("angle_distance_powers"))
    table_hash = hashlib.sha256(
        angle_table.contiguous().numpy().tobytes()).hexdigest()
    if table_hash != config["pair_angle_table_sha256"]:
        raise ValueError("Reconstructed torus angle table differs from checkpoint")
    angle_scales = angle_table.numpy().astype(np.float64)
    reference = _VariableSizeCircuit(
        dim, heads, stages, "original", float(config["angle"]), False,
        interstage_head_shuffle=True)
    pair_order = (reference.interstage_shuffle[::2] // 2).numpy()
    blocks = {}
    for block in range(int(config["blocks"])):
        base = np.stack([
            checkpoint["model"][f"blocks.{block}.circuit.switches.{level}.base"].numpy()
            for level in range(stages)
        ]).astype(np.float64)
        reverse = bool(config.get("reverse_odd_blocks")) and block % 2 == 1
        cases = {}
        for label, shuffled in (("fixed_heads", False), ("interstage_pair_shuffle", True)):
            operators, trajectories = _frozen_kernels(
                base, angle_scales, pair_order, shuffled, reverse)
            cases[label] = _kernel_summary(operators, trajectories)
        blocks[str(block)] = {
            "stage_order": "descending" if reverse else "ascending",
            "learned_base_angle_head_spread_by_stage": (
                base.max(axis=1) - base.min(axis=1)).tolist(),
            "cases": cases,
        }
    # At literal initialization all heads share angle .2 and the same 1/d
    # table. Pair wiring alone cannot then diversify the frozen operator.
    initial = np.full((stages, heads), float(config["angle"]))
    init_cases = {}
    for label, shuffled in (("fixed_heads", False), ("interstage_pair_shuffle", True)):
        operators, trajectories = _frozen_kernels(
            initial, angle_scales, pair_order, shuffled, False)
        init_cases[label] = _kernel_summary(operators, trajectories)
    report = {
        "probe": "frozen_base_angle_only_original_J_channel_spatial_kernels",
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_labels_read": False,
        "training_run": False,
        "content_gate": "fixed_to_zero_for_linear_operator_probe",
        "same_distance_angle_table_for_both_cases": True,
        "verified_checkpoint_pair_angle_table_sha256": table_hash,
        "channel_pair_permutation": pair_order.tolist(),
        "exit_permutation": "inverse_of_composite_interstage_permutation",
        "initial_equal_head_angle_control": init_cases,
        "learned_base_angle_checkpoints": blocks,
        "interpretation_limit": (
            "Kernel diversity is a structural diagnostic only; this probe does not "
            "measure model accuracy or predict training gains."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "initial_distinct": {
                          key: value["distinct_spatial_kernels_at_max_abs_tolerance_1e-10"]
                          for key, value in init_cases.items()},
                      "blocks": {block: {
                          key: value["distinct_spatial_kernels_at_max_abs_tolerance_1e-10"]
                          for key, value in content["cases"].items()}
                          for block, content in blocks.items()}}))


if __name__ == "__main__":
    main()
