"""Paired full-depth GCM training on the official ICLR 2026 LRIM benchmark.

This uses the published LRIM loader and its fixed graph-index split. The
reported target is *node* regression; the primary metric is log10 of MSE over
all nodes in a split. Checkpoints are selected on validation logMSE, and the
test split is evaluated exactly once per selected checkpoint.

Example (the data must already be present locally)::

    python scripts/run_lrim.py --dataset lrim_16_0.6_10k --device cuda \
        --seeds 0 1 2 --batch-size 32 --epochs 200 --patience 30 --resume

The script always runs the matched exchange_scale=0 and 0.25 pair. Both arms
have the same initialization, data order, optimizer, and full log2 node depth.
No subset or shortened-depth result can be produced by this runner.

``--layout torus_gray`` is a separate coordinate-order oracle ablation. It
uses the published row-major lattice order to map each coordinate through a
binary-reflected Gray code, making torus edges direct butterfly partners.
It is not a permutation-invariant, graph-derived compiler.

``--layout torus_feature_gray`` reconstructs the torus from adjacency and
selects its origin and orientation from observed input spins. It uses no
published coordinates or target labels; ambiguous feature symmetries fail
explicitly rather than falling back to node IDs.

``--spin-flip-gauge`` optionally makes the torus feature compiler choose the
lexicographically smaller of the compiled wire states for x and -x. The
chosen sign is applied to observed inputs in every split; targets are kept.

``--pair-distance-angle`` is optional for either torus Gray layout. One
offline breadth-first traversal table from graph 0's fixed topology assigns
each regular butterfly pair its shortest graph-hop distance. Both training
arms reuse its inverse distances to scale the existing original-J angles;
the switch architecture and trainable parameter set are unchanged.

``--global-angle-heads`` can retain the unscaled J angle in leading heads
while the other heads use the same compiled inverse graph distances.
``--angle-distance-powers`` instead assigns a fixed inverse-distance exponent
to each existing J head, allowing several geometric ranges without new heads,
parameters, pairs, or readout operations.

``--interstage-head-shuffle`` permutes complete real/imaginary channel pairs
across existing heads between J stages, then reverses the accumulated wiring
at circuit exit. It adds no parameters; the zero-exchange arm is exactly the
same function as before and the paired J arm alone tests the new transport.

``--lr-schedule cosine`` is an optional five-epoch linear warmup followed by
cosine decay across the configured epoch budget. The default remains a fixed
learning rate.

``--magnetization-control`` is a diagnostic control. It supplies each node
with its spin and spin times its own graph's mean spin, in both paired arms.
It uses input features only and leaves the regression target unchanged.

``--spin-field-readout`` decodes the unchanged full-node GCM output as a field
and multiplies it by the observed node spin to predict energy. Both arms use
the same task decoder; it uses no exact interaction kernel or target values.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset as TorchDataset


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gcm.multigraph import (  # noqa: E402
    MultiGraphGCM, compile_headwise_pair_angle_scales,
    compile_pair_distance_angle_scales,
)
from gcm.torus_gray import TorusFeatureGrayCompiler  # noqa: E402

SCALES = (0.0, 0.25)
SPLIT_LENGTHS = {"train": 8000, "val": 1000, "test": 1000}


def _official_loader_class():
    source = REPO / "external" / "LRIM" / "example-setup" / "lrim_loader.py"
    spec = importlib.util.spec_from_file_location("official_lrim_loader", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import the official LRIM loader from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LRIM


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "shuffle": generator.get_state(),
    }


def _restore_rng(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["shuffle"])


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _append_jsonl(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _parameter_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in model.named_parameters():
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _cosine_warmup_scheduler(optimizer: torch.optim.Optimizer,
                             epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Match LRIM's five-step warmup and half-cosine epoch schedule."""
    def factor(step: int) -> float:
        if step < 5:
            return max(1e-6, step / 5.)
        progress = (step - 5) / max(1, epochs - 5)
        return max(0., .5 * (1. + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _load_dataset(data_dir: Path, name: str):
    match = re.fullmatch(r"lrim_(16|32)_[0-9.]+_10k", name)
    if match is None:
        raise ValueError("This runner supports official 16x16 or 32x32 LRIM 10k datasets")
    root = (data_dir / name).resolve()
    raw = root / "raw" / f"{name}.pt"
    if not raw.is_file():
        raise FileNotFoundError(f"Official raw LRIM data is missing: {raw}")
    dataset = _official_loader_class()(root=str(root), name=name, hf_repo=None)
    if len(dataset) != 10000:
        raise ValueError(f"Expected 10,000 official graphs, found {len(dataset)}")
    splits = dataset.get_idx_split()
    start = 0
    for split, length in SPLIT_LENGTHS.items():
        expected = torch.arange(start, start + length, dtype=torch.long)
        if not torch.equal(splits[split].cpu(), expected):
            raise ValueError(f"Official {split} graph indices differ from the fixed protocol")
        start += length
    nodes = int(match.group(1)) ** 2
    for index in (0, 7999, 8000, 8999, 9000, 9999):
        graph = dataset[index]
        if graph.num_nodes != nodes or graph.x.shape != (nodes, 1) \
                or graph.y.shape != (nodes, 1):
            raise ValueError(f"Unexpected graph, feature, or target shape at index {index}")
        if not torch.isfinite(graph.x).all() or not torch.isfinite(graph.y).all():
            raise ValueError(f"Nonfinite features or targets at graph {index}")
    return dataset, splits, nodes, root


def _make_model(args: argparse.Namespace, nodes: int, seed: int,
                scale: float) -> MultiGraphGCM:
    stages = int(math.log2(nodes))
    if 1 << stages != nodes:
        raise ValueError("LRIM graph node count must be a power of two")
    return MultiGraphGCM(
        input_dim=2 if getattr(args, "magnetization_control", False) else 1,
        edge_dim=None, output_dim=1,
        dim=args.dim, heads=args.heads, blocks=args.blocks,
        max_stages=stages, mode="original", exchange_scale=scale,
        dropout=args.dropout, angle=args.angle, stage_ff=False,
        graph_embed=True, graph_refresh=True,
        # Supplied torus Gray-code mappings supersede this internal layout.
        layout="input_order" if args.layout in ("torus_gray", "torus_feature_gray")
        else args.layout,
        seed=seed,
        tie_pair_gate=args.tie_pair_gate,
        prior_only_angle_scale=getattr(args, "prior_only_angle_scale", False),
        reverse_odd_blocks=getattr(args, "reverse_odd_blocks", False),
        interstage_head_shuffle=getattr(args, "interstage_head_shuffle", False),
    )


def _torus_gray_mapping(nodes: int) -> torch.Tensor:
    """Map published row-major torus coordinates to binary-reflected Gray wires."""
    side = math.isqrt(nodes)
    if side * side != nodes or side & (side - 1):
        raise ValueError("Torus Gray mapping requires a square power-of-two grid")
    ids = torch.arange(nodes, dtype=torch.long)
    row, col = ids // side, ids % side
    row_gray = row ^ (row >> 1)
    col_gray = col ^ (col >> 1)
    mapping = side * row_gray + col_gray
    if mapping.unique().numel() != nodes:
        raise RuntimeError("Torus Gray wire mapping is not a permutation")
    return mapping


def _verify_torus_gray_dataset(dataset, nodes: int) -> None:
    """Require the published fixed, row-major periodic lattice topology."""
    side = math.isqrt(nodes)
    reference = dataset[0].edge_index
    observed = set(map(tuple, reference.T.tolist()))
    expected = set()
    for row in range(side):
        for col in range(side):
            node = row * side + col
            expected.add((node, row * side + (col + 1) % side))
            expected.add((node, row * side + (col - 1) % side))
            expected.add((node, ((row + 1) % side) * side + col))
            expected.add((node, ((row - 1) % side) * side + col))
    if observed != expected or reference.shape[1] != len(expected):
        raise ValueError("torus_gray requires the official row-major periodic grid")
    _verify_fixed_edge_index(dataset, "torus_gray")


def _verify_fixed_edge_index(dataset, layout: str) -> None:
    reference = dataset[0].edge_index
    for index in range(1, len(dataset)):
        if not torch.equal(reference, dataset[index].edge_index):
            raise ValueError(f"{layout} requires fixed edge order; graph {index} differs")


def _mappings_for_batch(batch: Batch,
                        mapping_source: torch.Tensor | list[torch.Tensor] | None
                        ) -> list[torch.Tensor] | None:
    if mapping_source is None:
        return None
    if isinstance(mapping_source, torch.Tensor):
        return [mapping_source] * batch.num_graphs
    sample_ids = getattr(batch, "sample_id", None)
    if sample_ids is None or sample_ids.numel() != batch.num_graphs:
        raise ValueError("Per-graph mappings require one sample_id per graph")
    return [mapping_source[int(index)] for index in sample_ids.tolist()]


class _IndexedLRIM(TorchDataset):
    """Copy graphs for IDs or input-only diagnostic feature augmentation."""

    def __init__(self, dataset, indices: torch.Tensor,
                 attach_sample_ids: bool = True,
                 magnetization_control: bool = False,
                 spin_flip_flags: list[bool] | None = None):
        self.dataset = dataset
        self.indices = indices.cpu().tolist()
        self.attach_sample_ids = attach_sample_ids
        self.magnetization_control = magnetization_control
        self.spin_flip_flags = spin_flip_flags
        if spin_flip_flags is not None and len(spin_flip_flags) != len(dataset):
            raise ValueError("Spin-flip flags require one entry per graph")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        graph = self.dataset[index].clone()
        if self.spin_flip_flags is not None and self.spin_flip_flags[index]:
            graph.x = -graph.x
        if self.magnetization_control:
            if graph.x.ndim != 2 or graph.x.shape[1] != 1:
                raise ValueError("Magnetization control requires one input spin per node")
            spins = graph.x
            graph.x = torch.cat((spins, spins * spins.mean(dim=0, keepdim=True)),
                                dim=1)
        if self.attach_sample_ids:
            graph.sample_id = torch.tensor([index], dtype=torch.long)
        return graph


def _loader(dataset, indices: torch.Tensor, batch_size: int,
            shuffle: bool, generator: torch.Generator | None = None,
            attach_sample_ids: bool = False,
            magnetization_control: bool = False,
            spin_flip_flags: list[bool] | None = None) -> DataLoader:
    subset = (_IndexedLRIM(dataset, indices, attach_sample_ids,
                           magnetization_control, spin_flip_flags)
              if attach_sample_ids or magnetization_control
              or spin_flip_flags is not None else dataset[indices])
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator)


def _metrics(sum_squared: float, sum_absolute: float, sum_y: float,
             sum_y_squared: float, count: int) -> dict[str, float]:
    if count < 1:
        raise ValueError("No nodes were evaluated")
    mse = sum_squared / count
    denominator = sum_y_squared - (sum_y * sum_y / count)
    return {
        "logMSE": math.log10(max(mse, np.finfo(float).tiny)),
        "MSE": mse,
        "MAE": sum_absolute / count,
        "R2": 1.0 - sum_squared / denominator if denominator > 0 else float("nan"),
        "nodes": count,
    }


@torch.no_grad()
def _evaluate(model: MultiGraphGCM, loader: DataLoader,
              device: torch.device,
              mapping_source: torch.Tensor | list[torch.Tensor] | None = None,
              pair_angle_scales: torch.Tensor | None = None,
              spin_field_readout: bool = False,
              ) -> dict[str, float]:
    model.eval()
    squared = absolute = target_sum = target_squared = 0.0
    count = 0
    for batch in loader:
        mappings = _mappings_for_batch(batch, mapping_source)
        batch = batch.to(device)
        if pair_angle_scales is None:
            prediction = model.predict_nodes(batch, mappings=mappings).reshape(-1)
        else:
            prediction = model.predict_nodes(
                batch, mappings=mappings,
                pair_angle_scales=pair_angle_scales).reshape(-1)
        if spin_field_readout:
            prediction = prediction * batch.x[:, 0]
        target = batch.y.reshape(-1)
        if prediction.shape != target.shape:
            raise ValueError("Node prediction and target shapes do not match")
        delta = prediction.double() - target.double()
        squared += float(delta.square().sum().item())
        absolute += float(delta.abs().sum().item())
        target_sum += float(target.double().sum().item())
        target_squared += float(target.double().square().sum().item())
        count += target.numel()
    return _metrics(squared, absolute, target_sum, target_squared, count)


def _train_epoch(model: MultiGraphGCM, loader: DataLoader,
                 optimizer: torch.optim.Optimizer, device: torch.device,
                 clip_norm: float,
                 mapping_source: torch.Tensor | list[torch.Tensor] | None = None,
                 pair_angle_scales: torch.Tensor | None = None,
                 spin_field_readout: bool = False) -> float:
    model.train()
    squared = 0.0
    count = 0
    for batch in loader:
        mappings = _mappings_for_batch(batch, mapping_source)
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        if pair_angle_scales is None:
            prediction = model.predict_nodes(batch, mappings=mappings).reshape(-1)
        else:
            prediction = model.predict_nodes(
                batch, mappings=mappings,
                pair_angle_scales=pair_angle_scales).reshape(-1)
        if spin_field_readout:
            prediction = prediction * batch.x[:, 0]
        target = batch.y.reshape(-1)
        if prediction.shape != target.shape:
            raise ValueError("Node prediction and target shapes do not match")
        loss = F.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        squared += float(loss.detach().item()) * target.numel()
        count += target.numel()
    return math.log10(max(squared / count, np.finfo(float).tiny))


def _config(args: argparse.Namespace, name: str, nodes: int,
            root: Path, seed: int, scale: float,
            pair_angle_scales: torch.Tensor | None = None) -> dict[str, Any]:
    config = {
        "benchmark": "ICLR 2026 LRIM",
        "official_loader": str((REPO / "external/LRIM/example-setup/lrim_loader.py").resolve()),
        "dataset": name,
        "dataset_root": str(root),
        "split_graph_indices": {"train": [0, 7999], "val": [8000, 8999],
                                "test": [9000, 9999]},
        "seed": seed,
        "model": {
            "class": "MultiGraphGCM", "task": "node_regression",
            "input_dim": 2 if getattr(args, "magnetization_control", False) else 1,
            "output_dim": 1, "dim": args.dim,
            "heads": args.heads, "blocks": args.blocks,
            "max_stages": int(math.log2(nodes)), "mode": "original",
            "exchange_scale": scale, "angle": args.angle,
            "stage_ff": False, "graph_embed": True,
            "graph_refresh": True, "layout": args.layout,
            "dropout": args.dropout, "tie_pair_gate": args.tie_pair_gate,
        },
        "training": {
            "loss": "node_MSE", "validation_selection": "minimum_log10_node_MSE",
            "batch_size": args.batch_size, "optimizer": "AdamW",
            "lr": args.lr, "weight_decay": args.weight_decay,
            "grad_clip_norm": args.clip_norm, "epochs": args.epochs,
            "patience": args.patience, "amp": False, "num_workers": 0,
            "train_shuffle_seed": seed + 104729,
        },
        "torch_version": torch.__version__,
    }
    if getattr(args, "magnetization_control", False):
        config["diagnostic_control"] = {
            "name": "per_graph_input_magnetization",
            "node_features": ["s_i", "s_i * mean_j(s_j)"],
            "source": "input_spins_of_current_graph_only",
            "applied_to_both_exchange_arms": True,
            "regression_targets_unchanged": True,
        }
    if getattr(args, "spin_field_readout", False):
        config["model"]["output_rule"] = "observed_spin_times_predicted_node_field"
    if getattr(args, "reverse_odd_blocks", False):
        config["model"]["reverse_odd_blocks"] = True
    if getattr(args, "interstage_head_shuffle", False):
        config["model"]["interstage_head_shuffle"] = (
            "fixed_complete_real_imag_pair_transpose_across_heads_between_stages")
    if getattr(args, "lr_schedule", "constant") == "cosine":
        config["training"].update({
            "lr_schedule": "cosine_with_warmup",
            "lr_warmup_epochs": 5,
            "lr_schedule_total_epochs": args.epochs,
        })
    if args.layout == "torus_gray":
        config["model"]["mapping_source"] = "published_row_major_torus_coordinates"
        config["model"]["internal_layout_when_mapping_supplied"] = "input_order"
    elif args.layout == "torus_feature_gray":
        config["model"].update({
            "mapping_source": "observed_binary_spins_and_graph_adjacency",
            "mapping_compiler": "TorusFeatureGrayCompiler",
            "feature_anchor": "first_unique_color_from_permutation_invariant_1WL",
            "frame_selection": "unique_lexicographic_minimum_among_eight_spin_frames",
            "max_refinement": 8,
            "internal_layout_when_mapping_supplied": "input_order",
        })
        if getattr(args, "spin_flip_gauge", False):
            config["model"].update({
                "spin_flip_gauge": "lexicographic_minimum_of_two_compiled_wire_spin_states",
                "spin_flip_gauge_source": "observed_spins_and_graph_adjacency_only",
                "spin_flip_gauge_ambiguous_ties": "raise",
                "input_sign_applied_to_both_exchange_arms": True,
                "regression_targets_unchanged": True,
            })
    if args.pair_distance_angle:
        if pair_angle_scales is None:
            raise ValueError("pair-distance angle requires a precomputed distance table")
        config["model"].update({
            "pair_distance_angle": True,
            "pair_distance_rule": "inverse_shortest_graph_hop_distance",
            "pair_distance_source": "edge_index_and_torus_gray_mapping",
            "pair_angle_table_sha256": hashlib.sha256(
                pair_angle_scales.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
        })
        if getattr(args, "prior_only_angle_scale", False):
            config["model"]["prior_only_angle_scale"] = True
        global_heads = getattr(args, "global_angle_heads", 0)
        distance_powers = getattr(args, "angle_distance_powers", None)
        if distance_powers is not None:
            config["model"].update({
                "angle_distance_powers": list(distance_powers),
                "pair_distance_rule": "headwise_inverse_graph_distance_powers",
            })
        if global_heads:
            config["model"].update({
                "global_angle_heads": global_heads,
                "distance_angle_heads": args.heads - global_heads,
                "pair_distance_rule":
                    "leading_heads_unit_angle_else_inverse_shortest_graph_hop_distance",
            })
    return config


def _arm_dir(output: Path, dataset: str, seed: int, scale: float) -> Path:
    scale_tag = f"{scale:g}".replace(".", "p")
    return output / dataset / f"seed_{seed}" / f"scale_{scale_tag}"


def _run_arm(args: argparse.Namespace, dataset, splits: dict[str, torch.Tensor],
             nodes: int, root: Path, seed: int, scale: float,
             device: torch.device,
             mapping_table: list[torch.Tensor] | None = None,
             mapping_precompute: dict[str, Any] | None = None,
             pair_angle_scales: torch.Tensor | None = None,
             spin_flip_flags: list[bool] | None = None) -> dict[str, Any]:
    folder = _arm_dir(args.output, args.dataset, seed, scale)
    result_path = folder / "result.json"
    latest_path = folder / "latest.pt"
    best_path = folder / "best.pt"
    log_path = folder / "epochs.jsonl"
    config = _config(args, args.dataset, nodes, root, seed, scale,
                     pair_angle_scales)
    if result_path.exists():
        if not args.resume:
            raise FileExistsError(f"Result exists; pass --resume or use another output: {result_path}")
        completed = json.loads(result_path.read_text(encoding="utf-8"))
        if completed["config"] != config:
            raise ValueError(f"Existing result has a different configuration: {result_path}")
        print(f"Already complete: {result_path}", flush=True)
        return completed
    if latest_path.exists() and not args.resume:
        raise FileExistsError(f"Checkpoint exists; pass --resume or use another output: {latest_path}")

    _seed_everything(seed)
    model = _make_model(args, nodes, seed, scale).to(device)
    device_pair_angle_scales = (pair_angle_scales.to(device)
                                if pair_angle_scales is not None else None)
    initial_digest = _parameter_digest(model)
    parameter_count = sum(p.numel() for p in model.parameters())
    # MultiGraphGCM also carries a graph-pooling head that predict_nodes never
    # calls. At scale zero, the pair switches are also bypassed. Keep all of
    # these in the checkpoint so both arms have the same initialized model.
    active_parameter_count = sum(
        p.numel() for name, p in model.named_parameters()
        if not name.startswith(("head.", "node_readout."))
        and (scale != 0. or ".circuit.switches." not in name)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = (_cosine_warmup_scheduler(optimizer, args.epochs)
                 if getattr(args, "lr_schedule", "constant") == "cosine" else None)
    generator = torch.Generator().manual_seed(seed + 104729)
    indexed = args.layout == "torus_feature_gray"
    train_loader = _loader(dataset, splits["train"], args.batch_size, True,
                           generator, indexed, args.magnetization_control,
                           spin_flip_flags)
    val_loader = _loader(dataset, splits["val"], args.batch_size, False,
                         attach_sample_ids=indexed,
                         magnetization_control=args.magnetization_control,
                         spin_flip_flags=spin_flip_flags)
    test_loader = _loader(dataset, splits["test"], args.batch_size, False,
                          attach_sample_ids=indexed,
                          magnetization_control=args.magnetization_control,
                          spin_flip_flags=spin_flip_flags)
    mapping_source = (_torus_gray_mapping(nodes) if args.layout == "torus_gray"
                      else mapping_table)
    if indexed and mapping_table is None:
        raise ValueError("torus_feature_gray requires precomputed per-graph mappings")
    start_epoch = 1
    best_epoch = 0
    best_val = float("inf")
    no_improvement = 0
    elapsed_before = 0.0
    if latest_path.exists():
        # RNG and DataLoader generator states are CPU byte tensors. Loading the
        # whole payload to CUDA would make their restore calls fail.
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        if saved["config"] != config or saved["initial_parameter_digest"] != initial_digest:
            raise ValueError(f"Cannot resume incompatible checkpoint: {latest_path}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler is not None:
            if "scheduler" not in saved:
                raise ValueError(f"Cosine checkpoint has no scheduler state: {latest_path}")
            scheduler.load_state_dict(saved["scheduler"])
        _restore_rng(saved["rng"], generator)
        start_epoch = int(saved["epoch"]) + 1
        best_epoch = int(saved["best_epoch"])
        best_val = float(saved["best_val_logMSE"])
        no_improvement = int(saved["no_improvement"])
        elapsed_before = float(saved["elapsed_seconds"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    last_epoch = start_epoch - 1
    epoch_durations = []
    print(json.dumps({"dataset": args.dataset, "seed": seed, "scale": scale,
                      "parameters_total": parameter_count,
                      "parameters_active": active_parameter_count,
                      "stages": int(math.log2(nodes)),
                      "start_epoch": start_epoch, "device": str(device)}), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        epoch_started = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"] if scheduler is not None else None
        train_logmse = _train_epoch(model, train_loader, optimizer, device,
                                    args.clip_norm, mapping_source,
                                    device_pair_angle_scales,
                                    getattr(args, "spin_field_readout", False))
        val = _evaluate(model, val_loader, device, mapping_source,
                        device_pair_angle_scales,
                        getattr(args, "spin_field_readout", False))
        improved = val["logMSE"] < best_val
        if improved:
            best_val = val["logMSE"]
            best_epoch = epoch
            no_improvement = 0
            _atomic_torch_save({"config": config, "epoch": epoch,
                                "val": val, "model": _state_cpu(model)}, best_path)
        else:
            no_improvement += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if scheduler is not None:
            scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_started
        epoch_durations.append(epoch_seconds)
        elapsed = elapsed_before + time.perf_counter() - started
        row = {"epoch": epoch, "train_logMSE": train_logmse,
               "val": val, "best_epoch": best_epoch,
               "best_val_logMSE": best_val, "elapsed_seconds": elapsed,
               "epoch_seconds": epoch_seconds}
        if scheduler is not None:
            row["lr"] = current_lr
        _append_jsonl(row, log_path)
        latest = {
            "config": config, "initial_parameter_digest": initial_digest,
            "epoch": epoch, "best_epoch": best_epoch,
            "best_val_logMSE": best_val, "no_improvement": no_improvement,
            "elapsed_seconds": elapsed, "model": _state_cpu(model),
            "optimizer": optimizer.state_dict(), "rng": _rng_state(generator),
        }
        if scheduler is not None:
            latest["scheduler"] = scheduler.state_dict()
        _atomic_torch_save(latest, latest_path)
        progress = {"seed": seed, "scale": scale, "epoch": epoch,
                    "train_logMSE": train_logmse,
                    "val_logMSE": val["logMSE"],
                    "best_epoch": best_epoch}
        if scheduler is not None:
            progress["lr"] = current_lr
        print(json.dumps(progress), flush=True)
        if no_improvement >= args.patience:
            break

    if best_epoch == 0 or not best_path.exists():
        raise RuntimeError("No validation-selected checkpoint is available")
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    if selected["config"] != config:
        raise ValueError(f"Best checkpoint has a different configuration: {best_path}")
    model.load_state_dict(selected["model"])
    test = _evaluate(model, test_loader, device, mapping_source,
                     device_pair_angle_scales,
                     getattr(args, "spin_field_readout", False))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_gpu_bytes = torch.cuda.max_memory_allocated(device)
    else:
        peak_gpu_bytes = None
    total_elapsed = elapsed_before + time.perf_counter() - started
    result = {
        "config": config,
        "initial_parameter_digest": initial_digest,
        "parameter_count": parameter_count,
        "active_parameter_count": active_parameter_count,
        "selected_epoch": best_epoch,
        "epochs_completed": last_epoch,
        "val": selected["val"],
        "test": test,
        "wall_seconds": total_elapsed,
        "median_epoch_seconds_this_invocation": (
            statistics.median(epoch_durations) if epoch_durations else None),
        "mapping_precompute": mapping_precompute,
        "peak_gpu_memory_bytes": peak_gpu_bytes,
        "best_checkpoint": str(best_path.resolve()),
    }
    _atomic_json(result, result_path)
    print(json.dumps({"result": str(result_path), "selected_epoch": best_epoch,
                      "val": result["val"], "test": test,
                      "wall_seconds": total_elapsed,
                      "peak_gpu_memory_bytes": peak_gpu_bytes}), flush=True)
    return result


def _check_only(args: argparse.Namespace, dataset, nodes: int) -> None:
    """A CPU data/model consistency check; this does not train or score LRIM."""
    _seed_everything(args.seeds[0])
    originals = [dataset[0], dataset[1]]
    copies = [(g.x.clone(), g.edge_index.clone(), g.y.clone()) for g in originals]
    model = _make_model(args, nodes, args.seeds[0], .25).cpu().eval()
    if args.layout == "torus_feature_gray":
        compiler = TorusFeatureGrayCompiler(originals[0].edge_index, nodes)
        if getattr(args, "spin_flip_gauge", False):
            choices = [compiler.compile_spin_flip_gauge_with_radius(graph.x)
                       for graph in originals]
            mapping_table = [choice[0] for choice in choices]
            # The check loader only indexes the first two official graphs.
            flip_flags = [False] * len(dataset)
            flip_flags[:len(choices)] = [choice[2] for choice in choices]
        else:
            mapping_table = [compiler.compile(graph.x) for graph in originals]
            flip_flags = None
        batch = next(iter(_loader(dataset, torch.arange(2), 2, False,
                                  attach_sample_ids=True,
                                  magnetization_control=args.magnetization_control,
                                  spin_flip_flags=flip_flags)))
        mappings = _mappings_for_batch(batch, mapping_table)
    else:
        batch = (next(iter(_loader(dataset, torch.arange(2), 2, False,
                                   magnetization_control=True)))
                 if args.magnetization_control else Batch.from_data_list(originals))
        mapping = _torus_gray_mapping(nodes) if args.layout == "torus_gray" else None
        mappings = _mappings_for_batch(batch, mapping)
    pair_angle_scales = None
    if args.pair_distance_angle:
        distance_scales = compile_pair_distance_angle_scales(
            originals[0].edge_index, mappings[0])
        second = compile_pair_distance_angle_scales(
            originals[1].edge_index, mappings[1])
        assert torch.equal(distance_scales, second)
        pair_angle_scales = compile_headwise_pair_angle_scales(
            distance_scales, args.heads, args.global_angle_heads,
            getattr(args, "angle_distance_powers", None))
    with torch.no_grad():
        if pair_angle_scales is None:
            output = model.predict_nodes(batch, mappings=mappings)
        else:
            output = model.predict_nodes(
                batch, mappings=mappings,
                pair_angle_scales=pair_angle_scales)
        if getattr(args, "spin_field_readout", False):
            output = output * batch.x[:, :1]
    assert output.shape == (2 * nodes, 1) and torch.isfinite(output).all()
    assert batch.x.shape == (2 * nodes,
                             2 if args.magnetization_control else 1)
    for graph, (x, edges, y) in zip(originals, copies):
        assert torch.equal(graph.x, x) and torch.equal(graph.edge_index, edges)
        assert torch.equal(graph.y, y)
    check = {"check": "passed", "graphs": len(dataset),
             "nodes_per_graph": nodes, "stages": int(math.log2(nodes)),
             "batch_output_shape": list(output.shape),
             "official_arrays_unchanged": True,
             "note": "No training or benchmark metric was run"}
    if args.magnetization_control:
        check.update({"magnetization_control": True, "node_input_dim": 2})
    if getattr(args, "spin_field_readout", False):
        check["spin_field_readout"] = True
    if getattr(args, "spin_flip_gauge", False):
        check["spin_flip_gauge"] = True
        check["flipped_graphs"] = sum(flip_flags)
    if args.global_angle_heads:
        check["global_angle_heads"] = args.global_angle_heads
    if getattr(args, "angle_distance_powers", None) is not None:
        check["angle_distance_powers"] = args.angle_distance_powers
    if args.lr_schedule == "cosine":
        check["lr_schedule"] = "cosine_with_warmup"
    print(json.dumps(check), flush=True)


def _precompute_feature_mappings(
        dataset, nodes: int, spin_flip_gauge: bool = False
        ) -> tuple[list[torch.Tensor], list[bool] | None, dict[str, Any]]:
    compiler = TorusFeatureGrayCompiler(dataset[0].edge_index, nodes)
    started = time.perf_counter()
    mappings = []
    flip_flags: list[bool] | None = [] if spin_flip_gauge else None
    radii: Counter[int] = Counter()
    for index in range(len(dataset)):
        try:
            if spin_flip_gauge:
                mapping, radius, flipped = compiler.compile_spin_flip_gauge_with_radius(
                    dataset[index].x)
                assert flip_flags is not None
                flip_flags.append(flipped)
            else:
                mapping, radius = compiler.compile_with_radius(dataset[index].x)
        except ValueError as error:
            raise ValueError(f"Torus feature compilation failed at graph {index}") from error
        mappings.append(mapping)
        radii[radius] += 1
    summary = {
        "graphs": len(mappings),
        "seconds": time.perf_counter() - started,
        "feature_anchor_radius_counts": {str(key): value
                                         for key, value in sorted(radii.items())},
        "uses_targets": False,
        "uses_published_coordinates": False,
    }
    if flip_flags is not None:
        summary.update({"spin_flip_gauge": True,
                        "flipped_graphs": sum(flip_flags),
                        "regression_targets_unchanged": True})
    print(json.dumps({"mapping_precompute": summary}), flush=True)
    return mappings, flip_flags, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="lrim_16_0.6_10k")
    parser.add_argument("--data-dir", type=Path, default=REPO / "data/lrim")
    parser.add_argument("--output", type=Path, default=REPO / "results/lrim_gcm")
    parser.add_argument("--seeds", "--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--angle", type=float, default=.2)
    parser.add_argument("--tie-pair-gate", action="store_true")
    parser.add_argument("--reverse-odd-blocks", action="store_true",
                        help="Run the regular J stages high-to-low in odd-numbered blocks")
    parser.add_argument("--interstage-head-shuffle", action="store_true",
                        help="Permute complete real/imaginary channel pairs across heads between J stages, in both paired arms")
    parser.add_argument("--pair-distance-angle", action="store_true",
                        help="Scale each original J angle by inverse butterfly-partner graph distance")
    parser.add_argument("--prior-only-angle-scale", action="store_true",
                        help="Apply compiled distance scaling to the base J angle, leaving its learned content correction unscaled")
    parser.add_argument("--global-angle-heads", type=int, default=0,
                        help="With --pair-distance-angle, keep this many leading J heads unscaled")
    parser.add_argument("--angle-distance-powers", type=float, nargs="+",
                        default=None,
                        help="Optional fixed exponent per J head: angle divided by graph distance to this power")
    parser.add_argument("--magnetization-control", action="store_true",
                        help="Diagnostic: give both arms [spin, spin * mean spin of this graph]")
    parser.add_argument("--spin-field-readout", action="store_true",
                        help="Decode predicted field by multiplying observed node spin")
    parser.add_argument("--spin-flip-gauge", action="store_true",
                        help="Canonicalize each feature-Gray torus under global spin inversion")
    parser.add_argument("--layout", choices=["graph_components", "graph", "input_order",
                                              "torus_gray", "torus_feature_gray"],
                        default="graph_components")
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"],
                        default="constant")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-norm", type=float, default=1.)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.patience < 1 or args.blocks < 1 \
            or args.threads < 1 or args.lr <= 0 or args.clip_norm <= 0:
        parser.error("Batch size, epochs, patience, blocks, threads, lr, and clip norm must be positive")
    if args.dim < 1 or args.heads < 1 or args.dim % (2 * args.heads):
        parser.error("dim must be divisible by twice heads")
    if not 0 <= args.dropout < 1 or not math.isfinite(args.angle):
        parser.error("dropout must be in [0,1) and angle must be finite")
    if args.pair_distance_angle and args.layout not in (
            "torus_gray", "torus_feature_gray"):
        parser.error("--pair-distance-angle requires torus_gray or torus_feature_gray")
    if args.spin_flip_gauge and args.layout != "torus_feature_gray":
        parser.error("--spin-flip-gauge requires --layout torus_feature_gray")
    if args.prior_only_angle_scale and not args.pair_distance_angle:
        parser.error("--prior-only-angle-scale requires --pair-distance-angle")
    if not 0 <= args.global_angle_heads <= args.heads:
        parser.error("--global-angle-heads must be between zero and --heads")
    if args.global_angle_heads and not args.pair_distance_angle:
        parser.error("--global-angle-heads requires --pair-distance-angle")
    if args.angle_distance_powers is not None:
        if not args.pair_distance_angle or args.global_angle_heads:
            parser.error("--angle-distance-powers requires --pair-distance-angle and no --global-angle-heads")
        if len(args.angle_distance_powers) != args.heads or any(
                not math.isfinite(power) or power < 0
                for power in args.angle_distance_powers):
            parser.error("--angle-distance-powers needs one finite nonnegative value per head")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    dataset, splits, nodes, root = _load_dataset(args.data_dir, args.dataset)
    if args.layout == "torus_gray":
        _verify_torus_gray_dataset(dataset, nodes)
    elif args.layout == "torus_feature_gray":
        _verify_fixed_edge_index(dataset, args.layout)
    if args.check_only:
        _check_only(args, dataset, nodes)
        return
    mapping_table = None
    mapping_precompute = None
    spin_flip_flags = None
    if args.layout == "torus_feature_gray":
        mapping_table, spin_flip_flags, mapping_precompute = _precompute_feature_mappings(
            dataset, nodes, args.spin_flip_gauge)
    pair_angle_scales = None
    if args.pair_distance_angle:
        # LRIM has one fixed torus topology. Its feature-canonical charts
        # differ only by automorphisms, so pair distances are the same for
        # every graph and one offline table serves both paired arms.
        representative_mapping = (_torus_gray_mapping(nodes)
                                  if args.layout == "torus_gray"
                                  else mapping_table[0])
        distance_scales = compile_pair_distance_angle_scales(
            dataset[0].edge_index, representative_mapping)
        pair_angle_scales = compile_headwise_pair_angle_scales(
            distance_scales, args.heads, args.global_angle_heads,
            args.angle_distance_powers)
        distances = distance_scales.reciprocal().round().long()
        distance_summary = {
            "pair_distance_angle": True,
            "stage_hop_distance_counts": [
                {str(int(distance)): int(count) for distance, count in zip(*torch.unique(
                    distances[level], return_counts=True))}
                for level in range(distances.shape[0])
            ],
        }
        if args.global_angle_heads:
            distance_summary["global_angle_heads"] = args.global_angle_heads
        if args.angle_distance_powers is not None:
            distance_summary["angle_distance_powers"] = args.angle_distance_powers
        print(json.dumps(distance_summary), flush=True)
    device = torch.device(args.device)
    for seed in args.seeds:
        results = [_run_arm(args, dataset, splits, nodes, root, seed, scale,
                            device, mapping_table, mapping_precompute,
                            pair_angle_scales, spin_flip_flags)
                   for scale in SCALES]
        if results[0]["initial_parameter_digest"] != results[1]["initial_parameter_digest"]:
            raise RuntimeError("The two exchange arms did not share the same initial parameters")
        summary = {
            "dataset": args.dataset, "seed": seed,
            "initial_parameter_digest": results[0]["initial_parameter_digest"],
            "zero": {"val": results[0]["val"], "test": results[0]["test"],
                     "selected_epoch": results[0]["selected_epoch"]},
            "exchange_0p25": {"val": results[1]["val"], "test": results[1]["test"],
                              "selected_epoch": results[1]["selected_epoch"]},
            "delta_logMSE_exchange_minus_zero": {
                "val": results[1]["val"]["logMSE"] - results[0]["val"]["logMSE"],
                "test": results[1]["test"]["logMSE"] - results[0]["test"]["logMSE"],
            },
        }
        if args.magnetization_control:
            summary["magnetization_control"] = True
        if args.global_angle_heads:
            summary["global_angle_heads"] = args.global_angle_heads
        summary_path = args.output / args.dataset / f"seed_{seed}" / "paired.json"
        if mapping_precompute is not None:
            summary["mapping_precompute"] = mapping_precompute
        _atomic_json(summary, summary_path)
        print(json.dumps({"paired_result": str(summary_path),
                          "delta_logMSE_exchange_minus_zero":
                          summary["delta_logMSE_exchange_minus_zero"]}), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
