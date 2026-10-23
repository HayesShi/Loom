"""Matched zero/J experiment on the released ICLR 2026 SNI Laplace task.

Train on all 20,000 simple-domain records, select checkpoints using the separate
2,500-record simple-domain file, then evaluate the untouched A/B/C files. A
record contains only solution coordinates and values plus Dirichlet boundary
coordinates and values. Coordinates and boundary values are the sole features;
solution values never enter graph construction or the circuit compiler.

The baseline and J arms share the graph, features, initialization, optimizer,
schedule, sample order, exact boundary projection, local path, and all-node
readout. Only the multiplier of the original J pair exchange differs (0/.25).
The repeated original pair switch shares one rule across the 8 train and up to
11 evaluation levels, avoiding untrained high-level switches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import random
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gcm.paths import data_path
from gcm.architecture import compile_interfaces  # noqa: E402
from gcm.multigraph import MultiGraphGCM  # noqa: E402
from audit_sni_geometry_graph import geometry_graph  # noqa: E402

FILES = {
    "train": "laplace2d_simple_20000_train.pkl",
    "val": "laplace2d_simple_2500_test.pkl",
    "A": "laplace2d_schwarz_100_test.pkl",
    "B": "laplace2d_holes_100_test.pkl",
    "C": "laplace2d_bosch_100_test.pkl",
}
COUNTS = {"train": 20_000, "val": 2_500, "A": 100, "B": 100, "C": 100}
SCALES = (0.0, 0.25)
OFFICIAL_REFERENCE = {
    "A": {"direct_GNOT_percent": 22, "SNI_GNOT_percent": 2.2},
    "B": {"direct_GNOT_percent": 22, "SNI_GNOT_percent": 2.1},
    "C": {"direct_GNOT_percent": 28, "SNI_GNOT_percent": 2.1},
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(payload, temp)
    os.replace(temp, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


def _state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _parameter_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boundary_ids(xy: np.ndarray, bc: np.ndarray) -> np.ndarray:
    if bc.ndim != 2 or bc.shape[1] != 4 or not np.all(bc[:, 3] == 0):
        raise ValueError("Expected [x,y,u_D,type=0] Dirichlet boundary rows")
    dist, ids = cKDTree(xy).query(bc[:, :2])
    if float(np.max(dist)) > 1e-7 or len(np.unique(ids)) != len(ids):
        raise ValueError("Boundary coordinates do not match unique solution nodes")
    return ids.astype(np.int64)


def _normalized_xy(xy: np.ndarray) -> np.ndarray:
    minimum, maximum = xy.min(axis=0), xy.max(axis=0)
    span = float(np.max(maximum - minimum))
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Degenerate geometry")
    return (xy - .5 * (minimum + maximum)) / span


def _spatial_morton_mapping(xy_normalized: np.ndarray) -> torch.Tensor:
    """Assign spatial Morton rank evenly over padded wires, at every size."""
    n = len(xy_normalized)
    padded = 1 << (n - 1).bit_length()
    code = _spatial_morton_codes(xy_normalized)
    order = np.lexsort((xy_normalized[:, 1], xy_normalized[:, 0], code))
    wires = ((2 * np.arange(n, dtype=np.int64) + 1) * padded) // (2 * n)
    mapping = np.empty(n, dtype=np.int64)
    mapping[order] = wires
    if len(np.unique(mapping)) != n:
        raise RuntimeError("Spatial compiler produced duplicate wires")
    return torch.from_numpy(mapping)


def _spatial_morton_codes(xy_normalized: np.ndarray) -> np.ndarray:
    quantized = np.floor(np.clip(xy_normalized + .5, 0, 1) * 65535).astype(np.uint64)
    code = np.zeros(len(xy_normalized), dtype=np.uint64)
    for bit in range(16):
        code |= ((quantized[:, 0] >> bit) & 1) << (2 * bit)
        code |= ((quantized[:, 1] >> bit) & 1) << (2 * bit + 1)
    return code


def _spatial_morton_anchor_mapping(xy_normalized: np.ndarray) -> torch.Tensor:
    """Project fixed Morton coordinate addresses to unique padded wires.

    For sorted nodes, wire[i] = i + offset[i]. The offset is an integer
    nondecreasing sequence, so a bounded isotonic fit keeps wires distinct
    while moving them toward their absolute coordinate addresses.
    """
    n = len(xy_normalized)
    padded = 1 << (n - 1).bit_length()
    code = _spatial_morton_codes(xy_normalized)
    order = np.lexsort((xy_normalized[:, 1], xy_normalized[:, 0], code))
    targets = (code[order].astype(np.float64) + .5) * (padded / 2**32)
    desired_offset = targets - np.arange(n, dtype=np.float64)
    # Pooled-adjacent-violators fit: each block stores [start, end, mean, size].
    blocks: list[list[float | int]] = []
    for index, value in enumerate(desired_offset):
        blocks.append([index, index, float(value), 1])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
            right = blocks.pop()
            left = blocks[-1]
            size = int(left[3]) + int(right[3])
            left[1] = right[1]
            left[2] = (float(left[2]) * int(left[3])
                       + float(right[2]) * int(right[3])) / size
            left[3] = size
    offsets = np.empty(n, dtype=np.int64)
    for start, end, mean, _ in blocks:
        offsets[int(start):int(end) + 1] = int(round(np.clip(mean, 0, padded - n)))
    wires = np.arange(n, dtype=np.int64) + offsets
    mapping = np.empty(n, dtype=np.int64)
    mapping[order] = wires
    if (wires[0] < 0 or wires[-1] >= padded
            or np.any(wires[1:] <= wires[:-1])):
        raise RuntimeError("Anchored spatial compiler produced invalid wires")
    return torch.from_numpy(mapping)


def _graph_hops(n: int, edges: np.ndarray, sources: np.ndarray) -> torch.Tensor:
    adjacent: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adjacent[int(a)].append(int(b))
        adjacent[int(b)].append(int(a))
    hops = np.full(n, -1, dtype=np.int16)
    hops[sources] = 0
    queue = deque(map(int, sources))
    while queue:
        src = queue.popleft()
        for dst in adjacent[src]:
            if hops[dst] < 0:
                hops[dst] = hops[src] + 1
                queue.append(dst)
    if np.any(hops < 0):
        raise ValueError("Coordinate graph leaves nodes unreachable from boundary")
    return torch.from_numpy(hops.astype(np.int64))


class SniDataset(Dataset):
    """Released records with cached, label-free geometry and circuit layouts."""

    def __init__(self, path: Path, split: str, ratio: float, layout: str):
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            self.records = pickle.load(handle)
        if len(self.records) != COUNTS[split]:
            raise ValueError(f"{split}: expected {COUNTS[split]} records, got {len(self.records)}")
        self.path = path
        self.split = split
        self.ratio = ratio
        self.layout = layout
        self._geometry_cache: dict[bytes, tuple[torch.Tensor, torch.Tensor,
                                                torch.Tensor, torch.Tensor]] = {}
        self._graph_cache: list[Data | None] = [None] * len(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def prepare_all(self) -> dict[str, Any]:
        """Build every graph once before either arm's timed training begins."""
        started = time.perf_counter()
        for index in range(len(self)):
            self[index]
        return {"records": len(self),
                "unique_coordinate_boundary_geometries": len(self._geometry_cache),
                "seconds": time.perf_counter() - started}

    def __getitem__(self, index: int) -> Data:
        cached = self._graph_cache[index]
        if cached is not None:
            return cached
        sol, bc_list = self.records[index]
        sol = np.asarray(sol, dtype=np.float64)
        bc = np.concatenate(bc_list, axis=0).astype(np.float64, copy=False)
        if sol.ndim != 2 or sol.shape[1] != 3 or len(sol) < 4 or not np.isfinite(sol).all() \
                or not np.isfinite(bc).all():
            raise ValueError(f"{self.split}[{index}]: invalid solution or boundary record")
        xy = np.ascontiguousarray(sol[:, :2])
        ids = _boundary_ids(xy, bc)
        if not np.allclose(sol[ids, 2], bc[:, 2], atol=1e-6, rtol=1e-6):
            raise ValueError(f"{self.split}[{index}]: boundary value differs from target")
        xy_normalized = _normalized_xy(xy)
        key = hashlib.sha256(xy.tobytes() + np.sort(ids).tobytes()).digest()
        static = self._geometry_cache.get(key)
        if static is None:
            edges = geometry_graph(xy, ids, ratio=self.ratio)
            if edges.ndim != 2 or edges.shape[1] != 2 or not len(edges):
                raise ValueError(f"{self.split}[{index}]: invalid coordinate edges")
            directed = np.concatenate((edges, edges[:, ::-1]), axis=0)
            edge_index = torch.from_numpy(directed.T.copy()).long()
            spacing = cKDTree(xy_normalized).query(xy_normalized, k=4)[0][:, 3]
            local_scale = np.maximum(spacing[directed[:, 0]], spacing[directed[:, 1]])
            delta = xy_normalized[directed[:, 1]] - xy_normalized[directed[:, 0]]
            relative = delta / local_scale[:, None]
            edge_attr = torch.from_numpy(np.column_stack((
                relative, np.linalg.norm(relative, axis=1))).astype(np.float32))
            if self.layout == "spatial_morton":
                mapping = _spatial_morton_mapping(xy_normalized)
            elif self.layout == "spatial_morton_anchor":
                mapping = _spatial_morton_anchor_mapping(xy_normalized)
            elif self.layout == "graph_components":
                mapping, _, _ = compile_interfaces(
                    edge_index, len(xy), layout="graph_components",
                    alignment=False, seed=0)
                mapping = mapping.long()
            else:
                raise ValueError(f"Unknown layout {self.layout}")
            hops = _graph_hops(len(xy), edges, ids)
            static = edge_index, edge_attr, mapping, hops
            self._geometry_cache[key] = static
        edge_index, edge_attr, mapping, hops = static
        boundary_mask = torch.zeros(len(xy), dtype=torch.bool)
        boundary_mask[torch.from_numpy(ids)] = True
        boundary_value = torch.zeros(len(xy), dtype=torch.float32)
        boundary_value[torch.from_numpy(ids)] = torch.from_numpy(bc[:, 2].astype(np.float32))
        x = torch.column_stack((torch.from_numpy(xy_normalized.astype(np.float32)),
                                boundary_value, boundary_mask.float()))
        graph = Data(x=x, y=torch.from_numpy(sol[:, 2].astype(np.float32)),
                     pos=torch.from_numpy(xy_normalized.astype(np.float32)),
                     edge_index=edge_index, edge_attr=edge_attr,
                     boundary_mask=boundary_mask, boundary_value=boundary_value,
                     wire_mapping=mapping, boundary_hops=hops)
        self._graph_cache[index] = graph
        return graph

    def sample_info(self) -> dict[str, Any]:
        graph = self[0]
        mapping = graph.wire_mapping.numpy()
        padded = 1 << (graph.num_nodes - 1).bit_length()
        real = np.zeros(padded, dtype=bool)
        real[mapping] = True
        real_pair_counts = []
        relay_pair_counts = []
        for level in range(padded.bit_length() - 1):
            bit = 1 << level
            left = np.arange(padded, dtype=np.int64)
            left = left[(left & bit) == 0]
            right = left ^ bit
            real_pair_counts.append(int(np.sum(real[left] & real[right])))
            relay_pair_counts.append(int(np.sum(real[left] ^ real[right])))
        return {
            "records": len(self), "nodes_first": graph.num_nodes,
            "directed_edges_first": int(graph.edge_index.shape[1]),
            "boundary_nodes_first": int(graph.boundary_mask.sum()),
            "interior_gt3_hops_first": int(((graph.boundary_hops > 3)
                                              & ~graph.boundary_mask).sum()),
            "padded_wires_first": padded,
            "active_stages_first": padded.bit_length() - 1,
            "real_wire_occupancy_first": graph.num_nodes / padded,
            "real_real_pairs_by_stage_first": real_pair_counts,
            "real_relay_pairs_by_stage_first": relay_pair_counts,
            "mapping_first_sha256": hashlib.sha256(mapping.tobytes()).hexdigest(),
        }

    def mapping_consistency_audit(self, groups: int = 3) -> dict[str, Any]:
        """Verify the ten boundary realizations of each sampled mesh share wires."""
        checked = []
        for group in range(min(groups, len(self) // 10)):
            first_index = group * 10
            reference = self[first_index]
            reference_xy = reference.pos
            reference_mapping = reference.wire_mapping
            for index in range(first_index + 1, first_index + 10):
                entry = self[index]
                if not torch.equal(entry.pos, reference_xy) or not torch.equal(
                        entry.wire_mapping, reference_mapping):
                    raise ValueError(f"{self.split}[{first_index}:{first_index+10}]: "
                                     "same-mesh boundary realizations change wires")
            checked.append({
                "record_indices": [first_index, first_index + 9],
                "nodes": reference.num_nodes,
                "padded_wires": 1 << (reference.num_nodes - 1).bit_length(),
                "wire_occupancy": (reference.num_nodes /
                                   (1 << (reference.num_nodes - 1).bit_length())),
                "mapping_sha256": hashlib.sha256(
                    reference_mapping.numpy().tobytes()).hexdigest(),
            })
        return {"checked_mesh_groups": checked,
                "boundary_realizations_per_group": 10,
                "mapping_identical_within_each_checked_group": True}


def _mappings(batch: Data) -> list[torch.Tensor]:
    sizes = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
    return list(torch.split(batch.wire_mapping.cpu(), sizes))


def _predict(model: MultiGraphGCM, batch: Data) -> torch.Tensor:
    raw = model.predict_nodes(batch, mappings=_mappings(batch)).flatten()
    return torch.where(batch.boundary_mask, batch.boundary_value, raw)


def _graph_relative_errors(prediction: torch.Tensor, batch: Data
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    per_graph = []
    interior = []
    for start, end in zip(batch.ptr[:-1].tolist(), batch.ptr[1:].tolist()):
        target = batch.y[start:end]
        error = prediction[start:end] - target
        mask = ~batch.boundary_mask[start:end]
        per_graph.append(torch.sqrt(error.square().sum() /
                                    target.square().sum().clamp_min(1e-12)))
        if bool(mask.any()):
            interior.append(torch.sqrt(error[mask].square().sum() /
                                       target[mask].square().sum().clamp_min(1e-12)))
    return torch.stack(per_graph), (torch.stack(interior) if interior else
                                    prediction.new_empty((0,)))


@torch.no_grad()
def _evaluate(model: MultiGraphGCM, loader: DataLoader,
              device: torch.device) -> dict[str, Any]:
    model.eval()
    _sync(device)
    started = time.perf_counter()
    all_errors: list[float] = []
    interior_errors: list[float] = []
    bin_sse = np.zeros(5, dtype=np.float64)
    bin_denominator = np.zeros(5, dtype=np.float64)
    bin_nodes = np.zeros(5, dtype=np.int64)
    interior_sae = 0.0
    interior_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        prediction = _predict(model, batch)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite SNI prediction")
        all_rel, interior_rel = _graph_relative_errors(prediction, batch)
        all_errors.extend(all_rel.cpu().tolist())
        interior_errors.extend(interior_rel.cpu().tolist())
        error = prediction - batch.y
        mask = ~batch.boundary_mask
        interior_sae += float(error[mask].abs().sum().item())
        interior_nodes += int(mask.sum().item())
        for index, category in enumerate((batch.boundary_hops == 0,
                                          batch.boundary_hops == 1,
                                          batch.boundary_hops == 2,
                                          batch.boundary_hops == 3,
                                          batch.boundary_hops >= 4)):
            bin_sse[index] += float(error[category].square().sum().item())
            bin_denominator[index] += float(batch.y[category].square().sum().item())
            bin_nodes[index] += int(category.sum().item())
    _sync(device)
    labels = ("boundary", "hop_1", "hop_2", "hop_3", "hop_4_plus")
    return {
        "graphs": len(all_errors),
        "all_node_relative_L2_percent": 100 * statistics.mean(all_errors),
        "all_node_relative_L2_std_percent": (100 * statistics.stdev(all_errors)
                                             if len(all_errors) > 1 else None),
        "interior_relative_L2_percent": (100 * statistics.mean(interior_errors)
                                         if interior_errors else None),
        "graphs_with_interior": len(interior_errors),
        "graphs_all_boundary": len(all_errors) - len(interior_errors),
        "interior_MAE": interior_sae / max(interior_nodes, 1),
        "nodes_by_boundary_hop": dict(zip(labels, bin_nodes.tolist())),
        "pooled_relative_L2_percent_by_hop": {
            label: (100 * math.sqrt(float(bin_sse[i] / bin_denominator[i]))
                    if bin_denominator[i] > 0 else None)
            for i, label in enumerate(labels)},
        "seconds": time.perf_counter() - started,
    }


def _train_epoch(model: MultiGraphGCM, loader: DataLoader,
                 optimizer: torch.optim.Optimizer, device: torch.device,
                 clip_norm: float) -> float:
    model.train()
    relative_sum = 0.0
    graph_count = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict(model, batch)
        all_rel, _ = _graph_relative_errors(prediction, batch)
        loss = all_rel.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite SNI training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        relative_sum += float(all_rel.detach().sum().item())
        graph_count += len(all_rel)
    return 100 * relative_sum / graph_count


def _make_model(args: argparse.Namespace, seed: int, scale: float) -> MultiGraphGCM:
    return MultiGraphGCM(
        input_dim=4, edge_dim=3, output_dim=1, dim=args.dim,
        heads=args.heads, blocks=args.blocks, max_stages=11,
        mode="original", exchange_scale=scale, dropout=args.dropout,
        angle=args.angle, stage_ff=False, graph_embed=True,
        graph_refresh=True, layout="graph_components", seed=seed,
        relay_padding=args.relay_padding, repeat_pair_switch=args.repeat_pair_switch,
        edge_weight_mode="symmetric",
    )


def _config(args: argparse.Namespace, data_info: dict[str, Any],
            seed: int, scale: float) -> dict[str, Any]:
    return {
        "runner": "run_sni_laplace", "seed": seed, "data": data_info,
        "model": {"class": "MultiGraphGCM", "mode": "original",
                  "exchange_scale": scale, "dim": args.dim,
                  "heads": args.heads, "blocks": args.blocks,
                  "max_stages": 11, "repeat_pair_switch": args.repeat_pair_switch,
                  "relay_padding": args.relay_padding, "dropout": args.dropout,
                  "angle": args.angle, "graph_embed": True,
                  "graph_refresh": True, "stage_ff": False,
                  "output": "all real-node states; exact known Dirichlet projection"},
        "training": {"optimizer": "AdamW", "lr": args.lr,
                     "weight_decay": args.weight_decay,
                     "batch_size": args.batch_size,
                     "val_batch_size": args.val_batch_size,
                     "epochs": args.epochs, "patience": args.patience,
                     "clip_norm": args.clip_norm,
                     "shuffle_seed": seed + 104729,
                     "objective": "mean per-graph all-node relative L2",
                     "selection": "minimum held-out simple-domain per-graph relative L2",
                     "test_access": ("A/B/C solution errors only after both arms have "
                                     "validation-selected checkpoints; official A/B/C "
                                     "meshes were used earlier to audit the "
                                     "coordinate-only graph rule"),
                     "timing": "train/val graph preprocessing is shared and excluded"},
    }


def _arm_paths(args: argparse.Namespace, seed: int, scale: float) -> dict[str, Path]:
    tag = f"{scale:g}".replace(".", "p")
    variant = f"{args.layout}_ratio_{args.edge_ratio:g}_repeat_{int(args.repeat_pair_switch)}"
    folder = args.output / variant / f"seed_{seed}" / f"scale_{tag}"
    return {name: folder / name for name in ("latest.pt", "best.pt", "epochs.jsonl",
                                             "fit.json", "result.json")}


def _fit_arm(args: argparse.Namespace, train: SniDataset, val: SniDataset,
             data_info: dict[str, Any], seed: int, scale: float,
             device: torch.device) -> dict[str, Any]:
    paths = _arm_paths(args, seed, scale)
    result_path = paths["fit.json"]
    latest_path, best_path = paths["latest.pt"], paths["best.pt"]
    config = _config(args, data_info, seed, scale)
    if result_path.is_file():
        if not args.resume:
            raise FileExistsError(f"Existing fit requires --resume: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["config"] != config:
            raise ValueError(f"Incompatible existing fit: {result_path}")
        return result
    if latest_path.is_file() and not args.resume:
        raise FileExistsError(f"Existing checkpoint requires --resume: {latest_path}")
    _seed_everything(seed)
    model = _make_model(args, seed, scale).to(device)
    initial_digest = _parameter_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed + 104729)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, generator=generator)
    val_loader = DataLoader(val, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=0)
    start_epoch, best_epoch, patience_count = 1, 0, 0
    best_score, elapsed_before = math.inf, 0.0
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        if saved["config"] != config or saved["initial_parameter_digest"] != initial_digest:
            raise ValueError(f"Incompatible checkpoint: {latest_path}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        _restore_rng(saved["rng"], generator)
        start_epoch = int(saved["epoch"]) + 1
        best_epoch = int(saved["best_epoch"])
        best_score = float(saved["best_val_score"])
        patience_count = int(saved["patience_count"])
        elapsed_before = float(saved["elapsed_seconds"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    epoch_durations: list[float] = []
    final_epoch = start_epoch - 1
    print(json.dumps({"seed": seed, "scale": scale, "start_epoch": start_epoch,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "device": str(device)}), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        final_epoch = epoch
        train_score = _train_epoch(model, train_loader, optimizer, device,
                                   args.clip_norm)
        val_metrics = _evaluate(model, val_loader, device)
        score = val_metrics["all_node_relative_L2_percent"]
        if score < best_score - 1e-12:
            best_score, best_epoch, patience_count = score, epoch, 0
            _save(best_path, {"config": config, "epoch": epoch,
                              "val": val_metrics, "model": _state_cpu(model)})
        else:
            patience_count += 1
        _sync(device)
        duration = time.perf_counter() - epoch_started
        epoch_durations.append(duration)
        elapsed = elapsed_before + time.perf_counter() - started
        _append_jsonl(paths["epochs.jsonl"], {
            "epoch": epoch, "train_percent": train_score, "val": val_metrics,
            "best_epoch": best_epoch, "best_val_percent": best_score,
            "epoch_seconds": duration, "elapsed_seconds": elapsed})
        _save(latest_path, {"config": config,
                            "initial_parameter_digest": initial_digest,
                            "epoch": epoch, "model": _state_cpu(model),
                            "optimizer": optimizer.state_dict(),
                            "rng": _rng_state(generator),
                            "best_epoch": best_epoch,
                            "best_val_score": best_score,
                            "patience_count": patience_count,
                            "elapsed_seconds": elapsed})
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(json.dumps({"seed": seed, "scale": scale, "epoch": epoch,
                              "train_percent": train_score,
                              "val_percent": score, "best_epoch": best_epoch}),
                  flush=True)
        if patience_count >= args.patience:
            break
    if best_epoch < 1 or not best_path.is_file():
        raise RuntimeError("No validation-selected SNI checkpoint")
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    result = {
        "status": "fit_complete", "config": config,
        "initial_parameter_digest": initial_digest,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "selected_epoch": best_epoch, "epochs_completed": final_epoch,
        "val": selected["val"],
        "training_seconds": elapsed_before + time.perf_counter() - started,
        "median_epoch_seconds_this_invocation": (
            statistics.median(epoch_durations) if epoch_durations else None),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
        "best_checkpoint": str(best_path.resolve()),
        "trace_path": str(paths["epochs.jsonl"].resolve()),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "device": str(device)},
    }
    _write_json(result_path, result)
    return result


def _test_arm(args: argparse.Namespace, fit: dict[str, Any], seed: int,
              scale: float, data_root: Path, device: torch.device
              ) -> dict[str, Any]:
    paths = _arm_paths(args, seed, scale)
    result_path = paths["result.json"]
    if result_path.is_file():
        if not args.resume:
            raise FileExistsError(f"Existing result requires --resume: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["fit"] != fit:
            raise ValueError(f"Incompatible existing result: {result_path}")
        return result
    model = _make_model(args, seed, scale).to(device)
    selected = torch.load(fit["best_checkpoint"], map_location="cpu",
                          weights_only=False)
    if selected["config"] != fit["config"]:
        raise ValueError("Selected checkpoint does not match fit")
    model.load_state_dict(selected["model"])
    tests = {}
    for split in ("A", "B", "C"):
        dataset = SniDataset(data_root / FILES[split], split,
                             args.edge_ratio, args.layout)
        loader = DataLoader(dataset, batch_size=args.test_batch_size,
                            shuffle=False, num_workers=0)
        tests[split] = _evaluate(model, loader, device)
        tests[split]["compiler_audit"] = {
            "sample": dataset.sample_info(),
            "same_mesh_boundary_realizations":
                dataset.mapping_consistency_audit(groups=1),
        }
        print(json.dumps({"seed": seed, "scale": scale, "test_domain": split,
                          "relative_L2_percent":
                          tests[split]["all_node_relative_L2_percent"]}), flush=True)
    result = {"status": "complete", "fit": fit, "tests": tests,
              "external_reference": OFFICIAL_REFERENCE}
    _write_json(result_path, result)
    return result


def _check_only(args: argparse.Namespace, root: Path) -> None:
    summaries = {}
    for split in FILES:
        data = SniDataset(root / FILES[split], split,
                          args.edge_ratio, args.layout)
        summaries[split] = {
            "sample": data.sample_info(),
            "same_mesh_boundary_realizations":
                data.mapping_consistency_audit(groups=3 if split in ("train", "val") else 1),
        }
        if split == "train":
            train = data
    _seed_everything(0)
    model = _make_model(args, 0, .25).cpu()
    batch = next(iter(DataLoader(train, batch_size=2, shuffle=False)))
    prediction = _predict(model, batch)
    rel, _ = _graph_relative_errors(prediction, batch)
    rel.mean().backward()
    if model.blocks[0].circuit.switches[0].base.grad is None:
        raise RuntimeError("Repeated original J switch did not receive gradient")
    print(json.dumps({"status": "protocol_check_passed", "summaries": summaries,
                      "train_active_stages_max": 8,
                      "test_active_stages": {"A": 10, "B": 11, "C": 10},
                      "model_max_stages": 11,
                      "repeat_pair_switch": args.repeat_pair_switch}), flush=True)


def _probe_only(args: argparse.Namespace, root: Path) -> None:
    """Time a worst-size train batch and one 11-level B forward, without updates."""
    train = SniDataset(root / FILES["train"], "train",
                       args.edge_ratio, args.layout)
    eval_b = SniDataset(root / FILES["B"], "B",
                        args.edge_ratio, args.layout)
    indices = sorted(range(len(train)),
                     key=lambda i: len(train.records[i][0]), reverse=True)
    indices = indices[:args.batch_size]
    train_batch = next(iter(DataLoader([train[i] for i in indices],
                                       batch_size=len(indices), shuffle=False)))
    b_batch = next(iter(DataLoader([eval_b[0]], batch_size=1, shuffle=False)))
    device = torch.device(args.device)
    train_batch, b_batch = train_batch.to(device), b_batch.to(device)
    arms = {}
    for scale in SCALES:
        _seed_everything(0)
        model = _make_model(args, 0, scale).to(device)
        digest = _parameter_digest(model)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for repeat in range(args.probe_repeats + 1):
            model.train()
            model.zero_grad(set_to_none=True)
            _sync(device)
            start = time.perf_counter()
            prediction = _predict(model, train_batch)
            loss = _graph_relative_errors(prediction, train_batch)[0].mean()
            loss.backward()
            _sync(device)
            if repeat:
                durations.append(time.perf_counter() - start)
        model.eval()
        with torch.no_grad():
            _sync(device)
            start = time.perf_counter()
            b_prediction = _predict(model, b_batch)
            _sync(device)
            b_seconds = time.perf_counter() - start
            if not torch.isfinite(b_prediction).all():
                raise FloatingPointError("Nonfinite B-domain probe prediction")
        arms[str(scale)] = {
            "initial_parameter_digest": digest,
            "train_step_seconds": durations,
            "train_step_median_seconds": statistics.median(durations),
            "B_forward_seconds": b_seconds,
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
            "peak_cuda_reserved_bytes": (
                torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if arms["0.0"]["initial_parameter_digest"] != arms["0.25"]["initial_parameter_digest"]:
        raise RuntimeError("Probe arms did not share initialization")
    result = {
        "status": "probe_only_no_optimizer_step_no_test_score",
        "device": str(device), "layout": args.layout, "edge_ratio": args.edge_ratio,
        "train_batch_graphs": len(indices),
        "train_batch_nodes": [train[i].num_nodes for i in indices],
        "train_batch_padded_wires": 256,
        "B_nodes": b_batch.num_nodes, "B_padded_wires": 2048,
        "max_stages": 11, "repeat_pair_switch": args.repeat_pair_switch,
        "dim": args.dim, "heads": args.heads, "blocks": args.blocks,
        "arms": arms,
    }
    path = args.output / "resource_probe.json"
    _write_json(path, result)
    print(json.dumps({"resource_probe": str(path), "arms": arms}), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=data_path('sni_laplace'))
    parser.add_argument("--output", type=Path,
                        default=REPO / "results/sni_laplace_gcm")
    parser.add_argument("--layout", choices=["spatial_morton", "spatial_morton_anchor",
                                               "graph_components"],
                        default="spatial_morton")
    parser.add_argument("--edge-ratio", type=float, default=1.3)
    parser.add_argument("--seeds", "--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--angle", type=float, default=0.2)
    parser.add_argument("--repeat-pair-switch", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--relay-padding", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-repeats", type=int, default=2)
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    for value in (args.batch_size, args.val_batch_size, args.test_batch_size,
                  args.epochs, args.patience, args.blocks, args.threads,
                  args.print_every, args.probe_repeats):
        if value < 1:
            parser.error("Batch sizes and positive settings must be >= 1")
    if args.dim < 1 or args.heads < 1 or args.dim % (2 * args.heads):
        parser.error("dim must be divisible by twice the head count")
    if not (math.isfinite(args.edge_ratio) and args.edge_ratio > 0
            and math.isfinite(args.angle) and 0 <= args.dropout < 1
            and args.lr > 0 and args.weight_decay >= 0 and args.clip_norm > 0):
        parser.error("Invalid finite geometry or training setting")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    if args.check_only and args.probe_only:
        parser.error("Choose only one of --check-only and --probe-only")
    return args


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.threads)
    root = args.data_root.resolve()
    if args.check_only:
        _check_only(args, root)
        return
    if args.probe_only:
        _probe_only(args, root)
        return
    files = {split: root / filename for split, filename in FILES.items()}
    for split in ("train", "val"):
        if not files[split].is_file():
            raise FileNotFoundError(files[split])
    train = SniDataset(files["train"], "train", args.edge_ratio, args.layout)
    val = SniDataset(files["val"], "val", args.edge_ratio, args.layout)
    prep = {"train": train.prepare_all(), "val": val.prepare_all()}
    prep_path = args.output / (
        f"{args.layout}_ratio_{args.edge_ratio:g}_repeat_{int(args.repeat_pair_switch)}"
    ) / "shared_preparation.json"
    _write_json(prep_path, {"status": "shared_preprocessing_before_both_arms",
                            "splits": prep})
    print(json.dumps({"shared_preparation": str(prep_path),
                      "split_seconds": {k: v["seconds"] for k, v in prep.items()}}),
          flush=True)
    data_info = {
        "benchmark": "ICLR 2026 SNI Laplace2d-Dirichlet",
        "official_source_commit": "67d3227e9d5938d6cbb0dcc06ec2db11c652418b",
        "official_file_sha256": {split: _sha256(path)
                                 for split, path in files.items()},
        "file_counts": COUNTS,
        "graph_rule": ("all-coordinate Delaunay, retain edge length <= "
                       f"{args.edge_ratio:g} * max(endpoint third-neighbor spacing)"),
        "graph_rule_input": "coordinates only; never released .msh or solution values",
        "compiler_layout": args.layout,
        "compiler_layout_definition": (
            "isotropically bbox-normalized xy, 16-bit Morton key, stable coordinate "
            "tie break, rank evenly spaced across each graph's next-power-of-two wires"
            if args.layout == "spatial_morton" else
            "isotropically bbox-normalized xy, 16-bit Morton coordinate address, "
            "monotone least-squares projection to unique next-power-of-two wires"
            if args.layout == "spatial_morton_anchor" else
            "topology-only graph_components on coordinate graph; fixed compiler seed 0"),
        "input_features": ("bbox-centered, isotropically scaled xy; observed "
                           "Dirichlet value; boundary mask"),
        "edge_features": "relative xy and length divided by local third-neighbor spacing",
        "boundary_projection": "copy exact observed Dirichlet values at boundary nodes",
        "score": "100 * mean_graph sqrt(sum_all_nodes squared error / sum_all_nodes squared target)",
        "test_domains_not_for_selection": ["A", "B", "C"],
        "first_graphs": {"train": train.sample_info(), "val": val.sample_info()},
        "unique_train_geometry_count": prep["train"]["unique_coordinate_boundary_geometries"],
        "unique_val_geometry_count": prep["val"]["unique_coordinate_boundary_geometries"],
        "compiler_same_mesh_consistency": {
            "train": train.mapping_consistency_audit(),
            "val": val.mapping_consistency_audit(),
        },
        "stage_coverage": {"train_max": 8, "A": 10, "B": 11, "C": 10},
    }
    device = torch.device(args.device)
    for seed in args.seeds:
        fits = [_fit_arm(args, train, val, data_info, seed, scale, device)
                for scale in SCALES]
        if fits[0]["initial_parameter_digest"] != fits[1]["initial_parameter_digest"]:
            raise RuntimeError("Paired arms do not share initialized parameters")
        # Neither evaluation geometry nor its target is accessed until both
        # validation-selected checkpoints exist for this seed.
        results = [_test_arm(args, fit, seed, scale, root, device)
                   for fit, scale in zip(fits, SCALES)]
        paired = {
            "seed": seed, "data": data_info,
            "initial_parameter_digest": fits[0]["initial_parameter_digest"],
            "zero": results[0], "J": results[1],
            "J_minus_zero_relative_L2_percent_points": {
                split: (results[1]["tests"][split]["all_node_relative_L2_percent"]
                        - results[0]["tests"][split]["all_node_relative_L2_percent"])
                for split in ("A", "B", "C")},
        }
        folder = _arm_paths(args, seed, .0)["fit.json"].parent.parent
        path = folder / "paired.json"
        _write_json(path, paired)
        print(json.dumps({"paired_result": str(path),
                          "delta_percent_points":
                          paired["J_minus_zero_relative_L2_percent_points"]}),
              flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
