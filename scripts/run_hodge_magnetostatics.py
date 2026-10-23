"""Matched zero/original-J GCM on the public ICML 2026 Magnetostatics task.

The official generated fixed mesh supplies one scalar source and one vector
target per node. Only mesh positions and tetrahedral adjacency enter the
compiler. The zero and J arms share every learned module and initialization;
the exchange scale is their sole model difference. The test split is evaluated
only after both arms have completed validation selection.
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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gcm.paths import data_path
from gcm.multigraph import MultiGraphGCM  # noqa: E402

OFFICIAL_COMMIT = "3ce166072f33237163ae68e620f9a109564f22f4"
DEFAULT_DATA = data_path('hodge_magnetostatics/official_cpu_seed42_3000/flux_field_dataset.pkl')
DEFAULT_SPLIT = DEFAULT_DATA.with_name("split_indices_public_code_seed42.npz")
EXPECTED = {"samples": 3000, "nodes": 3017, "elements": 19010,
            "train": 2040, "val": 360, "test": 600}


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    return value


def _replace(temp: Path, path: Path) -> None:
    for attempt in range(5):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (2 ** attempt))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True)
                        + "\n", encoding="utf-8")
        _replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temp)
        _replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _append(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")
        handle.flush()


def _state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def _rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "shuffle": generator.get_state()}


def _restore_rng(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["shuffle"])


def _mesh_edges(elements: np.ndarray, nodes: int) -> np.ndarray:
    """The six undirected edges of each tetrahedron, deduplicated."""
    pairs = np.concatenate(
        [elements[:, [i, j]] for i in range(4) for j in range(i + 1, 4)],
        axis=0,
    )
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or np.any(pairs < 0) \
            or np.any(pairs >= nodes) or np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("Invalid tetrahedral mesh edges")
    return pairs.astype(np.int64, copy=False)


def compile_fixed_mesh(points: np.ndarray, undirected: np.ndarray) -> torch.Tensor:
    """Recursive geometric bisection with minimum mesh-edge-cut axis.

    Each split uses *only* fixed coordinates and mesh adjacency. At each
    power-of-two wire region, sort nodes along each coordinate axis, split
    the nodes as evenly as capacity permits, and choose the axis cutting the
    fewest edges. Thus neighboring wires have a reproducible spatial meaning
    across every sample. No source value, target, or split membership enters.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if points.shape != (n, 3) or not np.isfinite(points).all():
        raise ValueError("Fixed mesh requires finite [nodes, 3] positions")
    padded = 1 << (n - 1).bit_length()
    mapping = np.full(n, -1, dtype=np.int64)
    side = np.zeros(n, dtype=np.bool_)

    def recurse(ids: np.ndarray, edges: np.ndarray, begin: int,
                capacity: int) -> None:
        count = len(ids)
        if not count:
            return
        if count == 1:
            mapping[ids[0]] = begin
            return
        half = capacity // 2
        left_count = min(max(count // 2, count - half), half)
        if not 0 < left_count < count:
            raise AssertionError("Impossible spatial bisection")
        ranges = np.ptp(points[ids], axis=0)
        best: tuple[int, float, int] | None = None
        best_left: np.ndarray | None = None
        for axis in range(3):
            order = np.lexsort((ids, points[ids, axis]))
            left = ids[order[:left_count]]
            side[left] = True
            cuts = int(np.count_nonzero(side[edges[:, 0]] ^ side[edges[:, 1]]))
            side[left] = False
            candidate = (cuts, -float(ranges[axis]), axis)
            if best is None or candidate < best:
                best, best_left = candidate, left
        assert best_left is not None
        side[best_left] = True
        in_left = side[ids]
        edge_left = side[edges[:, 0]] & side[edges[:, 1]]
        edge_right = ~side[edges[:, 0]] & ~side[edges[:, 1]]
        left_ids, right_ids = ids[in_left], ids[~in_left]
        side[best_left] = False
        recurse(left_ids, edges[edge_left], begin, half)
        recurse(right_ids, edges[edge_right], begin + half, half)

    recurse(np.arange(n, dtype=np.int64), np.asarray(undirected), 0, padded)
    if np.any(mapping < 0) or len(np.unique(mapping)) != n or np.max(mapping) >= padded:
        raise AssertionError("Mesh compiler did not produce an injective mapping")
    return torch.from_numpy(mapping)


def _official_splits(samples: int, split_file: Path | None = None) -> dict[str, np.ndarray]:
    indices = np.arange(samples)
    train_val, test = train_test_split(indices, test_size=.2, random_state=42)
    train, val = train_test_split(train_val, test_size=.15, random_state=42)
    expected = {"train": train, "val": val, "test": test}
    if split_file is not None:
        with np.load(split_file) as stored:
            for key, value in expected.items():
                if not np.array_equal(stored[key], value):
                    raise ValueError(f"Split file {split_file} differs from official {key} indices")
    if len(np.unique(np.concatenate(tuple(expected.values())))) != samples:
        raise AssertionError("Official split indices overlap")
    return expected


class HodgeGraphs(Dataset):
    """Lazy fixed-mesh PyG graphs; arrays are shared across all samples."""

    def __init__(self, source: np.ndarray, target: np.ndarray,
                 indices: np.ndarray, points: torch.Tensor,
                 edge_index: torch.Tensor, edge_attr: torch.Tensor,
                 x_scale: float, y_scale: float):
        self.source = source
        self.target = target
        self.indices = np.asarray(indices, dtype=np.int64)
        self.points = points
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.x_scale = float(x_scale)
        self.y_scale = float(y_scale)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> Data:
        index = int(self.indices[position])
        scalar = torch.from_numpy(self.source[index] / self.x_scale)
        x = torch.cat((scalar[:, None], self.points), dim=1)
        y = torch.from_numpy(self.target[index] / self.y_scale)
        return Data(x=x, y=y, edge_index=self.edge_index,
                    edge_attr=self.edge_attr, num_nodes=len(self.points))


def load_official(data_file: Path, split_file: Path | None = None,
                  strict_size: bool = True) -> tuple[dict[str, HodgeGraphs],
                                                       dict[str, Any], torch.Tensor]:
    # The pickle is created locally by the pinned official generator, not a
    # remote arbitrary object stream.
    with data_file.open("rb") as handle:
        raw = pickle.load(handle)
    required = ("X_data", "Y_data", "nodes", "elements")
    if any(key not in raw for key in required):
        raise ValueError(f"Official data requires {required}")
    source = np.asarray(raw["X_data"], dtype=np.float32)
    target = np.asarray(raw["Y_data"], dtype=np.float32)
    points = np.asarray(raw["nodes"], dtype=np.float32)
    elements = np.asarray(raw["elements"], dtype=np.int64)
    samples, nodes = source.shape
    if target.shape != (samples, nodes, 3) or points.shape != (nodes, 3) \
            or elements.ndim != 2 or elements.shape[1] != 4:
        raise ValueError("Official source, vector target, or tetrahedron shape is invalid")
    if strict_size and (samples != EXPECTED["samples"] or nodes != EXPECTED["nodes"]
                        or len(elements) != EXPECTED["elements"]):
        raise ValueError("Generated data does not have the official 3000/3017/19010 shape")
    if any(not np.isfinite(array).all() for array in (source, target, points)):
        raise ValueError("Official data contains nonfinite source, target, or coordinates")
    if np.any(elements < 0) or np.any(elements >= nodes):
        raise ValueError("Tetrahedron references a nonexistent node")
    splits = _official_splits(samples, split_file)
    if strict_size and any(len(splits[key]) != EXPECTED[key]
                           for key in ("train", "val", "test")):
        raise ValueError("Official split counts do not match 2040/360/600")
    train_val = np.concatenate((splits["train"], splits["val"]))
    # This deliberately matches public main.py lines 329-338: both maxima
    # use train+validation, and vector MSE is on these normalized targets.
    x_scale = float(np.max(np.abs(source[train_val])) + 1e-9)
    y_scale = float(np.max(np.abs(target[train_val])) + 1e-9)
    if not x_scale > 0 or not y_scale > 0:
        raise ValueError("Official scaling factor must be positive")
    undirected = _mesh_edges(elements, nodes)
    directed = np.concatenate((undirected, undirected[:, ::-1]), axis=0)
    edge_index = torch.from_numpy(directed.T.copy())
    point_tensor = torch.from_numpy(points.copy())
    edge_attr = point_tensor[edge_index[1]] - point_tensor[edge_index[0]]
    mapping = compile_fixed_mesh(points, undirected)
    datasets = {
        key: HodgeGraphs(source, target, value, point_tensor,
                         edge_index, edge_attr, x_scale, y_scale)
        for key, value in splits.items()
    }
    xor = mapping[undirected[:, 0]] ^ mapping[undirected[:, 1]]
    pair_levels = torch.floor(torch.log2(xor.float())).long()
    info = {
        "benchmark": "ICML 2026 HSD Magnetostatics, public main.py random split",
        "official_code_commit": OFFICIAL_COMMIT,
        "data_file": str(data_file.resolve()), "data_sha256": _sha256(data_file),
        "split_file": str(split_file.resolve()) if split_file else None,
        "split_sha256": _sha256(split_file) if split_file else None,
        "split_method": "sklearn train_test_split twice, random_state=42, test=.2, val=.15 of trainval",
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "sample_count": samples, "node_count": nodes,
        "tetrahedron_count": len(elements), "undirected_edge_count": len(undirected),
        "padded_wires": 1 << (nodes - 1).bit_length(),
        "max_stages": (nodes - 1).bit_length(),
        "x_scale_trainval_maxabs": x_scale,
        "y_scale_trainval_maxabs": y_scale,
        "compiler": "fixed recursive coordinate bisection; minimum tetrahedral edge-cut axis",
        "compiler_sha256": hashlib.sha256(mapping.numpy().tobytes()).hexdigest(),
        "mesh_edge_direct_pair_fraction": float((pair_levels == 0).float().mean()),
        "mesh_edge_mean_pair_level": float(pair_levels.float().mean()),
        "node_features": "[X_normalized, raw x/y/z coordinates]",
        "edge_features": "directed target-minus-source x/y/z coordinate difference",
        "primary_metric": "mean squared error over all normalized scalar vector components",
    }
    return datasets, info, mapping


class _Metrics:
    def __init__(self) -> None:
        self.sse = torch.zeros(3, dtype=torch.float64)
        self.sae = torch.zeros(3, dtype=torch.float64)
        self.target_sum = torch.zeros(3, dtype=torch.float64)
        self.target_sum2 = torch.zeros(3, dtype=torch.float64)
        self.nodes = 0
        self.graphs = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor,
               graphs: int) -> None:
        if prediction.shape != target.shape or target.ndim != 2 or target.shape[1] != 3:
            raise ValueError("Vector metric requires matching [all nodes, 3] tensors")
        residual = (prediction - target).detach().double().cpu()
        truth = target.detach().double().cpu()
        self.sse += residual.square().sum(dim=0)
        self.sae += residual.abs().sum(dim=0)
        self.target_sum += truth.sum(dim=0)
        self.target_sum2 += truth.square().sum(dim=0)
        self.nodes += len(truth)
        self.graphs += graphs

    def result(self) -> dict[str, Any]:
        if not self.nodes:
            raise ValueError("Empty metric collection")
        total_sse = float(self.sse.sum())
        sst = float((self.target_sum2 - self.target_sum.square() / self.nodes).sum())
        return {
            "normalized_scalar_MSE": total_sse / (3 * self.nodes),
            "normalized_nodewise_vector_norm_SSE": total_sse / self.nodes,
            "normalized_scalar_MAE": float(self.sae.sum()) / (3 * self.nodes),
            "normalized_component_MSE": (self.sse / self.nodes).tolist(),
            "normalized_component_MAE": (self.sae / self.nodes).tolist(),
            "normalized_vector_R2": 1. - total_sse / sst if sst > 0 else None,
            "nodes": self.nodes, "graphs": self.graphs,
        }


def _model(args: argparse.Namespace, info: dict[str, Any], seed: int,
           scale: float) -> MultiGraphGCM:
    model = MultiGraphGCM(
        input_dim=4, edge_dim=3, output_dim=3,
        dim=args.dim, heads=args.heads, blocks=args.blocks,
        max_stages=info["max_stages"], mode="original",
        exchange_scale=scale, dropout=args.dropout, angle=args.angle,
        stage_ff=False, graph_embed=True, graph_refresh=True,
        layout="graph_components", seed=seed,
        relay_padding=True, edge_weight_mode="symmetric",
    )
    if args.balanced_carrier_heads:
        with torch.no_grad():
            for block in model.blocks:
                for switch in block.circuit.switches:
                    switch.base[:args.balanced_carrier_heads] = math.pi / 4
    return model


def _predict(model: MultiGraphGCM, batch: Data,
             mapping: torch.Tensor) -> torch.Tensor:
    return model.predict_nodes(batch, mappings=[mapping] * (len(batch.ptr) - 1))


def _loader(dataset: HodgeGraphs, batch_size: int, shuffle: bool = False,
            generator: torch.Generator | None = None) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator)


@torch.no_grad()
def _evaluate(model: MultiGraphGCM, dataset: HodgeGraphs,
              batch_size: int, mapping: torch.Tensor,
              device: torch.device) -> dict[str, Any]:
    model.eval()
    _sync(device)
    started = time.perf_counter()
    metrics = _Metrics()
    for batch in _loader(dataset, batch_size):
        batch = batch.to(device)
        prediction = _predict(model, batch, mapping)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite vector prediction")
        metrics.update(prediction, batch.y, len(batch.ptr) - 1)
    _sync(device)
    result = metrics.result()
    result["seconds"] = time.perf_counter() - started
    return result


def _train_epoch(model: MultiGraphGCM, dataset: HodgeGraphs,
                 optimizer: torch.optim.Optimizer, batch_size: int,
                 generator: torch.Generator, mapping: torch.Tensor,
                 device: torch.device, clip_norm: float) -> float:
    model.train()
    sse, count = 0., 0
    for batch in _loader(dataset, batch_size, shuffle=True, generator=generator):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict(model, batch, mapping)
        loss = (prediction - batch.y).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        sse += float(loss.detach()) * batch.y.numel()
        count += batch.y.numel()
    return sse / count


def _config(args: argparse.Namespace, info: dict[str, Any], seed: int,
            scale: float) -> dict[str, Any]:
    return {
        "runner": "run_hodge_magnetostatics", "data": info,
        "seed": seed,
        "model": {"class": "MultiGraphGCM", "mode": "original",
                  "exchange_scale": scale, "dim": args.dim,
                  "heads": args.heads, "blocks": args.blocks,
                  "max_stages": info["max_stages"], "dropout": args.dropout,
                  "angle": args.angle,
                  "balanced_carrier_heads": args.balanced_carrier_heads,
                  "balanced_carrier_initial_angle": (
                      math.pi / 4 if args.balanced_carrier_heads else None),
                  "graph_embed": True,
                  "graph_refresh": True, "relay_padding": True,
                  "stage_ff": False},
        "training": {"optimizer": "AdamW", "lr": args.lr,
                     "weight_decay": args.weight_decay,
                     "batch_size": args.batch_size, "epochs": args.epochs,
                     "patience": args.patience, "clip_norm": args.clip_norm,
                     "shuffle_seed": seed + 104729,
                     "selection": "minimum full validation normalized scalar MSE",
                     "test_access": "after both arms have validation-selected checkpoints"},
    }


def _train_arm(args: argparse.Namespace, datasets: dict[str, HodgeGraphs],
               info: dict[str, Any], mapping: torch.Tensor, seed: int,
               scale: float, device: torch.device) -> dict[str, Any]:
    folder = args.output / f"seed_{seed}" / f"scale_{str(scale).replace('.', 'p')}"
    complete_path = folder / "train_complete.json"
    latest_path, best_path = folder / "latest.pt", folder / "best.pt"
    trace_path = folder / "epochs.jsonl"
    config = _config(args, info, seed, scale)
    if complete_path.is_file():
        if not args.resume:
            raise FileExistsError(f"Use --resume for existing result: {complete_path}")
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete["config"] != config or not best_path.is_file():
            raise ValueError("Completed arm is incompatible or missing checkpoint")
        return complete
    if latest_path.is_file() and not args.resume:
        raise FileExistsError(f"Use --resume for existing checkpoint: {latest_path}")
    _seed(seed)
    model = _model(args, info, seed, scale).to(device)
    initial_digest = _parameter_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed + 104729)
    epoch_start, best_epoch, stale = 1, 0, 0
    best_score, elapsed_before = math.inf, 0.
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        if saved["config"] != config or saved["initial_digest"] != initial_digest:
            raise ValueError("Checkpoint configuration or initialization differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        _restore_rng(saved["rng"], generator)
        epoch_start = saved["epoch"] + 1
        best_epoch, best_score, stale = (saved["best_epoch"],
                                       saved["best_score"], saved["stale"])
        elapsed_before = saved["elapsed_seconds"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    durations: list[float] = []
    final_epoch = epoch_start - 1
    print(json.dumps({"seed": seed, "scale": scale,
                      "start_epoch": epoch_start,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "device": str(device)}), flush=True)
    # A process can stop after its last epoch checkpoint but before writing
    # train_complete.json. Preserve an already-reached patience stop on resume.
    epochs_to_run = (range(epoch_start, args.epochs + 1)
                     if stale < args.patience else range(0))
    for epoch in epochs_to_run:
        epoch_time = time.perf_counter()
        final_epoch = epoch
        train_mse = _train_epoch(model, datasets["train"], optimizer,
                                 args.batch_size, generator, mapping,
                                 device, args.clip_norm)
        validation = _evaluate(model, datasets["val"], args.batch_size,
                               mapping, device)
        score = validation["normalized_scalar_MSE"]
        if score < best_score - 1e-12:
            best_score, best_epoch, stale = score, epoch, 0
            _save(best_path, {"config": config, "epoch": epoch,
                              "val": validation, "model": _state_cpu(model)})
        else:
            stale += 1
        _sync(device)
        duration = time.perf_counter() - epoch_time
        durations.append(duration)
        elapsed = elapsed_before + time.perf_counter() - started
        _append(trace_path, {"epoch": epoch, "train_scalar_MSE": train_mse,
                             "val": validation, "best_epoch": best_epoch,
                             "best_val_scalar_MSE": best_score,
                             "epoch_seconds": duration, "elapsed_seconds": elapsed})
        _save(latest_path, {"config": config, "initial_digest": initial_digest,
                            "epoch": epoch, "model": _state_cpu(model),
                            "optimizer": optimizer.state_dict(),
                            "rng": _rng_state(generator), "best_epoch": best_epoch,
                            "best_score": best_score, "stale": stale,
                            "elapsed_seconds": elapsed})
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(json.dumps({"seed": seed, "scale": scale, "epoch": epoch,
                              "train_scalar_MSE": train_mse,
                              "val_scalar_MSE": score,
                              "best_epoch": best_epoch}), flush=True)
        if stale >= args.patience:
            break
    if best_epoch < 1 or not best_path.is_file():
        raise RuntimeError("No validation-selected checkpoint")
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    complete = {
        "status": "validation_selected", "config": config,
        "initial_parameter_digest": initial_digest,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "selected_epoch": best_epoch, "epochs_completed": final_epoch,
        "val": selected["val"],
        "training_seconds": elapsed_before + time.perf_counter() - started,
        "median_epoch_seconds_this_invocation": (
            statistics.median(durations) if durations else None),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
        "best_checkpoint": str(best_path.resolve()),
        "trace_path": str(trace_path.resolve()),
        "environment": {"python": sys.version,
                        "platform": platform.platform(),
                        "torch": torch.__version__, "device": str(device)},
    }
    _write_json(complete_path, complete)
    print(json.dumps({"validation_selected": str(complete_path),
                      "best_val_scalar_MSE": best_score}), flush=True)
    return complete


def _test_selected(complete: dict[str, Any], args: argparse.Namespace,
                   info: dict[str, Any], test: HodgeGraphs,
                   mapping: torch.Tensor, device: torch.device) -> dict[str, Any]:
    config = complete["config"]
    seed = config["seed"]
    scale = config["model"]["exchange_scale"]
    model = _model(args, info, seed, scale).to(device)
    selected = torch.load(complete["best_checkpoint"], map_location="cpu",
                          weights_only=False)
    if selected["config"] != config or selected["epoch"] != complete["selected_epoch"]:
        raise ValueError("Best checkpoint and validation-selection record differ")
    model.load_state_dict(selected["model"])
    return _evaluate(model, test, args.batch_size, mapping, device)


def _check_or_probe(args: argparse.Namespace,
                    datasets: dict[str, HodgeGraphs], info: dict[str, Any],
                    mapping: torch.Tensor, device: torch.device) -> None:
    batch_size = args.probe_batch_size if args.resource_probe else 2
    batch = next(iter(_loader(datasets["train"], batch_size))).to(device)
    digests = []
    timings = []
    for scale in (0., args.j_exchange_scale):
        _seed(args.seeds[0])
        model = _model(args, info, args.seeds[0], scale).to(device)
        digests.append(_parameter_digest(model))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        started = time.perf_counter()
        model.train()
        prediction = _predict(model, batch, mapping)
        if prediction.shape != batch.y.shape or not torch.isfinite(prediction).all():
            raise AssertionError("Paired full-node vector forward failed")
        loss = (prediction - batch.y).square().mean()
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in model.parameters()):
            raise AssertionError("Nonfinite paired backward gradient")
        if args.resource_probe:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                          weight_decay=args.weight_decay)
            optimizer.step()
        else:
            optimizer = None
        _sync(device)
        timings.append({"scale": scale, "graphs": len(batch.ptr) - 1,
                        "nodes": int(batch.y.shape[0]),
                        "seconds": time.perf_counter() - started,
                        "peak_cuda_allocated_bytes": (
                            torch.cuda.max_memory_allocated(device)
                            if device.type == "cuda" else None)})
        prediction_shape = list(prediction.shape)
        del prediction, loss, optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if digests[0] != digests[1]:
        raise AssertionError("Zero/J initial learned parameters differ")
    print(json.dumps({"check": "passed", "mode": (
        "one_batch_resource_probe_with_optimizer_step" if args.resource_probe
        else "forward_backward_no_optimizer_step"),
        "paired_initialization": True, "device": str(device),
        "balanced_carrier_heads": args.balanced_carrier_heads,
        "balanced_carrier_initial_angle": (
            math.pi / 4 if args.balanced_carrier_heads else None),
        "prediction_shape": prediction_shape, "timings": timings,
        "note": "Train samples only; no validation selection or test evaluation"}),
        flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split-indices", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/hodge_magnetostatics_gcm/fixed_mesh")
    parser.add_argument("--seeds", "--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--j-exchange-scale", type=float, default=.25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--angle", type=float, default=.2)
    parser.add_argument("--balanced-carrier-heads", type=int, default=1,
                        help="Initialize first K existing J heads at pi/4 at all stages; 0 disables")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--clip-norm", type=float, default=1.)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resource-probe", action="store_true")
    parser.add_argument("--probe-batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if not args.data.is_file() or not args.split_indices.is_file():
        parser.error("Generated official data and saved split indices are required")
    if args.check_only and args.resource_probe:
        parser.error("Choose check-only or resource-probe")
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        parser.error("Seeds must be unique and nonempty")
    if any(value < 1 for value in (args.batch_size, args.probe_batch_size,
                                    args.epochs, args.patience, args.blocks,
                                    args.threads, args.print_every)):
        parser.error("Positive batch, training, and thread settings required")
    if args.dim < 1 or args.heads < 1 or args.dim % (2 * args.heads):
        parser.error("dim must be divisible by twice the head count")
    if not 0 <= args.balanced_carrier_heads <= args.heads:
        parser.error("balanced-carrier-heads must be between zero and heads")
    if not 0. < args.j_exchange_scale <= 1. or not math.isfinite(args.j_exchange_scale):
        parser.error("J exchange scale must be finite in (0, 1]")
    if not 0. <= args.dropout < 1. or not math.isfinite(args.angle) \
            or args.lr <= 0. or args.weight_decay < 0. or args.clip_norm <= 0.:
        parser.error("Invalid optimizer, dropout, or angle setting")
    return args


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.threads)
    datasets, info, mapping = load_official(args.data, args.split_indices)
    print(json.dumps(info, sort_keys=True), flush=True)
    device = torch.device(args.device)
    if args.check_only or args.resource_probe:
        _check_or_probe(args, datasets, info, mapping, device)
        return
    fitting = {key: datasets[key] for key in ("train", "val")}
    for seed in args.seeds:
        # Both arms finish validation selection before either sees test.
        zero = _train_arm(args, fitting, info, mapping, seed, 0., device)
        exchange = _train_arm(args, fitting, info, mapping, seed,
                              args.j_exchange_scale, device)
        if zero["initial_parameter_digest"] != exchange["initial_parameter_digest"]:
            raise AssertionError("Paired arms did not share initialization")
        paired_path = args.output / f"seed_{seed}" / "paired.json"
        if paired_path.is_file():
            if not args.resume:
                raise FileExistsError(f"Use --resume for existing pair: {paired_path}")
            paired = json.loads(paired_path.read_text(encoding="utf-8"))
            if paired["zero"]["training"] != zero \
                    or paired["original_J"]["training"] != exchange:
                raise ValueError("Existing paired report has incompatible arms")
            print(json.dumps({"already_complete": str(paired_path)}), flush=True)
            continue
        zero_test = _test_selected(zero, args, info, datasets["test"], mapping, device)
        exchange_test = _test_selected(exchange, args, info, datasets["test"],
                                       mapping, device)
        paired = {
            "status": "complete", "benchmark": info["benchmark"],
            "seed": seed, "initial_parameter_digest": zero["initial_parameter_digest"],
            "balanced_carrier_heads": args.balanced_carrier_heads,
            "balanced_carrier_initial_angle": (
                math.pi / 4 if args.balanced_carrier_heads else None),
            "zero": {"training": zero, "test": zero_test},
            "original_J": {"training": exchange, "test": exchange_test},
            "delta_J_minus_zero": {
                split: {key: (exchange_test if split == "test" else exchange["val"])[key]
                        - (zero_test if split == "test" else zero["val"])[key]
                        for key in ("normalized_scalar_MSE",
                                    "normalized_nodewise_vector_norm_SSE",
                                    "normalized_scalar_MAE", "normalized_vector_R2")}
                for split in ("val", "test")},
            "external_paper_table_1": {"HSD_reported_MSE": 1.84e-4,
                                        "DeepONet_reported_MSE": 2.89e-4,
                                        "MGN_reported_MSE": 3.31e-3,
                                        "warning": "Paper says temporal split and nodewise norm MSE; public code uses random split and flattened per-component MSE"},
        }
        _write_json(paired_path, paired)
        print(json.dumps({"paired_result": str(paired_path),
                          "delta_test_scalar_MSE": paired["delta_J_minus_zero"]["test"]["normalized_scalar_MSE"]}),
              flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
