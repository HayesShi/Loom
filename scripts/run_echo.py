"""Paired full-depth GCM runs on the official ECHO benchmark.

The official ECHO loader is used for every task and split.  SSSP and ECC are
node-level regression tasks; DIAM is graph-level regression.  Synthetic
targets are divided by 40, Charge targets stay in native units, and Energy
targets are log10(original energy).  Training follows the official log10(MSE)
objective.  Checkpoint selection uses the global validation log10 MSE in the
loader target space, while every result also contains task-unit MAE and MSE.

The two matched arms share the graph compiler, initialization, data order,
optimizer, and full ceil(log2(max_nodes)) depth.  Only exchange_scale changes:
zero versus 0.25.  The check-only mode performs CPU data and forward checks
without training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gcm.multigraph import MultiGraphGCM  # noqa: E402

TASKS = ("sssp", "ecc", "diam", "charge", "energy")
SPLITS = ("train", "val", "test")
SCALES = (0.0, 0.25)
RAW_SCALE = 40.0
CHARGE_TABLE_SCALE = 1000.0
SSSP_DISTANCE_BINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0),
                      (30.0, 40.0), (40.0, float("inf")))


def _target_transform(task: str) -> str:
    if task in {"sssp", "ecc", "diam"}:
        return "divide_by_40"
    if task == "charge":
        return "native_charge_units"
    if task == "energy":
        return "log10_original_energy"
    raise ValueError(f"Unknown ECHO task: {task}")


def _loader_class():
    source = REPO / "external" / "ECHO" / "PyG_implementation" / "ECHOBenchmarkDataset.py"
    spec = importlib.util.spec_from_file_location("official_echo_loader", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import official ECHO loader: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ECHOBenchmark


def _seed(seed: int) -> None:
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
    if isinstance(value, (list, tuple)):
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


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def _atomic_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def _append_jsonl(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")
        handle.flush()


def _state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}


def _digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_task(root: Path, task: str, loader_cls):
    datasets = {
        split: loader_cls(root=str(root), task=task, split=split)
        for split in SPLITS
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError(f"{task}: one official split is empty")
    train = datasets["train"]
    input_dim = int(train.num_features)
    edge_dim = int(train.num_edge_features or 0) or None
    node_level = bool(train.is_node_level_task)
    max_nodes = 0
    total_nodes = {}
    for split, dataset in datasets.items():
        total = 0
        for index in range(len(dataset)):
            graph = dataset[index]
            if graph.x.ndim != 2 or graph.x.shape[1] != input_dim:
                raise ValueError(f"{task}/{split}/{index}: inconsistent features")
            if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
                raise ValueError(f"{task}/{split}/{index}: invalid edge_index")
            edge_attr = getattr(graph, "edge_attr", None)
            if edge_dim is None:
                if edge_attr is not None and edge_attr.numel():
                    raise ValueError(
                        f"{task}/{split}/{index}: unexpected edge_attr"
                    )
            else:
                if edge_attr is None or edge_attr.ndim != 2:
                    raise ValueError(
                        f"{task}/{split}/{index}: missing edge_attr"
                    )
                if edge_attr.shape != (graph.edge_index.shape[1], edge_dim):
                    raise ValueError(
                        f"{task}/{split}/{index}: edge_attr shape mismatch "
                        f"{tuple(edge_attr.shape)}"
                    )
                if edge_attr.is_floating_point() and not torch.isfinite(edge_attr).all():
                    raise ValueError(f"{task}/{split}/{index}: nonfinite edge_attr")
            if not torch.isfinite(graph.x).all() or not torch.isfinite(graph.y).all():
                raise ValueError(f"{task}/{split}/{index}: nonfinite arrays")
            nodes = int(graph.num_nodes)
            if nodes < 1:
                raise ValueError(f"{task}/{split}/{index}: empty graph")
            targets = int(graph.y.reshape(-1).numel())
            if targets != (nodes if node_level else 1):
                raise ValueError(f"{task}/{split}/{index}: target shape mismatch")
            total += nodes
            max_nodes = max(max_nodes, nodes)
        total_nodes[split] = total
    info = {
        "task": task,
        "loader": "ECHOBenchmark",
        "loader_source": str((REPO / "external" / "ECHO" / "PyG_implementation"
                              / "ECHOBenchmarkDataset.py").resolve()),
        "root": str(root.resolve()),
        "node_level_task": node_level,
        "input_dim": input_dim,
        "edge_dim": edge_dim,
        "split_graphs": {s: len(datasets[s]) for s in SPLITS},
        "split_nodes": total_nodes,
        "max_nodes": max_nodes,
        "full_max_stages": max(1, (max_nodes - 1).bit_length()),
        "official_target_scale": (RAW_SCALE if task in {"sssp", "ecc", "diam"}
                                   else None),
        "target_transform": _target_transform(task),
        "target_note": (
            "sssp/ecc/diam loader targets are divided by 40"
            if task in {"sssp", "ecc", "diam"}
            else ("charge targets are in native charge units"
                  if task == "charge"
                  else "energy targets are log10(original graph energy)")
        ),
    }
    return datasets, info


def _make_model(args, info: dict[str, Any], seed: int, scale: float):
    return MultiGraphGCM(
        input_dim=info["input_dim"], edge_dim=info["edge_dim"], output_dim=1,
        dim=args.dim, heads=args.heads, blocks=args.blocks,
        max_stages=info["full_max_stages"], mode=args.mode,
        exchange_scale=scale, dropout=args.dropout, angle=args.angle,
        stage_ff=False, graph_embed=True, graph_refresh=True,
        layout=args.layout, seed=seed, tie_pair_gate=args.tie_pair_gate,
    )


def _predict(model, batch, node_level: bool) -> torch.Tensor:
    output = model.predict_nodes(batch) if node_level else model(batch)
    return output.reshape(-1)


def _metrics(squared: float, absolute: float, target_sum: float,
             target_squared: float, count: int, task: str,
             task_squared: float, task_absolute: float,
             distance_absolute: list[float] | None = None,
             distance_counts: list[int] | None = None) -> dict[str, Any]:
    if count < 1:
        raise ValueError("Cannot score an empty split")
    mse = squared / count
    mae = absolute / count
    task_mse = task_squared / count
    task_mae = task_absolute / count
    denominator = target_squared - target_sum * target_sum / count
    r2 = 1.0 - squared / denominator if denominator > 0 else float("nan")
    metrics: dict[str, Any] = {
        "logMSE": math.log10(max(mse, np.finfo(float).tiny)),
        "normalized_MSE": mse,
        "normalized_MAE": mae,
        "task_MSE": task_mse,
        "task_MAE": task_mae,
        "R2": r2,
        "elements": count,
    }
    if task in {"sssp", "ecc", "diam"}:
        metrics["raw_MSE"] = task_mse
        metrics["raw_MAE"] = task_mae
    elif task == "charge":
        metrics["raw_MSE"] = task_mse
        metrics["raw_MAE"] = task_mae
        metrics["charge_raw_MSE"] = task_mse
        metrics["charge_raw_MAE"] = task_mae
        metrics["charge_MAE_x1000"] = task_mae * CHARGE_TABLE_SCALE
    elif task == "energy":
        metrics["raw_MSE"] = task_mse
        metrics["raw_MAE"] = task_mae
        metrics["energy_original_MSE"] = task_mse
        metrics["energy_original_MAE"] = task_mae
    else:
        raise ValueError(f"Unknown ECHO task: {task}")
    if distance_absolute is not None and distance_counts is not None:
        metrics["distance_binned_raw_MAE"] = {
            f"{lo:g}-{hi:g}": (
                distance_absolute[index] / distance_counts[index]
                if distance_counts[index] else None
            )
            for index, (lo, hi) in enumerate(SSSP_DISTANCE_BINS)
        }
        metrics["distance_binned_elements"] = {
            f"{lo:g}-{hi:g}": distance_counts[index]
            for index, (lo, hi) in enumerate(SSSP_DISTANCE_BINS)
        }
    return metrics


@torch.no_grad()
def _evaluate(model, loader, device, node_level: bool, task: str):
    model.eval()
    squared = absolute = target_sum = target_squared = 0.0
    task_squared = task_absolute = 0.0
    count = 0
    distance_absolute = [0.0] * len(SSSP_DISTANCE_BINS) if task == "sssp" else None
    distance_counts = [0] * len(SSSP_DISTANCE_BINS) if task == "sssp" else None
    for batch in loader:
        batch = batch.to(device)
        prediction = _predict(model, batch, node_level)
        target = batch.y.reshape(-1)
        if prediction.shape != target.shape:
            raise ValueError(f"prediction/target mismatch: {prediction.shape}/{target.shape}")
        delta = prediction.double() - target.double()
        squared += float(delta.square().sum().item())
        absolute += float(delta.abs().sum().item())
        if task in {"sssp", "ecc", "diam"}:
            task_prediction = prediction.double() * RAW_SCALE
            task_target = target.double() * RAW_SCALE
        elif task == "charge":
            task_prediction = prediction.double()
            task_target = target.double()
        elif task == "energy":
            # The official loader stores log10(original graph energy).
            # Compute task metrics after the inverse transform.
            task_prediction = torch.pow(10.0, prediction.double())
            task_target = torch.pow(10.0, target.double())
        else:
            raise ValueError(f"Unknown ECHO task: {task}")
        task_delta = task_prediction - task_target
        if not torch.isfinite(task_delta).all():
            raise FloatingPointError(f"Nonfinite {task} metric values")
        task_squared += float(task_delta.square().sum().item())
        task_absolute += float(task_delta.abs().sum().item())
        target_sum += float(target.double().sum().item())
        target_squared += float(target.double().square().sum().item())
        count += int(target.numel())
        if distance_absolute is not None and distance_counts is not None:
            raw_target = target.double() * RAW_SCALE
            raw_absolute = delta.abs() * RAW_SCALE
            for index, (lo, hi) in enumerate(SSSP_DISTANCE_BINS):
                mask = ((raw_target >= lo) if math.isinf(hi)
                        else ((raw_target >= lo) & (raw_target < hi)))
                distance_absolute[index] += float(raw_absolute[mask].sum().item())
                distance_counts[index] += int(mask.sum().item())
    return _metrics(squared, absolute, target_sum, target_squared, count, task,
                    task_squared, task_absolute, distance_absolute,
                    distance_counts)


def _train_epoch(model, loader, optimizer, device, node_level: bool,
                 clip_norm: float) -> float:
    model.train()
    squared = count = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict(model, batch, node_level)
        target = batch.y.reshape(-1)
        if prediction.shape != target.shape:
            raise ValueError(f"prediction/target mismatch: {prediction.shape}/{target.shape}")
        mse = F.mse_loss(prediction, target)
        if not torch.isfinite(mse):
            raise FloatingPointError("Nonfinite training MSE")
        loss = torch.log10(mse.clamp_min(torch.finfo(mse.dtype).tiny))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite log10 training MSE")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        squared += float(mse.detach().item()) * int(target.numel())
        count += int(target.numel())
    return math.log10(max(squared / count, np.finfo(float).tiny))


def _config(args, info: dict[str, Any], seed: int, scale: float):
    return {
        "runner": "run_echo",
        "benchmark": ("ECHO-Chem" if info["task"] in {"charge", "energy"}
                      else "ECHO-Synth"),
        "task": info["task"],
        "seed": seed,
        "exchange_scale": scale,
        "official_split": "full train/val/test",
        "model": {
            "class": "MultiGraphGCM", "input_dim": info["input_dim"],
            "edge_dim": info["edge_dim"], "output_dim": 1, "dim": args.dim,
            "heads": args.heads, "blocks": args.blocks, "mode": args.mode,
            "layout": args.layout, "max_stages": info["full_max_stages"],
            "max_nodes": info["max_nodes"], "stage_ff": False,
            "graph_embed": True, "graph_refresh": True, "dropout": args.dropout,
            "angle": args.angle, "tie_pair_gate": args.tie_pair_gate,
        },
        "training": {
            "loss": "log10(clamped MSE) on official loader targets",
            "validation_selection": "minimum global validation log10 normalized MSE",
            "batch_size": args.batch_size, "optimizer": "AdamW",
            "lr": args.lr, "weight_decay": args.weight_decay,
            "clip_norm": args.clip_norm, "epochs": args.epochs,
            "patience": args.patience, "num_workers": 0,
            "shuffle_seed": seed + 104729,
        },
        "raw_target_scale": (RAW_SCALE
                              if info["task"] in {"sssp", "ecc", "diam"}
                              else None),
        "charge_table_scale": (CHARGE_TABLE_SCALE
                                if info["task"] == "charge" else None),
    }


def _arm_dir(output: Path, task: str, seed: int, scale: float) -> Path:
    tag = f"{scale:g}".replace(".", "p")
    return output / task / f"seed_{seed}" / f"scale_{tag}"


def _run_arm(args, datasets, info, seed: int, scale: float, device):
    folder = _arm_dir(args.output, info["task"], seed, scale)
    result_path = folder / "result.json"
    best_path = folder / "best.pt"
    trace_path = folder / "epochs.jsonl"
    config = _config(args, info, seed, scale)
    if result_path.exists():
        if not args.resume:
            raise FileExistsError(f"Result exists; use --resume or another output: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["config"] != config:
            raise ValueError(f"Existing result has a different configuration: {result_path}")
        return result
    folder.mkdir(parents=True, exist_ok=True)
    _seed(seed)
    init_started = time.perf_counter()
    model = _make_model(args, info, seed, scale).to(device)
    _sync(device)
    init_seconds = time.perf_counter() - init_started
    digest = _digest(model)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed + 104729)
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size,
                              shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    node_level = bool(info["node_level_task"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    best_epoch = 0
    best_val = float("inf")
    best_metrics = None
    best_state = None
    no_improvement = 0
    epochs_completed = 0
    for epoch in range(1, args.epochs + 1):
        epochs_completed = epoch
        train_logmse = _train_epoch(model, train_loader, optimizer, device,
                                    node_level, args.clip_norm)
        val = _evaluate(model, val_loader, device, node_level, info["task"])
        if val["logMSE"] < best_val - 1e-12:
            best_val = val["logMSE"]
            best_epoch = epoch
            best_metrics = val
            best_state = _state_cpu(model)
            no_improvement = 0
            _atomic_save({"config": config, "initial_parameter_digest": digest,
                          "epoch": epoch, "val": val, "model": best_state}, best_path)
        else:
            no_improvement += 1
        _sync(device)
        row = {"epoch": epoch, "train_logMSE": train_logmse, "val": val,
               "best_epoch": best_epoch, "best_val_logMSE": best_val,
               "elapsed_seconds": time.perf_counter() - started}
        _append_jsonl(row, trace_path)
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(json.dumps({"task": info["task"], "seed": seed, "scale": scale,
                              "epoch": epoch, "train_logMSE": train_logmse,
                              "val_logMSE": val["logMSE"],
                              "best_epoch": best_epoch}), flush=True)
        if no_improvement >= args.patience:
            break
    if best_state is None or best_metrics is None:
        raise RuntimeError("No validation-selected ECHO checkpoint was produced")
    model.load_state_dict(best_state)
    test = _evaluate(model, test_loader, device, node_level, info["task"])
    _sync(device)
    peak_allocated = (float(torch.cuda.max_memory_allocated(device) / 2**20)
                      if device.type == "cuda" else None)
    peak_reserved = (float(torch.cuda.max_memory_reserved(device) / 2**20)
                     if device.type == "cuda" else None)
    train_seconds = time.perf_counter() - started
    result = {
        "status": "completed", "config": config, "dataset_info": info,
        "initial_parameter_digest": digest, "parameter_count": parameter_count,
        "compiled_graphs": model.compiled_graphs, "best_epoch": best_epoch,
        "epochs_completed": epochs_completed, "val": best_metrics, "test": test,
        "model_init_seconds": init_seconds, "training_seconds": train_seconds,
        "total_wall_seconds": init_seconds + train_seconds,
        "peak_cuda_allocated_mb": peak_allocated,
        "peak_cuda_reserved_mb": peak_reserved,
        "best_checkpoint": str(best_path.resolve()),
        "trace_path": str(trace_path.resolve()),
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "device": str(device),
            "cuda_device": (torch.cuda.get_device_name(device)
                            if device.type == "cuda" else None),
        },
    }
    _atomic_json(result, result_path)
    print(json.dumps({"status": "completed", "task": info["task"], "seed": seed,
                      "scale": scale, "best_epoch": best_epoch, "val": best_metrics,
                      "test": test, "training_seconds": train_seconds}), flush=True)
    return result


def _check_only(args, task_data):
    for task, (datasets, info) in task_data.items():
        _seed(args.seeds[0])
        graph = datasets["train"][0]
        original = (graph.x.clone(), graph.edge_index.clone(), graph.y.clone())
        batch = next(iter(DataLoader(datasets["train"][:2], batch_size=2))).to("cpu")
        model = _make_model(args, info, args.seeds[0], 0.25).cpu().eval()
        with torch.inference_mode():
            prediction = _predict(model, batch, bool(info["node_level_task"]))
        target = batch.y.reshape(-1)
        if prediction.shape != target.shape or not torch.isfinite(prediction).all():
            raise AssertionError(f"{task}: invalid CPU forward output")
        metric_loader = DataLoader(datasets["train"][:2], batch_size=2,
                                   shuffle=False, num_workers=0)
        check_metrics = _evaluate(model, metric_loader, torch.device("cpu"),
                                  bool(info["node_level_task"]), task)
        required_metrics = {"logMSE", "normalized_MAE", "raw_MAE"}
        if task == "charge":
            required_metrics.update({"charge_raw_MAE", "charge_MAE_x1000"})
        if task == "energy":
            required_metrics.add("energy_original_MAE")
        if not required_metrics.issubset(check_metrics):
            missing = sorted(required_metrics.difference(check_metrics))
            raise AssertionError(f"{task}: missing metrics {missing}")
        if any(not math.isfinite(float(check_metrics[name]))
               for name in required_metrics):
            raise AssertionError(f"{task}: nonfinite metric check")
        after = datasets["train"][0]
        if not (torch.equal(after.x, original[0]) and
                torch.equal(after.edge_index, original[1]) and
                torch.equal(after.y, original[2])):
            raise AssertionError(f"{task}: official arrays changed during check")
        print(json.dumps({"check": "passed", "task": task,
                          "node_level_task": info["node_level_task"],
                          "split_graphs": info["split_graphs"],
                          "max_nodes": info["max_nodes"],
                          "full_max_stages": info["full_max_stages"],
                          "batch_prediction_shape": list(prediction.shape),
                          "metric_check": {
                              name: check_metrics[name]
                              for name in sorted(required_metrics)
                          },
                          "official_arrays_unchanged": True, "device": "cpu"}),
              flush=True)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--data-root", type=Path,
                        default=REPO / "data" / "echo_benchmark" / "official")
    parser.add_argument("--output", type=Path, default=REPO / "results" / "echo_gcm")
    parser.add_argument("--seeds", "--seed", nargs="+", type=int, default=[0])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--angle", type=float, default=0.2)
    parser.add_argument("--mode", choices=["centered_j", "phase", "original"],
                        default="centered_j")
    parser.add_argument("--tie-pair-gate", action="store_true")
    parser.add_argument("--layout", choices=["graph_components", "graph", "input_order"],
                        default="graph_components")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        parser.error("batch-size, epochs, and patience must be positive")
    if args.dim < 1 or args.heads < 1 or args.dim % (2 * args.heads):
        parser.error("dim must be divisible by twice heads")
    if args.blocks < 1 or args.threads < 1 or args.clip_norm <= 0:
        parser.error("blocks, threads, and clip-norm must be positive")
    if not 0 <= args.dropout < 1 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("invalid dropout, learning rate, or weight decay")
    if args.print_every < 1:
        parser.error("print-every must be positive")
    return args


def main():
    args = _parse_args()
    torch.set_num_threads(args.threads)
    loader_cls = _loader_class()
    task_data = {}
    for task in args.tasks:
        datasets, info = _load_task(args.data_root.resolve(), task, loader_cls)
        task_data[task] = (datasets, info)
        _append_jsonl(info, args.output / "dataset_inventory.jsonl")
        print(json.dumps(info, sort_keys=True), flush=True)
    if args.check_only:
        _check_only(args, task_data)
        return
    device = torch.device(args.device)
    for task in args.tasks:
        datasets, info = task_data[task]
        for seed in args.seeds:
            results = [_run_arm(args, datasets, info, seed, scale, device)
                       for scale in SCALES]
            if results[0]["initial_parameter_digest"] != results[1]["initial_parameter_digest"]:
                raise RuntimeError(f"{task}/seed{seed}: arms have different initialization")
            zero, exchange = results
            paired = {
                "task": task, "seed": seed,
                "initial_parameter_digest": zero["initial_parameter_digest"],
                "config": zero["config"],
                "zero": {"best_epoch": zero["best_epoch"], "val": zero["val"],
                         "test": zero["test"]},
                "exchange_0p25": {"best_epoch": exchange["best_epoch"],
                                  "val": exchange["val"], "test": exchange["test"]},
                "delta_exchange_minus_zero": {
                    "val_logMSE": exchange["val"]["logMSE"] - zero["val"]["logMSE"],
                    "test_logMSE": exchange["test"]["logMSE"] - zero["test"]["logMSE"],
                    "val_raw_MAE": exchange["val"]["raw_MAE"] - zero["val"]["raw_MAE"],
                    "test_raw_MAE": exchange["test"]["raw_MAE"] - zero["test"]["raw_MAE"],
                    "val_raw_MSE": exchange["val"]["raw_MSE"] - zero["val"]["raw_MSE"],
                    "test_raw_MSE": exchange["test"]["raw_MSE"] - zero["test"]["raw_MSE"],
                },
            }
            if task == "charge":
                paired["delta_exchange_minus_zero"].update({
                    "val_charge_MAE_x1000": (
                        exchange["val"]["charge_MAE_x1000"]
                        - zero["val"]["charge_MAE_x1000"]
                    ),
                    "test_charge_MAE_x1000": (
                        exchange["test"]["charge_MAE_x1000"]
                        - zero["test"]["charge_MAE_x1000"]
                    ),
                })
            if task == "energy":
                paired["delta_exchange_minus_zero"].update({
                    "val_energy_original_MAE": (
                        exchange["val"]["energy_original_MAE"]
                        - zero["val"]["energy_original_MAE"]
                    ),
                    "test_energy_original_MAE": (
                        exchange["test"]["energy_original_MAE"]
                        - zero["test"]["energy_original_MAE"]
                    ),
                })
            paired_path = args.output / task / f"seed_{seed}" / "paired.json"
            _atomic_json(paired, paired_path)
            print(json.dumps({"paired_result": str(paired_path),
                              "delta_exchange_minus_zero":
                              paired["delta_exchange_minus_zero"]}), flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
