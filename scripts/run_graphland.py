"""Paired PureGCM experiments on official GraphLand full-graph node tasks.

This runner deliberately reuses ``external/GraphLand/dataset.py`` for the RL
split, one-hot encoding, feature transforms, label masking, and regression
target scaling. It is a GCM comparison protocol, not the GraphLand paper's
model or hyperparameter protocol. Test metrics are computed only after a
checkpoint has been selected by the validation primary metric.

Example::

    python scripts/run_graphland.py --datasets city-roads-M tolokers-2 \
        --seeds 0 1 2 --steps 1000 --device cuda --resume
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch import nn
from torch.nn import functional as F


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gcm.pure import PureGCM  # noqa: E402


def _official_dataset_class():
    source = REPO / "external" / "GraphLand" / "dataset.py"
    spec = importlib.util.spec_from_file_location("official_graphland_dataset", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official GraphLand loader: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Dataset


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _atomic_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _parameter_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _dataset_description(dataset: Any) -> dict[str, Any]:
    mask = lambda value: int(value.sum().item())
    target = dataset.targets
    result = {
        "nodes": int(dataset.features.shape[0]),
        "directed_edges_after_official_loader": int(dataset.graph.shape[1]),
        "features": int(dataset.features.shape[1]),
        "train_labeled": mask(dataset.train_mask),
        "val_labeled": mask(dataset.val_mask),
        "test_labeled": mask(dataset.test_mask),
        "unlabeled": int(torch.isnan(target).sum().item()),
        "official_task": dataset.task,
        "official_primary_metric": dataset.metric_name,
        "official_split": dataset.split,
        "feature_numerical_transform": getattr(dataset, "numerical_features_transform_name", None),
        "feature_fraction_transform": getattr(dataset, "fraction_features_transform_name", None),
        "target_transform": getattr(dataset, "regression_targets_transform_name", None),
    }
    if dataset.task == "binary_classification":
        result["positive_rate"] = {
            split: float(target[getattr(dataset, f"{split}_mask")].mean().item())
            for split in ("train", "val", "test")
        }
    return result


def _score(dataset: Any, logits: torch.Tensor, split: str) -> dict[str, float | None]:
    mask = getattr(dataset, f"{split}_mask").cpu().numpy().astype(bool)
    # GraphLand's default StandardScaler is constructed with copy=False.
    # inverse_transform otherwise mutates a CPU tensor through its NumPy view.
    raw = logits.detach().cpu().reshape(-1).numpy().copy()
    if dataset.task == "regression":
        prediction = dataset.regression_targets_transform.inverse_transform(raw[:, None]).reshape(-1)
        truth = dataset.targets_orig[mask]
        pred = prediction[mask]
        return {
            "R2": float(r2_score(truth, pred)),
            "MAE": float(mean_absolute_error(truth, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(truth, pred))),
        }
    truth = dataset.targets.cpu().numpy()[mask].astype(np.int64)
    pred = raw[mask]
    hard = (pred >= 0).astype(np.int64)
    try:
        roc_auc = float(roc_auc_score(truth, pred))
    except ValueError:
        roc_auc = None
    return {
        "AP": float(average_precision_score(truth, pred)),
        "ROC_AUC": roc_auc,
        "accuracy": float(accuracy_score(truth, hard)),
        "F1": float(f1_score(truth, hard, zero_division=0)),
    }


def _validation_score(dataset: Any, logits: torch.Tensor) -> float:
    metrics = _score(dataset, logits, "val")
    value = metrics[dataset.metric_name]
    if value is None or not math.isfinite(value):
        raise ValueError(f"nonfinite validation {dataset.metric_name}: {value}")
    return value


def _make_model(dataset: Any, args: argparse.Namespace, seed: int, exchange_scale: float) -> PureGCM:
    # PureGCM's constructor sizes a multiclass head from data.y; GraphLand
    # regression labels can contain NaNs, so provide a valid placeholder and
    # replace that head with a one-output task head immediately afterwards.
    proxy = SimpleNamespace(
        edge_index=dataset.graph,
        num_nodes=int(dataset.features.shape[0]),
        num_features=int(dataset.features.shape[1]),
        x=dataset.features,
        y=torch.zeros(1, dtype=torch.long),
    )
    model = PureGCM(
        proxy,
        dim=args.dim,
        heads=args.heads,
        blocks=args.blocks,
        mode=args.mode,
        dropout=args.dropout,
        layout=args.layout,
        stage_ff=False,
        angle=args.angle,
        graph_embed=True,
        seed=seed,
        graph_refresh=True,
        refresh_after_circuit=True,
        graph_innovation=args.graph_innovation,
        checkpoint_circuit=args.checkpoint_circuit,
        skip_padding_pairs=True,
        max_stages=args.max_stages,
        exchange_scale=exchange_scale,
        gate_normalize=args.gate_normalize,
        multiscale_reach=args.multiscale_reach,
        tie_pair_gate=args.tie_pair_gate,
    )
    model.classifier = nn.Sequential(nn.LayerNorm(args.dim), nn.Linear(args.dim, 1))
    return model


def _loss(dataset: Any, logits: torch.Tensor) -> torch.Tensor:
    mask = dataset.train_mask
    target = dataset.targets[mask].to(device=logits.device, dtype=logits.dtype)
    score = logits.reshape(-1)[mask]
    if dataset.task == "regression":
        return F.mse_loss(score, target)
    return F.binary_cross_entropy_with_logits(score, target)


def _initial_key(dataset_name: str, seed: int, args: argparse.Namespace) -> str:
    settings = {
        "dataset": dataset_name,
        "seed": seed,
        "split": "RL",
        "steps": args.steps,
        "eval_every": args.eval_every,
        "dim": args.dim,
        "heads": args.heads,
        "blocks": args.blocks,
        "max_stages": args.max_stages,
        "mode": args.mode,
        "stage_ff": False,
        "graph_innovation": args.graph_innovation,
        "append_log_degree": args.append_log_degree,
        "checkpoint_circuit": args.checkpoint_circuit,
        "dropout": args.dropout,
        "layout": args.layout,
        "angle": args.angle,
        "gate_normalize": args.gate_normalize,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "deterministic": args.deterministic,
    }
    if args.multiscale_reach:
        settings["multiscale_reach"] = True
    if args.tie_pair_gate:
        settings["tie_pair_gate"] = True
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]


def _train_one(
    dataset: Any,
    dataset_name: str,
    description: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
    exchange_scale: float,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    run_key = _initial_key(dataset_name, seed, args)
    arm = f"exchange{exchange_scale:g}"
    run_id = f"{dataset_name}__seed{seed}__{arm}__{run_key}"
    checkpoint_path = output / "checkpoints" / f"{run_id}.pt"
    prediction_path = output / "predictions" / f"{run_id}.pt"
    trace_path = output / "traces" / f"{run_id}.jsonl"
    config = {
        "split": "RL",
        "add_self_loops": False,
        "to_undirected": True,
        "node_embeddings": None,
        "preprocessing": "official GraphLand Dataset defaults",
        "model": "PureGCM",
        "mode": args.mode,
        "layout": args.layout,
        "dim": args.dim,
        "heads": args.heads,
        "blocks": args.blocks,
        "max_stages": args.max_stages,
        "stage_ff": False,
        "graph_embed": True,
        "graph_refresh": True,
        "refresh_after_circuit": True,
        "graph_innovation": args.graph_innovation,
        "append_log_degree": args.append_log_degree,
        "checkpoint_circuit": args.checkpoint_circuit,
        "skip_padding_pairs": True,
        "dropout": args.dropout,
        "angle": args.angle,
        "gate_normalize": args.gate_normalize,
        "exchange_scale": exchange_scale,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "steps": args.steps,
        "eval_every": args.eval_every,
        "checkpoint_selection": f"highest validation {dataset.metric_name}",
        "seed": seed,
        "deterministic": args.deterministic,
    }
    if args.multiscale_reach:
        config["multiscale_reach"] = True
    if args.tie_pair_gate:
        config["tie_pair_gate"] = True
    started = time.perf_counter()
    _seed_everything(seed, args.deterministic)
    build_started = time.perf_counter()
    model = _make_model(dataset, args, seed, exchange_scale)
    build_seconds = time.perf_counter() - build_started
    initial_parameter_digest = _parameter_digest(model)
    model = model.to(device)
    features = dataset.features.to(device)
    dataset.train_mask = dataset.train_mask.to(device)
    dataset.targets = dataset.targets.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_score = -math.inf
    best_step = None
    best_state = None
    step_times: list[float] = []
    val_eval_seconds = 0.0
    train_started = time.perf_counter()
    for step in range(1, args.steps + 1):
        step_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, dataset.graph)
        loss = _loss(dataset, logits)
        loss.backward()
        optimizer.step()
        _sync(device)
        step_times.append(time.perf_counter() - step_started)

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            eval_started = time.perf_counter()
            model.eval()
            with torch.inference_mode():
                val_logits = model(features, dataset.graph)
            score = _validation_score(dataset, val_logits)
            val_eval_seconds += time.perf_counter() - eval_started
            improved = score > best_score
            if improved:
                best_score = score
                best_step = step
                best_state = _cpu_state(model)
            trace_row = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                f"val_{dataset.metric_name}": score,
                "best_step": best_step,
                "best_score": best_score,
                "step_seconds": step_times[-1],
                "elapsed_seconds": time.perf_counter() - train_started,
                "peak_cuda_allocated_mb_so_far": (
                    float(torch.cuda.max_memory_allocated(device) / 2**20)
                    if device.type == "cuda" else None
                ),
            }
            _append_jsonl(trace_path, trace_row)
            if step == 1 or step % args.print_every == 0 or step == args.steps:
                print(json.dumps({"run_id": run_id, **trace_row}, ensure_ascii=False), flush=True)
            del val_logits
        del logits, loss
    _sync(device)
    train_seconds = time.perf_counter() - train_started
    if best_state is None or best_step is None:
        raise RuntimeError("no validation checkpoint produced")
    peak_training_mb = (
        float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None
    )

    model.load_state_dict(best_state)
    model.eval()
    _sync(device)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        selected_logits = model(features, dataset.graph).reshape(-1).detach().cpu()
    _sync(device)
    inference_seconds = time.perf_counter() - inference_started
    # Every reported split comes from this same validation-selected checkpoint.
    metrics = {split: _score(dataset, selected_logits, split) for split in ("train", "val", "test")}
    official_metrics = dataset.compute_metrics_transductive(selected_logits.clone())
    for split in ("train", "val", "test"):
        own = metrics[split][dataset.metric_name]
        reference = official_metrics[f"{split} {dataset.metric_name}"]
        if not np.isclose(own, reference, rtol=1e-5, atol=1e-4):
            raise AssertionError(
                f"metric mismatch with official GraphLand loader on {split}: "
                f"runner={own}, official={reference}"
            )

    checkpoint = {
        "model": best_state,
        "run_id": run_id,
        "config": config,
        "dataset": description,
        "best_step": best_step,
        "best_val_primary": best_score,
        "metrics_at_selected_checkpoint": metrics,
    }
    _atomic_save(checkpoint, checkpoint_path)
    if dataset.task == "regression":
        predicted_original = dataset.regression_targets_transform.inverse_transform(
            selected_logits.numpy().copy()[:, None]
        ).reshape(-1)
        labels_original = dataset.targets_orig
    else:
        predicted_original = selected_logits.numpy()
        labels_original = dataset.targets.detach().cpu().numpy()
    _atomic_save(
        {
            "run_id": run_id,
            "logits_or_scaled_prediction": selected_logits,
            "prediction_original_units": torch.from_numpy(np.asarray(predicted_original)),
            "target_original_units": torch.from_numpy(np.asarray(labels_original)),
            "train_mask": dataset.train_mask.cpu(),
            "val_mask": dataset.val_mask.cpu(),
            "test_mask": dataset.test_mask.cpu(),
            "best_step": best_step,
            "checkpoint_selection": f"validation {dataset.metric_name}",
        },
        prediction_path,
    )
    steady_steps = step_times[min(10, len(step_times) - 1):]
    result = {
        "status": "completed",
        "run_id": run_id,
        "dataset": dataset_name,
        "seed": seed,
        "exchange_scale": exchange_scale,
        "config": config,
        "dataset_description": description,
        "initial_parameter_digest": initial_parameter_digest,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "padded_nodes": model.padded,
        "best_step": best_step,
        "best_val_primary": best_score,
        "metrics": metrics,
        "official_primary_metrics": official_metrics,
        "build_and_compile_seconds": build_seconds,
        "training_seconds": train_seconds,
        "validation_evaluation_seconds": val_eval_seconds,
        "mean_step_ms_after_warmup": 1000 * float(np.mean(steady_steps)),
        "median_step_ms_after_warmup": 1000 * float(np.median(steady_steps)),
        "selected_inference_seconds": inference_seconds,
        "peak_cuda_allocated_mb_training": peak_training_mb,
        "total_wall_seconds": time.perf_counter() - started,
        "checkpoint_path": str(checkpoint_path),
        "predictions_path": str(prediction_path),
        "trace_path": str(trace_path),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    _append_jsonl(output / "runs.jsonl", result)
    return result


def _completed_ids(output: Path) -> set[str]:
    path = output / "runs.jsonl"
    if not path.exists():
        return set()
    found = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status") == "completed":
                found.add(row["run_id"])
    return found


def _check_relabeling(
    dataset: Any,
    dataset_name: str,
    args: argparse.Namespace,
    seed: int,
    exchange_scale: float,
    device: torch.device,
    output: Path,
) -> None:
    """Evaluate a fixed checkpoint under random, equivalent node ID changes.

    The learned parameters are transferred; graph-dependent buffers are
    freshly compiled on the relabeled graph. No labels influence the compiler.
    """
    checkpoint_args = copy.copy(args)
    checkpoint_args.layout = args.checkpoint_layout or args.layout
    run_key = _initial_key(dataset_name, seed, checkpoint_args)
    run_id = f"{dataset_name}__seed{seed}__exchange{exchange_scale:g}__{run_key}"
    checkpoint_path = output / "checkpoints" / f"{run_id}.pt"
    prediction_path = output / "predictions" / f"{run_id}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    predictions = torch.load(prediction_path, map_location="cpu", weights_only=False)
    if checkpoint_args.layout == args.layout:
        reference = predictions["logits_or_scaled_prediction"].reshape(-1)
        reference_val = checkpoint["metrics_at_selected_checkpoint"]["val"]
        reference_test = checkpoint["metrics_at_selected_checkpoint"]["test"]
    else:
        # Probe whether a new compiler itself has stable semantics, using
        # exactly the learned weights of the old checkpoint and no retraining.
        _seed_everything(seed, args.deterministic)
        reference_model = _make_model(dataset, args, seed, exchange_scale)
        learned = set(dict(reference_model.named_parameters()))
        reference_model.load_state_dict(
            {name: checkpoint["model"][name] for name in learned}, strict=False
        )
        reference_model = reference_model.to(device).eval()
        with torch.inference_mode():
            reference = reference_model(dataset.features.to(device), dataset.graph).reshape(-1).cpu()
        reference_val = _score(dataset, reference, "val")
        reference_test = _score(dataset, reference, "test")
        del reference_model
    edges = dataset.graph.cpu().numpy()
    nodes = int(dataset.features.shape[0])
    for index in range(args.relabel_permutations):
        rng = np.random.default_rng(args.relabel_seed + index)
        order = rng.permutation(nodes)
        old_to_new = np.empty(nodes, dtype=np.int64)
        old_to_new[order] = np.arange(nodes)
        new_edges = torch.from_numpy(old_to_new[edges].copy())
        new_features = dataset.features[torch.from_numpy(order.copy())]
        relabeled = SimpleNamespace(graph=new_edges, features=new_features)
        _seed_everything(seed, args.deterministic)
        model = _make_model(relabeled, args, seed, exchange_scale)
        learned_names = set(dict(model.named_parameters()))
        saved_state = checkpoint["model"]
        missing_parameters = learned_names - saved_state.keys()
        if missing_parameters:
            raise ValueError(f"checkpoint misses learned parameters: {sorted(missing_parameters)}")
        # load_state_dict(strict=False) leaves graph buffers from the newly
        # compiled permutation intact while restoring exactly the learned map.
        model.load_state_dict({name: saved_state[name] for name in learned_names}, strict=False)
        model = model.to(device).eval()
        with torch.inference_mode():
            changed = model(new_features.to(device), new_edges).reshape(-1).cpu()
        restored = torch.empty_like(changed)
        restored[torch.from_numpy(order.copy())] = changed
        delta = (restored - reference).abs()
        val = _score(dataset, restored, "val")
        test = _score(dataset, restored, "test")
        row = {
            "run_id": run_id,
            "checkpoint_layout": checkpoint_args.layout,
            "compiled_layout": args.layout,
            "permutation_index": index,
            "permutation_seed": args.relabel_seed + index,
            "mean_absolute_scaled_prediction_change": float(delta.mean()),
            "median_absolute_scaled_prediction_change": float(delta.median()),
            "max_absolute_scaled_prediction_change": float(delta.max()),
            "val_metrics": val,
            "test_metrics": test,
            "reference_val_metrics": reference_val,
            "reference_test_metrics": reference_test,
            "checkpoint_path": str(checkpoint_path),
        }
        _append_jsonl(output / "relabel_checks.jsonl", row)
        print(json.dumps(_jsonable({"status": "relabel_checked", **row}), ensure_ascii=False), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["city-roads-M", "tolokers-2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--exchange-scales", nargs="+", type=float, default=[0.0, 0.25])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--max-stages", default="full",
                        help="full butterfly depth (default) or a positive stage count; use 3 for the locality ablation")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--layout", default="graph_components")
    parser.add_argument("--mode", choices=["phase", "original", "centered_j"], default="phase",
                        help="phase, the original J rotation, or its pair-mean-preserving centered form")
    parser.add_argument("--checkpoint-layout", default=None,
                        help="in relabel mode, use a checkpoint trained under this layout")
    parser.add_argument("--angle", type=float, default=0.2)
    parser.add_argument("--multiscale-reach", action="store_true", default=False,
                        help="initialize existing centered-J heads with different analytic Hamming reach budgets")
    parser.add_argument("--tie-pair-gate", action="store_true", default=False,
                        help="reuse one centered-J content gate across all pair stages in each block")
    parser.add_argument("--gate-normalize", action="store_true")
    parser.add_argument("--graph-innovation", action="store_true",
                        help="route state minus one-hop mean through the unchanged pair circuit")
    parser.add_argument("--append-log-degree", action="store_true",
                        help="append one graph-derived log-degree feature to both paired arms; separate augmented protocol")
    parser.add_argument("--checkpoint-circuit", action="store_true",
                        help="recompute unchanged pair circuits in backward to reduce peak activation memory")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, default=REPO / "results" / "graphland_gcm_rl")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--data-check-only", action="store_true")
    parser.add_argument("--relabel-check-only", action="store_true",
                        help="evaluate saved checkpoints after node relabeling; all training flags must match")
    parser.add_argument("--relabel-permutations", type=int, default=3)
    parser.add_argument("--relabel-seed", type=int, default=441)
    args = parser.parse_args()
    if args.max_stages == "full":
        args.max_stages = None
    else:
        try:
            args.max_stages = int(args.max_stages)
        except ValueError:
            parser.error("max-stages must be 'full' or a positive integer")
        if args.max_stages < 1:
            parser.error("max-stages must be 'full' or a positive integer")
    if args.steps < 1 or args.eval_every < 1 or args.print_every < 1:
        parser.error("steps, eval-every and print-every must be positive")
    if not args.exchange_scales or any(not 0 <= scale <= 1 for scale in args.exchange_scales):
        parser.error("exchange scales must be in [0, 1]")
    if args.relabel_permutations < 1:
        parser.error("relabel-permutations must be positive")
    return args


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.threads)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # The official Dataset loader resolves files under relative data/{name}.
    os.chdir(REPO)
    official_dataset = _official_dataset_class()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    done = _completed_ids(output) if args.resume else set()
    for dataset_name in args.datasets:
        load_started = time.perf_counter()
        dataset = official_dataset(
            name=dataset_name,
            split="RL",
            add_self_loops=False,
            to_undirected=True,
            node_embeddings=None,
            device="cpu",
            use_pyg=True,
        )
        if not dataset.transductive or dataset.task not in ("binary_classification", "regression"):
            raise ValueError(f"unsupported GraphLand task for this runner: {dataset_name}: {dataset.task}")
        if args.append_log_degree:
            degree = torch.bincount(dataset.graph[0], minlength=dataset.features.shape[0]).float()
            log_degree = degree.log1p()
            log_degree = (log_degree - log_degree.mean()) / log_degree.std().clamp_min(1e-6)
            dataset.features = torch.cat([dataset.features, log_degree[:, None]], dim=1)
        description = _dataset_description(dataset)
        description["added_feature"] = "standardized log(1+degree)" if args.append_log_degree else None
        description["official_load_seconds"] = time.perf_counter() - load_started
        print(json.dumps({"dataset": dataset_name, **description}, ensure_ascii=False), flush=True)
        _append_jsonl(output / "dataset_inventory.jsonl", {"dataset": dataset_name, **description})
        if args.data_check_only:
            continue
        if args.relabel_check_only:
            for seed in args.seeds:
                for scale in args.exchange_scales:
                    _check_relabeling(dataset, dataset_name, args, seed, scale, device, output)
            continue
        for seed in args.seeds:
            paired_initial_digest = None
            for scale in args.exchange_scales:
                run_key = _initial_key(dataset_name, seed, args)
                run_id = f"{dataset_name}__seed{seed}__exchange{scale:g}__{run_key}"
                if args.resume and run_id in done:
                    print(json.dumps({"status": "skipped_completed", "run_id": run_id}), flush=True)
                    continue
                try:
                    result = _train_one(dataset, dataset_name, description, args, seed, scale, device, output)
                    if paired_initial_digest is None:
                        paired_initial_digest = result["initial_parameter_digest"]
                    elif paired_initial_digest != result["initial_parameter_digest"]:
                        raise AssertionError("paired arms did not start from identical parameters")
                    print(json.dumps({"status": "completed", "run_id": run_id,
                                      "best_step": result["best_step"],
                                      "val": result["metrics"]["val"],
                                      "test": result["metrics"]["test"]}, ensure_ascii=False), flush=True)
                except Exception as error:
                    row = {"status": "error", "run_id": run_id, "dataset": dataset_name,
                           "seed": seed, "exchange_scale": scale, "error": str(error),
                           "traceback": traceback.format_exc()}
                    _append_jsonl(output / "runs.jsonl", row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    raise
                finally:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
