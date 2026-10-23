"""Matched original-J GCM experiment on NeurIPS 2025 2Dele.

The released ``train_set2D.pth`` / ``val_set2D.pth`` / ``test_set2D.pth``
files fix 120/15/15 samples. Their ``p_x`` and ``d_x`` are observed primal
vertex and dual face inputs; only ``p_y`` supplies the four target channels.
The primary metric exactly follows the released ``main.py`` criterion:
sum of four squared errors divided by the number of all 105 primal vertices,
then averaged over loader batches (including the smaller final batch).

``vertex_mean`` attaches the mean of incident face inputs to each vertex. It
is a deterministic but lossy representation of the 173 face inputs.
``primal_dual`` instead keeps all 105 vertex and 173 face inputs as separate
real circuit states and scores the 105 vertex outputs. Both use mesh topology
only for the label-free compiler. ``--geometry-input`` additionally gives the
existing node/edge encoders fixed released positions, boundary flags, volumes,
and relative edge geometry; it changes no J switches or learned modules.
Zero and original-J arms share initialization, optimizer, sampling, features,
compilation, readout, and validation selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gcm.architecture import compile_interfaces  # noqa: E402
from gcm.multigraph import MultiGraphGCM  # noqa: E402

OFFICIAL_ROOT = REPO / "external/Higher-Order-Differential-Topology-aware-GNN"
SPLIT_SIZE = {"train": 120, "val": 15, "test": 15}
SCALES = (0.0, 0.25)


def _seed_everything(seed: int) -> None:
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
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
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2,
                                    sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")
        handle.flush()


def _state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().cpu().clone()
            for key, tensor in model.state_dict().items()}


def _parameter_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_legacy_split(path: Path, official_root: Path):
    # datagen2D.py saved its class while executing as __main__. The class name
    # must be registered there for the released pickle to resolve correctly.
    # Only load the author's local files, never arbitrary downloaded pickles.
    sys.path.insert(0, str(official_root.resolve()))
    official_datagen = importlib.import_module("datagen2D")
    main_module = sys.modules["__main__"]
    previous = getattr(main_module, "HOGraphDataset", None)
    main_module.HOGraphDataset = official_datagen.HOGraphDataset
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        if previous is None:
            delattr(main_module, "HOGraphDataset")
        else:
            main_module.HOGraphDataset = previous


def _validate_official_graph(graph: Any, name: str) -> None:
    if graph.p_x.shape != (105, 5) or graph.d_x.shape != (173, 5) \
            or graph.p_y.shape != (105, 4):
        raise ValueError(f"{name}: unexpected 2Dele primal/dual dimensions")
    if len(graph.adjs) < 9 or graph.adjs[6].shape != (277, 2) \
            or graph.adjs[7].shape != (173, 3) \
            or graph.adjs[8].shape != (277, 2):
        raise ValueError(f"{name}: missing released mesh incidence arrays")
    if graph.bdry_markers[0].shape != (105, 1):
        raise ValueError(f"{name}: invalid primal boundary mask")
    if len(graph.poses) < 2 or graph.poses[0].shape != (105, 2) \
            or graph.poses[1].shape != (173, 2) \
            or len(graph.vols) < 4 or graph.vols[1].numel() != 173 \
            or graph.vols[3].numel() != 105 \
            or graph.bdry_markers[2].shape != (173, 1):
        raise ValueError(f"{name}: invalid released geometry arrays")
    for tensor in (graph.p_x, graph.d_x, graph.p_y):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name}: nonfinite input or target")
    for tensor in (graph.poses[0], graph.poses[1], graph.vols[1],
                   graph.vols[3]):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name}: nonfinite mesh geometry")
    triangles = graph.adjs[7].long()
    edges = graph.adjs[6].long()
    dual_edges = graph.adjs[8].long()
    if bool((triangles < 0).any()) or bool((triangles >= 105).any()) \
            or bool((edges < 0).any()) or bool((edges >= 105).any()) \
            or bool((dual_edges < 0).any()) or bool((dual_edges >= 173).any()):
        raise ValueError(f"{name}: mesh references a nonexistent vertex or face")
    mask = graph.bdry_markers[0].reshape(-1)
    if not torch.equal(graph.p_x[mask, :4], graph.p_y[mask]):
        raise ValueError(f"{name}: observed primal boundary values differ from target")


def _undirected_edges(pairs: torch.Tensor) -> torch.Tensor:
    pairs = pairs.long().T.contiguous()
    return torch.cat((pairs, pairs.flip(0)), dim=1)


def _incident_face_mean(d_x: torch.Tensor,
                        triangles: torch.Tensor) -> torch.Tensor:
    """Mean of observed face channels over the triangles incident on a vertex."""
    vertices = triangles.long().reshape(-1)
    counts = torch.bincount(vertices, minlength=105)
    if bool((counts == 0).any()):
        raise ValueError("2Dele mesh has an isolated primal vertex")
    incident = d_x.repeat_interleave(3, dim=0)
    summed = d_x.new_zeros((105, d_x.shape[1]))
    summed.index_add_(0, vertices, incident)
    return summed / counts.unsqueeze(-1)


def _prepare_graph(graph: Any, representation: str,
                   geometry_input: bool = False) -> Data:
    """Use observed fields and fixed incidence; never inspect d_y or labels for x."""
    p_x = graph.p_x.float()
    d_x = graph.d_x.float()
    triangles = graph.adjs[7].long()
    primal_edges = _undirected_edges(graph.adjs[6])
    if representation == "vertex_mean":
        x = torch.cat((p_x, _incident_face_mean(d_x, triangles)), dim=1)
        edge_index = primal_edges
        y = graph.p_y.float().clone()
        score_mask = torch.ones(105, dtype=torch.bool)
        boundary_mask = graph.bdry_markers[0].reshape(-1).bool().clone()
        edge_types = torch.zeros(edge_index.shape[1], dtype=torch.long)
        positions = graph.poses[0].float()
        volumes = graph.vols[3].float().reshape(105, 1)
    elif representation == "primal_dual":
        # The two typed feature slots are disjoint. Consequently p_x and d_x
        # can be reconstructed exactly from x, including each individual face.
        x = torch.cat((
            torch.cat((p_x, p_x.new_zeros((105, 5)),
                       p_x.new_ones((105, 1))), dim=1),
            torch.cat((d_x.new_zeros((173, 5)), d_x,
                       d_x.new_zeros((173, 1))), dim=1),
        ), dim=0)
        dual_edges = _undirected_edges(graph.adjs[8]) + 105
        face_ids = (torch.arange(173, dtype=torch.long)
                    .repeat_interleave(3) + 105)
        vertices = triangles.reshape(-1)
        cross = torch.stack((vertices, face_ids))
        edge_index = torch.cat((primal_edges, dual_edges, cross,
                                cross.flip(0)), dim=1)
        edge_types = torch.cat((
            torch.zeros(primal_edges.shape[1], dtype=torch.long),
            torch.ones(dual_edges.shape[1], dtype=torch.long),
            torch.full((2 * cross.shape[1],), 2, dtype=torch.long),
        ))
        positions = torch.cat((graph.poses[0].float(),
                               graph.poses[1].float()), dim=0)
        volumes = torch.cat((graph.vols[3].float().reshape(105, 1),
                             graph.vols[1].float().reshape(173, 1)), dim=0)
        y = torch.cat((graph.p_y.float(), x.new_zeros((173, 4))), dim=0)
        score_mask = torch.cat((torch.ones(105, dtype=torch.bool),
                                torch.zeros(173, dtype=torch.bool)))
        boundary_mask = torch.cat((graph.bdry_markers[0].reshape(-1).bool(),
                                   graph.bdry_markers[2].reshape(-1).bool()))
    else:
        raise ValueError(f"Unknown representation: {representation}")
    edge_attr = None
    if geometry_input:
        # Fixed geometry is the same for every released sample. Normalizing
        # volumes by this mesh's own mean needs neither fields nor labels.
        relative_volume = volumes / volumes.mean().clamp_min(1e-12)
        x = torch.cat((x, positions, boundary_mask.float().unsqueeze(-1),
                       relative_volume), dim=1)
        source, target = edge_index
        displacement = positions[target] - positions[source]
        length = torch.linalg.vector_norm(displacement, dim=1, keepdim=True)
        kinds = torch.nn.functional.one_hot(edge_types, num_classes=3).float()
        edge_attr = torch.cat((displacement, length, kinds), dim=1)
    return Data(x=x, edge_index=edge_index.contiguous(), y=y,
                edge_attr=edge_attr, score_mask=score_mask,
                boundary_mask=boundary_mask)


def _load_data(data_root: Path, representation: str,
               geometry_input: bool = False
               ) -> tuple[dict[str, list[Data]], dict[str, Any], torch.Tensor]:
    root = data_root.resolve()
    if root.name != "2Dele":
        raise ValueError("Expected the released data/2Dele directory")
    official_root = root.parent.parent
    files = {split: root / f"{split}_set2D.pth" for split in SPLIT_SIZE}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing official fixed split: {path}")
    splits: dict[str, list[Data]] = {}
    topology = None
    geometry = None
    boundary = None
    fingerprints: set[str] = set()
    for split, expected in SPLIT_SIZE.items():
        source = _load_legacy_split(files[split], official_root)
        if len(source) != expected:
            raise ValueError(f"{split}: expected {expected}, found {len(source)}")
        converted = []
        for index in range(expected):
            graph = source[index]
            _validate_official_graph(graph, f"{split}[{index}]")
            entry = _prepare_graph(graph, representation, geometry_input)
            if topology is None:
                topology = entry.edge_index
                geometry = entry.edge_attr
                boundary = entry.boundary_mask
            elif not torch.equal(topology, entry.edge_index) \
                    or not torch.equal(boundary, entry.boundary_mask) \
                    or (geometry_input and not torch.equal(geometry,
                                                           entry.edge_attr)):
                raise ValueError("2Dele mesh/topology/boundary changed across samples")
            fingerprint = hashlib.sha256(
                graph.p_x.contiguous().numpy().tobytes()
                + graph.d_x.contiguous().numpy().tobytes()).hexdigest()
            if fingerprint in fingerprints:
                raise ValueError(f"Duplicate observed sample: {split}[{index}]")
            fingerprints.add(fingerprint)
            converted.append(entry)
        splits[split] = converted
    first = splits["train"][0]
    mapping, _, _ = compile_interfaces(
        first.edge_index, first.num_nodes, layout="graph_components",
        alignment=False, seed=0)
    mapping = mapping.long().cpu()
    if mapping.numel() != first.num_nodes or mapping.unique().numel() != first.num_nodes:
        raise ValueError("Compiler did not assign every real node one unique wire")
    padded = 1 << (first.num_nodes - 1).bit_length()
    if bool((mapping < 0).any()) or bool((mapping >= padded).any()):
        raise ValueError("Compiler produced a wire outside the padded circuit")
    info = {
        "benchmark": "NeurIPS 2025 DEC-HOGNN 2Dele",
        "paper_table_2_reference_test_score": {"mean": 0.623, "std": 0.058,
                                                "source": "external published reference, not this run"},
        "official_data_root": str(root),
        "official_split_files_sha256": {split: _sha256(path)
                                        for split, path in files.items()},
        "fixed_split_sizes": SPLIT_SIZE,
        "representation": representation,
        "run_variant": representation + ("_geometry" if geometry_input
                                          else "_fields"),
        "geometry_input": geometry_input,
        "representation_note": (
            "Each primal vertex gets p_x and the unweighted mean of d_x over incident "
            "triangles; individual face details are lost."
            if representation == "vertex_mean" else
            "Each primal vertex and dual face is a separate typed real wire; all "
            "p_x and d_x values are preserved. Only primal outputs are scored."),
        "learned_input": (
            "released p_x, d_x, positions, primal/dual boundary markers, "
            "primal/dual face volumes, and edge relative vector/length/type"
            if geometry_input else
            "released p_x and d_x only; no positions, labels, or d_y"),
        "geometry_note": (
            "Node position, boundary indicator, and within-mesh-mean-scaled "
            "dual or primal face volume enter the existing node encoder; relative edge "
            "displacement, length, and primal/dual/incidence type enter the "
            "existing edge encoder. No new model module or target statistic."
            if geometry_input else None),
        "compiler_input": "released fixed primal/dual mesh adjacency only; no labels",
        "target": "released p_y[:,0:4] on all 105 primal vertices",
        "score": "mean over batches of sum of four squared errors / 105 primal vertices",
        "nodes_per_graph": int(first.num_nodes),
        "scored_nodes_per_graph": 105,
        "padded_wires_per_graph": padded,
        "max_stages": padded.bit_length() - 1,
        "feature_dim": int(first.x.shape[1]),
        "edge_feature_dim": (int(first.edge_attr.shape[1])
                             if first.edge_attr is not None else None),
        "edges_per_graph_directed": int(first.edge_index.shape[1]),
        "primal_boundary_nodes": int((first.boundary_mask
                                      & first.score_mask).sum().item()),
        "sample_fingerprints_unique": len(fingerprints),
        "compiler_mapping_sha256": hashlib.sha256(
            mapping.numpy().tobytes()).hexdigest(),
        "official_preprocessing_note": (
            "The author released already normalized split pickles; datagen2D.py "
            "normalizes the full pre-split field array when regenerating them. "
            "This runner introduces no additional normalization."),
    }
    return splits, info, mapping


class _Metrics:
    def __init__(self):
        self.nodes = 0
        self.batches = 0
        self.batch_score_sum = 0.0
        self.sse = torch.zeros(4, dtype=torch.float64)
        self.sae = torch.zeros(4, dtype=torch.float64)
        self.target_sum = torch.zeros(4, dtype=torch.float64)
        self.target_sum2 = torch.zeros(4, dtype=torch.float64)
        self.region_sse = {"interior": 0.0, "boundary": 0.0}
        self.region_sae = {"interior": 0.0, "boundary": 0.0}
        self.region_nodes = {"interior": 0, "boundary": 0}

    def update(self, prediction: torch.Tensor, batch: Data) -> None:
        if prediction.shape != batch.y.shape or prediction.shape[1] != 4:
            raise ValueError("Model must return four values for every real node")
        score_mask = batch.score_mask
        error = (prediction[score_mask] - batch.y[score_mask]).double()
        target = batch.y[score_mask].double()
        nodes = error.shape[0]
        if nodes < 1:
            raise ValueError("No primal vertices in a score batch")
        self.nodes += nodes
        self.batches += 1
        squared = error.square()
        self.batch_score_sum += float(squared.sum().item()) / nodes
        self.sse += squared.sum(dim=0).cpu()
        self.sae += error.abs().sum(dim=0).cpu()
        self.target_sum += target.sum(dim=0).cpu()
        self.target_sum2 += target.square().sum(dim=0).cpu()
        boundary = batch.boundary_mask[score_mask]
        for region, mask in (("boundary", boundary), ("interior", ~boundary)):
            self.region_nodes[region] += int(mask.sum().item())
            self.region_sse[region] += float(squared[mask].sum().item())
            self.region_sae[region] += float(error[mask].abs().sum().item())

    def result(self) -> dict[str, Any]:
        if not self.nodes or not self.batches:
            raise ValueError("Empty 2Dele metric accumulator")
        centered_sst = (self.target_sum2
                        - self.target_sum.square() / self.nodes).sum().item()
        result: dict[str, Any] = {
            "score": self.batch_score_sum / self.batches,
            "score_node_weighted": self.sse.sum().item() / self.nodes,
            "MSE_scalar": self.sse.sum().item() / (4 * self.nodes),
            "MAE_scalar": self.sae.sum().item() / (4 * self.nodes),
            "E_MSE_per_node": self.sse[:2].sum().item() / self.nodes,
            "D_MSE_per_node": self.sse[2:].sum().item() / self.nodes,
            "E_MAE_scalar": self.sae[:2].sum().item() / (2 * self.nodes),
            "D_MAE_scalar": self.sae[2:].sum().item() / (2 * self.nodes),
            "channel_MSE": (self.sse / self.nodes).tolist(),
            "channel_MAE": (self.sae / self.nodes).tolist(),
            "R2_four_channel": (1.0 - self.sse.sum().item() / centered_sst
                                if centered_sst > 0 else None),
            "scored_nodes": self.nodes,
            "batches": self.batches,
        }
        result["regions"] = {
            region: {"nodes": self.region_nodes[region],
                     "score_per_node": (self.region_sse[region]
                                        / self.region_nodes[region]
                                        if self.region_nodes[region] else None),
                     "MAE_scalar": (self.region_sae[region]
                                    / (4 * self.region_nodes[region])
                                    if self.region_nodes[region] else None)}
            for region in ("interior", "boundary")
        }
        return result


def _predict(model: MultiGraphGCM, batch: Data,
             mapping: torch.Tensor) -> torch.Tensor:
    graphs = int(batch.ptr.numel() - 1)
    return model.predict_nodes(batch, mappings=[mapping] * graphs)


@torch.no_grad()
def _evaluate(model: MultiGraphGCM, loader: DataLoader,
              mapping: torch.Tensor, device: torch.device) -> dict[str, Any]:
    model.eval()
    _sync(device)
    started = time.perf_counter()
    metrics = _Metrics()
    for batch in loader:
        batch = batch.to(device)
        prediction = _predict(model, batch, mapping)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite 2Dele prediction")
        metrics.update(prediction, batch)
    _sync(device)
    result = metrics.result()
    result["seconds"] = time.perf_counter() - started
    return result


def _train_epoch(model: MultiGraphGCM, loader: DataLoader,
                 optimizer: torch.optim.Optimizer, mapping: torch.Tensor,
                 device: torch.device, clip_norm: float) -> float:
    model.train()
    node_sse = 0.0
    node_count = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict(model, batch, mapping)
        error = prediction[batch.score_mask] - batch.y[batch.score_mask]
        loss = error.square().sum(dim=1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite 2Dele training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        nodes = error.shape[0]
        node_sse += float(loss.detach().item()) * nodes
        node_count += nodes
    return node_sse / node_count


def _make_model(args: argparse.Namespace, info: dict[str, Any],
                seed: int, scale: float) -> MultiGraphGCM:
    return MultiGraphGCM(
        input_dim=info["feature_dim"], edge_dim=info["edge_feature_dim"],
        output_dim=4,
        dim=args.dim, heads=args.heads, blocks=args.blocks,
        max_stages=info["max_stages"], mode="original",
        exchange_scale=scale, dropout=args.dropout, angle=args.angle,
        stage_ff=False, graph_embed=True, graph_refresh=True,
        layout="graph_components", seed=seed,
        relay_padding=args.relay_padding, edge_weight_mode="symmetric",
    )


def _config(args: argparse.Namespace, info: dict[str, Any],
            seed: int, scale: float) -> dict[str, Any]:
    return {
        "runner": "run_2dele", "data": info, "seed": seed,
        "model": {"class": "MultiGraphGCM", "mode": "original",
                  "exchange_scale": scale, "dim": args.dim,
                  "heads": args.heads, "blocks": args.blocks,
                  "max_stages": info["max_stages"],
                  "dropout": args.dropout, "angle": args.angle,
                  "relay_padding": args.relay_padding,
                  "graph_embed": True, "graph_refresh": True,
                  "stage_ff": False, "layout": "graph_components"},
        "training": {"optimizer": "AdamW", "lr": args.lr,
                     "weight_decay": args.weight_decay,
                     "batch_size": args.batch_size, "epochs": args.epochs,
                     "patience": args.patience, "clip_norm": args.clip_norm,
                     "shuffle_seed": seed + 104729,
                     "selection": "minimum official validation batch-mean score",
                     "test_access": "once after loading validation-selected best.pt"},
    }


def _run_arm(args: argparse.Namespace, splits: dict[str, list[Data]],
             info: dict[str, Any], mapping: torch.Tensor,
             seed: int, scale: float, device: torch.device) -> dict[str, Any]:
    tag = f"{scale:g}".replace(".", "p")
    folder = args.output / info["run_variant"] / f"seed_{seed}" / f"scale_{tag}"
    result_path = folder / "result.json"
    latest_path = folder / "latest.pt"
    best_path = folder / "best.pt"
    trace_path = folder / "epochs.jsonl"
    config = _config(args, info, seed, scale)
    if result_path.is_file():
        if not args.resume:
            raise FileExistsError(f"Existing result requires --resume: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["config"] != config:
            raise ValueError(f"Cannot reuse incompatible result: {result_path}")
        print(json.dumps({"already_complete": str(result_path)}), flush=True)
        return result
    if latest_path.is_file() and not args.resume:
        raise FileExistsError(f"Existing checkpoint requires --resume: {latest_path}")
    _seed_everything(seed)
    model = _make_model(args, info, seed, scale).to(device)
    initial_digest = _parameter_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed + 104729)
    loaders = {
        split: DataLoader(data, batch_size=args.batch_size,
                          shuffle=(split == "train"), num_workers=0,
                          generator=(generator if split == "train" else None))
        for split, data in splits.items()
    }
    start_epoch, best_epoch, patience_count = 1, 0, 0
    best_score, elapsed_before = math.inf, 0.0
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        if saved["config"] != config or saved["initial_parameter_digest"] != initial_digest:
            raise ValueError(f"Cannot resume incompatible checkpoint: {latest_path}")
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
    print(json.dumps({"seed": seed, "exchange_scale": scale,
                      "representation": args.representation,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "start_epoch": start_epoch, "device": str(device)}), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        final_epoch = epoch
        train = _train_epoch(model, loaders["train"], optimizer, mapping,
                             device, args.clip_norm)
        val = _evaluate(model, loaders["val"], mapping, device)
        if val["score"] < best_score - 1e-12:
            best_score, best_epoch, patience_count = val["score"], epoch, 0
            _save(best_path, {"config": config, "epoch": epoch,
                              "val": val, "model": _state_cpu(model)})
        else:
            patience_count += 1
        _sync(device)
        duration = time.perf_counter() - epoch_started
        epoch_durations.append(duration)
        elapsed = elapsed_before + time.perf_counter() - started
        _append_jsonl(trace_path, {"epoch": epoch, "train_score": train,
                                   "val": val, "best_epoch": best_epoch,
                                   "best_val_score": best_score,
                                   "epoch_seconds": duration,
                                   "elapsed_seconds": elapsed})
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
            print(json.dumps({"seed": seed, "exchange_scale": scale,
                              "epoch": epoch, "train_score": train,
                              "val_score": val["score"],
                              "best_epoch": best_epoch}), flush=True)
        if patience_count >= args.patience:
            break
    if best_epoch < 1 or not best_path.is_file():
        raise RuntimeError("No validation-selected 2Dele checkpoint")
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    if selected["config"] != config:
        raise ValueError("Best checkpoint configuration changed")
    model.load_state_dict(selected["model"])
    test = _evaluate(model, loaders["test"], mapping, device)
    _sync(device)
    result = {
        "status": "complete", "config": config,
        "initial_parameter_digest": initial_digest,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "selected_epoch": best_epoch, "epochs_completed": final_epoch,
        "val": selected["val"], "test": test,
        "training_seconds": elapsed_before + time.perf_counter() - started,
        "median_epoch_seconds_this_invocation": (
            statistics.median(epoch_durations) if epoch_durations else None),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
        "best_checkpoint": str(best_path.resolve()),
        "trace_path": str(trace_path.resolve()),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "device": str(device)},
    }
    _write_json(result_path, result)
    print(json.dumps({"result": str(result_path),
                      "selected_epoch": best_epoch,
                      "val_score": result["val"]["score"],
                      "test_score": test["score"]}), flush=True)
    return result


def _check_only(args: argparse.Namespace, splits: dict[str, list[Data]],
                info: dict[str, Any], mapping: torch.Tensor) -> None:
    """Official-data CPU forward/backward; no optimizer step or score report."""
    _seed_everything(args.seeds[0])
    batch = next(iter(DataLoader(splits["train"], batch_size=2,
                                 shuffle=False, num_workers=0)))
    digests = []
    for scale in SCALES:
        _seed_everything(args.seeds[0])
        model = _make_model(args, info, args.seeds[0], scale).cpu().eval()
        digests.append(_parameter_digest(model))
        with torch.inference_mode():
            prediction = _predict(model, batch, mapping)
        if prediction.shape != batch.y.shape or not torch.isfinite(prediction).all():
            raise AssertionError("CPU paired original-J forward failed")
        if scale:
            # The J path has a distinct backward graph. Exercise it on the
            # released train split without updating parameters or scoring it.
            model.train()
            prediction = _predict(model, batch, mapping)
            error = prediction[batch.score_mask] - batch.y[batch.score_mask]
            error.square().sum(dim=1).mean().backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters()):
                raise AssertionError("Nonfinite original-J backward gradient")
    if digests[0] != digests[1]:
        raise AssertionError("Zero and J arms initialized differently")
    print(json.dumps({"check": "passed", "graphs": 2,
                      "representation": args.representation,
                      "prediction_shape": list(prediction.shape),
                      "scored_nodes": int(batch.score_mask.sum().item()),
                      "max_stages": info["max_stages"],
                      "matched_initial_parameters": True,
                      "j_backward": "passed",
                      "note": "No optimizer step or benchmark metric was run"}),
          flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=OFFICIAL_ROOT / "data/2Dele")
    parser.add_argument("--output", type=Path,
                        default=REPO / "results/2dele_gcm")
    parser.add_argument("--representation",
                        choices=["vertex_mean", "primal_dual"],
                        default="primal_dual")
    parser.add_argument("--geometry-input", action="store_true",
                        help="Feed released static geometry through existing node/edge encoders")
    parser.add_argument("--seeds", "--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--angle", type=float, default=0.2)
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
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if (args.batch_size < 1 or args.epochs < 1 or args.patience < 1
            or args.blocks < 1 or args.threads < 1 or args.print_every < 1
            or args.lr <= 0 or args.clip_norm <= 0 or args.weight_decay < 0):
        parser.error("Invalid positive training setting")
    if args.dim < 1 or args.heads < 1 or args.dim % (2 * args.heads):
        parser.error("dim must be divisible by twice the head count")
    if not 0 <= args.dropout < 1 or not math.isfinite(args.angle):
        parser.error("dropout must be in [0,1) and angle must be finite")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    return args


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.threads)
    splits, info, mapping = _load_data(args.data_root, args.representation,
                                      args.geometry_input)
    print(json.dumps(info, sort_keys=True), flush=True)
    if args.check_only:
        _check_only(args, splits, info, mapping)
        return
    device = torch.device(args.device)
    for seed in args.seeds:
        zero, exchange = [
            _run_arm(args, splits, info, mapping, seed, scale, device)
            for scale in SCALES
        ]
        if zero["initial_parameter_digest"] != exchange["initial_parameter_digest"]:
            raise RuntimeError("Paired arms did not share initialized parameters")
        paired = {
            "benchmark": info["benchmark"], "seed": seed,
            "representation": args.representation,
            "initial_parameter_digest": zero["initial_parameter_digest"],
            "zero": {"selected_epoch": zero["selected_epoch"],
                     "val": zero["val"], "test": zero["test"]},
            "exchange_0p25": {"selected_epoch": exchange["selected_epoch"],
                               "val": exchange["val"], "test": exchange["test"]},
            "delta_J_minus_zero": {
                split: {key: exchange[split][key] - zero[split][key]
                        for key in ("score", "MSE_scalar", "MAE_scalar",
                                    "R2_four_channel")
                        if exchange[split][key] is not None
                        and zero[split][key] is not None}
                for split in ("val", "test")},
            "external_reference": info["paper_table_2_reference_test_score"],
        }
        path = args.output / info["run_variant"] / f"seed_{seed}" / "paired.json"
        _write_json(path, paired)
        print(json.dumps({"paired_result": str(path),
                          "delta_test_score":
                          paired["delta_J_minus_zero"]["test"]["score"]}),
              flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
