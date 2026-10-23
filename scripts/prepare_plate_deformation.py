"""Prepare the official IGNS Plate Deformation ZIP as static PyG graphs.

The source protocol is ICLR 2026 IGNS, repository commit 328985ee30b4ecabb06463600f57ac7f67c420bc.
The official 48-frame single-step window uses t=0 as input and t=47 as target.
Only Parameter.yaml and PLY-1-{0,47}.vtk are read from each SimNNN entry.

Example:
    python scripts/prepare_plate_deformation.py \
        --zip data/plate_deformation/plate_deformation.zip \
        --out data/plate_deformation/processed_static --resume

Output: {train,val,test}/SimNNN.pt and metadata.json. Each PyG Data has raw
x=[u0(6),properties(6)], y=t47 state(6), pos, edge_index, edge_attr(9),
node_type, loss_mask, force_pos, and mesh_node_count. The force properties are
unscaled; edge_attr uses the official 1/400 geometry scaling. The six-channel
target and original-scale per-node sum-of-squares must be retained for the
paper's test/loss metric. The paired runner uses only ``properties`` as its
learned node input and keeps ``u0`` for the official residual output rule.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
from gcm.paths import data_path
import re
import tempfile
from typing import Any
import zipfile

import meshio
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


REPO = Path(__file__).resolve().parents[1]
OFFICIAL = REPO / "external" / "neural_pde_matching"
OFFICIAL_COMMIT = "328985ee30b4ecabb06463600f57ac7f67c420bc"
DEFAULT_ZIP = data_path('plate_deformation/plate_deformation.zip')
DEFAULT_OUT = data_path('plate_deformation/processed_static')
SIM_FILE = re.compile(r"(?:^|/)Sim(\d{3})/(Parameter\.yaml|PLY-1/PLY-1-(?:0|47)\.vtk)$", re.I)
REQUIRED = {"Parameter.yaml", "PLY-1/PLY-1-0.vtk", "PLY-1/PLY-1-47.vtk"}
SPLIT_IDS = {"train": list(range(0, 800)), "val": list(range(800, 900)), "test": list(range(900, 1000))}


@lru_cache(maxsize=None)
def _official_module(name: str, relative_path: str):
    """Load original helper files without importing the optional Hydra stack."""
    path = OFFICIAL / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Official code missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official source {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _official_helpers():
    plate = _official_module(
        "official_plate_utils",
        "src/pde_matching/envs/abaqus_plate_deformation/utils.py",
    )
    graphs = _official_module("official_env_utils", "src/pde_matching/envs/utils.py")
    return plate, graphs


def _index_zip(zf: zipfile.ZipFile) -> dict[int, dict[str, zipfile.ZipInfo]]:
    entries: dict[int, dict[str, zipfile.ZipInfo]] = defaultdict(dict)
    for info in zf.infolist():
        match = SIM_FILE.search(info.filename.replace("\\", "/"))
        if not match:
            continue
        sim_id, suffix = int(match.group(1)), match.group(2)
        if sim_id >= 1000:
            continue
        canonical = next((name for name in REQUIRED if name.lower() == suffix.lower()), None)
        assert canonical is not None
        if canonical in entries[sim_id]:
            raise ValueError(f"Duplicate {canonical} in Sim{sim_id:03d}")
        entries[sim_id][canonical] = info
    return entries


def _read_vtk(zf: zipfile.ZipFile, info: zipfile.ZipInfo, out_dir: Path):
    # meshio's VTK reader requires a path even though the generic API accepts
    # buffers. Extract only one selected frame at a time, then remove it.
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".vtk", delete=False) as file:
        temporary = Path(file.name)
        file.write(zf.read(info))
    try:
        return meshio.read(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def _read_forces(zf: zipfile.ZipFile, info: zipfile.ZipInfo, out_dir: Path, plate):
    # The official parser accepts a path and preserves YAML insertion order.
    # Keep just one small temporary YAML on disk for each graph.
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".yaml", delete=False) as file:
        temporary = Path(file.name)
        file.write(zf.read(info))
    try:
        return plate.parse_yaml_to_numpy(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def _source_info(files: dict[str, zipfile.ZipInfo]) -> dict[str, Any]:
    return {
        key: {"zip_path": info.filename, "size": info.file_size, "crc32": f"{info.CRC:08x}"}
        for key, info in sorted(files.items())
    }


def _as_float32(array, *, shape_last: int, label: str) -> torch.Tensor:
    result = torch.as_tensor(np.asarray(array), dtype=torch.float32)
    if result.ndim != 2 or result.shape[1] != shape_last:
        raise ValueError(f"Expected {label} [N,{shape_last}], got {tuple(result.shape)}")
    if not torch.isfinite(result).all():
        raise ValueError(f"Non-finite values in {label}")
    return result


def build_graph(
    zf: zipfile.ZipFile,
    files: dict[str, zipfile.ZipInfo],
    sim_id: int,
    out_dir: Path,
    plate,
    graphs,
) -> Data:
    if set(files) != REQUIRED:
        raise ValueError(f"Sim{sim_id:03d} missing {sorted(REQUIRED - set(files))}")
    start = _read_vtk(zf, files["PLY-1/PLY-1-0.vtk"], out_dir)
    end = _read_vtk(zf, files["PLY-1/PLY-1-47.vtk"], out_dir)
    force_array = _read_forces(zf, files["Parameter.yaml"], out_dir, plate)
    if len(force_array) == 0:
        raise ValueError(f"Sim{sim_id:03d} has no applied forces")

    edges, faces = plate.extract_faces_edges(start)
    if not edges:
        raise ValueError(f"Sim{sim_id:03d} has no mesh edges")
    pos0 = _as_float32(plate.extract_geometry(start), shape_last=3, label="t0 position")
    pos47 = _as_float32(plate.extract_geometry(end), shape_last=3, label="t47 position")
    displacement0 = _as_float32(start.point_data["U"], shape_last=3, label="t0 U")
    displacement47 = _as_float32(end.point_data["U"], shape_last=3, label="t47 U")
    if pos0.shape != pos47.shape or pos0.shape != displacement0.shape or pos47.shape != displacement47.shape:
        raise ValueError(f"Sim{sim_id:03d} changed mesh point count or U shape")

    n_mesh = len(pos0)
    mesh_edges = to_undirected(torch.as_tensor(edges, dtype=torch.long).T.contiguous(), num_nodes=n_mesh)
    mesh = Data(pos=pos0, edge_index=mesh_edges)
    # Faces are not needed by the model, but reading them via the official
    # helper above verifies that mixed triangle/quad/tetra meshes are parsed.
    del faces
    force_pos = _as_float32(force_array["position"], shape_last=3, label="force position")
    force_dir = _as_float32(force_array["direction"], shape_last=3, label="force direction")
    force_graph = Data(
        pos=force_pos,
        edge_index=torch.arange(len(force_array), dtype=torch.long).repeat(2, 1),
    )

    # These enum values and the radius selection are copied from the official
    # AbaqusPlateDeformationEnv._process_by_stage implementation.
    class NodeType:
        NORMAL, OBSTACLE, BOUNDARY, SIZE = 0, 1, 2, 3

    class EdgeType:
        NORMAL_NORMAL, OBSTACLE_OBSTACLE, NORMAL_OBSTACLE, OBSTACLE_NORMAL, BOUNDARY_NORMAL, SIZE = range(6)

    condition = ~(
        (pos0[:, 0] > 1.0)
        & (pos0[:, 0] < 139.0)
        & (pos0[:, 1] > 1.0)
        & (pos0[:, 1] < 139.0)
    )
    condition = torch.cat([condition, torch.zeros(len(force_pos), dtype=torch.bool)])
    force_radius = float(force_array["length"][0]) / 2.0
    graph = graphs.connect_two_graphs(
        mesh,
        force_graph,
        node_type=NodeType,
        edge_type=EdgeType,
        radius=force_radius,
        max_num_neighbors=n_mesh,
        collider_to_collider_edges=False,
        mesh_to_collider_edges=False,
        collider_to_mesh_edges=True,
        device=torch.device("cpu"),
    )

    # Official u(t) = [pos(t), U(t)] and properties = node-type one-hot
    # concatenated with force direction. The force-node U is padded with zero.
    zeros_force = torch.zeros((len(force_pos), 3), dtype=torch.float32)
    u0 = torch.cat([graph.pos, torch.cat([displacement0, zeros_force])], dim=1)
    y = torch.cat(
        [torch.cat([pos47, force_pos]), torch.cat([displacement47, zeros_force])], dim=1
    )
    properties = torch.cat(
        [torch.eye(NodeType.SIZE)[graph.node_type],
         torch.cat([torch.zeros((n_mesh, 3)), force_dir])], dim=1
    ).float()
    graph.node_type[condition] = NodeType.BOUNDARY
    properties[condition] = torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float32)

    # The official EdgeCategorical and EdgeFeatureScaler define __call__ rather
    # than forward and cannot instantiate under local PyG 2.7. Reproduce their
    # exact operations, including PyG Cartesian's row-minus-column direction.
    row, col = graph.edge_index
    cartesian = graph.pos[row] - graph.pos[col]
    distance = torch.linalg.vector_norm(cartesian, dim=1, keepdim=True)
    graph.edge_attr = torch.cat(
        [torch.nn.functional.one_hot(graph.edge_type, num_classes=EdgeType.SIZE).float(),
         cartesian / 400.0, distance / 400.0], dim=1
    )
    graph.x = torch.cat([u0, properties], dim=1)
    graph.u0 = u0
    graph.properties = properties
    graph.y = y
    graph.loss_mask = (graph.node_type == NodeType.NORMAL) | (graph.node_type == NodeType.BOUNDARY)
    graph.output_mask = graph.node_type == NodeType.NORMAL
    graph.force_pos = force_pos
    graph.mesh_node_count = n_mesh
    graph.sim_id = sim_id
    graph.force_radius = force_radius

    if graph.x.shape != (n_mesh + len(force_pos), 12) or graph.y.shape != (n_mesh + len(force_pos), 6):
        raise AssertionError("Unexpected graph feature shape")
    if graph.edge_attr.shape != (graph.edge_index.shape[1], 9):
        raise AssertionError("Unexpected official edge feature shape")
    if not graph.loss_mask[:n_mesh].all() or graph.loss_mask[n_mesh:].any():
        raise AssertionError("Official NORMAL+BOUNDARY mask was not reproduced")
    if not torch.isfinite(graph.edge_attr).all():
        raise ValueError(f"Sim{sim_id:03d} has non-finite edge features")
    return graph


class ChannelStats:
    def __init__(self, width: int):
        self.count = 0
        self.sum = np.zeros(width, dtype=np.float64)
        self.sum_sq = np.zeros(width, dtype=np.float64)

    def add(self, tensor: torch.Tensor) -> None:
        array = tensor.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1, len(self.sum))
        self.count += len(array)
        self.sum += array.sum(axis=0)
        self.sum_sq += np.square(array).sum(axis=0)

    def finish(self) -> tuple[list[float], list[float]]:
        if self.count == 0:
            raise ValueError("Cannot compute train-only statistics from empty training split")
        mean = self.sum / self.count
        variance = np.maximum(self.sum_sq / self.count - mean * mean, 0.0)
        std = np.sqrt(variance)
        std[std == 0] = 1e-6
        return mean.astype(np.float32).tolist(), std.astype(np.float32).tolist()


def prepare(zip_path: Path, out_dir: Path, *, resume: bool = False,
            ids: list[int] | None = None, allow_incomplete: bool = False) -> dict[str, Any]:
    zip_path = zip_path.resolve()
    out_dir = out_dir.resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    plate, graphs = _official_helpers()
    wanted = set(range(1000)) if ids is None else set(ids)
    if not wanted or min(wanted) < 0 or max(wanted) >= 1000:
        raise ValueError("IDs must be in [0,999]")
    if not allow_incomplete and wanted != set(range(1000)):
        raise ValueError("Use --allow-incomplete for a subset; it is not benchmark-comparable")
    input_stats, target_stats = ChannelStats(6), ChannelStats(6)
    records: list[dict[str, Any]] = []
    produced: dict[str, list[int]] = {stage: [] for stage in SPLIT_IDS}
    with zipfile.ZipFile(zip_path) as zf:
        entries = _index_zip(zf)
        missing = sorted(i for i in wanted if set(entries.get(i, {})) != REQUIRED)
        if missing:
            raise ValueError(f"ZIP missing selected triplets: {missing[:15]} (total {len(missing)})")
        for stage, all_ids in SPLIT_IDS.items():
            stage_dir = out_dir / stage
            stage_dir.mkdir(exist_ok=True)
            stage_ids = [i for i in all_ids if i in wanted]
            for index, sim_id in enumerate(stage_ids, 1):
                path = stage_dir / f"Sim{sim_id:03d}.pt"
                if resume and path.is_file():
                    graph = torch.load(path, map_location="cpu", weights_only=False)
                else:
                    graph = build_graph(zf, entries[sim_id], sim_id, out_dir, plate, graphs)
                    pending = path.with_suffix(".pt.tmp")
                    torch.save(graph, pending)
                    pending.replace(path)
                if graph.sim_id != sim_id:
                    raise ValueError(f"Stale or wrong graph at {path}")
                produced[stage].append(sim_id)
                if stage == "train":
                    input_stats.add(graph.u0)
                    target_stats.add(graph.y[graph.loss_mask])
                records.append({
                    "sim_id": sim_id,
                    "split": stage,
                    "file": f"{stage}/{path.name}",
                    "num_nodes": int(graph.num_nodes),
                    "num_mesh_nodes": int(graph.mesh_node_count),
                    "num_force_nodes": int(graph.num_nodes - graph.mesh_node_count),
                    "num_edges": int(graph.num_edges),
                    "num_scored_nodes": int(graph.loss_mask.sum()),
                    "force_radius": float(graph.force_radius),
                    "raw": _source_info(entries[sim_id]),
                })
                if index % 25 == 0 or index == len(stage_ids):
                    print(f"{stage}: {index}/{len(stage_ids)} graphs", flush=True)
        zip_count = len(zf.infolist())

    input_mean, input_std = input_stats.finish()
    target_mean, target_std = target_stats.finish()
    metadata = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete_official_split": all(produced[k] == v for k, v in SPLIT_IDS.items()),
        "source": {
            "zip_path": str(zip_path),
            "zip_bytes": zip_path.stat().st_size,
            "zip_entries": zip_count,
            "official_repository": "https://github.com/thobotics/neural_pde_matching",
            "official_commit": OFFICIAL_COMMIT,
            "input_frame": 0,
            "target_frame": 47,
            "window_size": 48,
        },
        "schema": {
            "x": "[u0_xyz, u0_Uxyz, node_type_onehot, force_direction] (12 raw channels)",
            "y": "[t47_xyz, t47_Uxyz] (6 raw channels)",
            "node_type": {"NORMAL": 0, "OBSTACLE_FORCE": 1, "BOUNDARY": 2},
            "loss_mask": "NORMAL or BOUNDARY, excluding force nodes",
            "output_mask": "Environment-defined NORMAL-only mask; official PDEMatching.forward currently ignores it",
            "edge_attr": "edge_type one-hot(5), (pos[source]-pos[target])/400 (3), Euclidean distance/400 (1)",
            "edge_direction": "bidirectional mesh edges; force-to-mesh edges only",
            "test_loss": "mean over loss-masked nodes of sum over six raw squared state errors",
        },
        "splits": produced,
        "stats": {
            "input_u_mean": input_mean,
            "input_u_std": input_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "target_stats_mask": "train split NORMAL and BOUNDARY nodes only",
        },
        "graphs": records,
    }
    pending = out_dir / "metadata.json.tmp"
    pending.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pending.replace(out_dir / "metadata.json")
    print(f"Wrote {out_dir / 'metadata.json'}", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ids", type=int, nargs="+", default=None,
                        help="Specific Sim IDs for a smoke check; not a comparable benchmark")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    prepare(args.zip, args.out, resume=args.resume, ids=args.ids,
            allow_incomplete=args.allow_incomplete)


if __name__ == "__main__":
    main()
