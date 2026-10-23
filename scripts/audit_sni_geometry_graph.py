"""Audit a label-free coordinate graph for the released SNI Laplace data.

The graph builder receives only node coordinates and boundary-node coordinates.
The official .msh files are read exclusively by the audit functions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import deque
from pathlib import Path
from gcm.paths import data_path

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


FILES = {
    "train": "laplace2d_simple_20000_train.pkl",
    "heldout": "laplace2d_simple_2500_test.pkl",
    "A": "laplace2d_schwarz_100_test.pkl",
    "B": "laplace2d_holes_100_test.pkl",
    "C": "laplace2d_bosch_100_test.pkl",
}
MESHES = {"A": "A-schwarz.msh", "B": "B-holes.msh", "C": "C-bosch.msh"}


def read_gmsh41(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ASCII Gmsh 4.1 node, line, and triangle blocks for audit only."""
    lines = path.read_text(encoding="utf-8").splitlines()
    nodes_start = lines.index("$Nodes") + 1
    elements_start = lines.index("$Elements") + 1
    n_blocks, n_nodes, _, _ = map(int, lines[nodes_start].split())
    cursor = nodes_start + 1
    nodes: dict[int, np.ndarray] = {}
    for _ in range(n_blocks):
        entity_dim, entity_tag, parametric, block_n = map(int, lines[cursor].split())
        del entity_dim, entity_tag
        cursor += 1
        tags = [int(lines[cursor + j].strip()) for j in range(block_n)]
        cursor += block_n
        for tag in tags:
            vals = np.fromstring(lines[cursor], sep=" ")
            nodes[tag] = vals[:2]
            cursor += 1
            if parametric and len(vals) < 3 + parametric:
                raise ValueError(f"Unexpected parametric node at {path}:{cursor}")
    if len(nodes) != n_nodes or lines[cursor] != "$EndNodes":
        raise ValueError(f"Malformed node section: {path}")
    n_blocks, _, _, _ = map(int, lines[elements_start].split())
    cursor = elements_start + 1
    mesh_lines: list[list[int]] = []
    triangles: list[list[int]] = []
    for _ in range(n_blocks):
        entity_dim, entity_tag, element_type, block_n = map(int, lines[cursor].split())
        del entity_dim, entity_tag
        cursor += 1
        for _ in range(block_n):
            vals = list(map(int, lines[cursor].split()))
            cursor += 1
            if element_type == 1:
                mesh_lines.append(vals[1:3])
            elif element_type == 2:
                triangles.append(vals[1:4])
    if lines[cursor] != "$EndElements":
        raise ValueError(f"Malformed element section: {path}")
    tags = sorted(nodes)
    index = {tag: i for i, tag in enumerate(tags)}
    xy = np.array([nodes[tag] for tag in tags], dtype=np.float64)
    edges = np.array([[index[t] for t in edge] for edge in mesh_lines], dtype=np.int64)
    tris = np.array([[index[t] for t in tri] for tri in triangles], dtype=np.int64)
    return xy, edges, tris


def canonical_edges(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.int64).reshape(-1, 2)
    if len(raw) == 0:
        return raw
    return np.unique(np.sort(raw, axis=1), axis=0)


def triangle_edges(tris: np.ndarray) -> np.ndarray:
    return canonical_edges(np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]])))


def map_coordinates(query: np.ndarray, reference: np.ndarray, *, tol: float = 1e-7) -> tuple[np.ndarray, float]:
    distances, ids = cKDTree(reference).query(query)
    if np.max(distances) > tol or len(np.unique(ids)) != len(ids):
        raise ValueError(f"Coordinates do not match uniquely: max distance={np.max(distances):.3g}")
    return ids.astype(np.int64), float(np.max(distances))


def geometry_graph(xy: np.ndarray, boundary_ids: np.ndarray, *, ratio: float = 1.3) -> np.ndarray:
    """Delaunay edges bounded by local three-neighbor coordinate spacing.

    An edge of length d survives when d <= ratio * max(s_i, s_j), where s is
    the distance to the third other node. The rule is identical across splits
    and never receives targets or the official mesh.
    """
    del boundary_ids  # Kept for the input signature used by the runner.
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 4:
        raise ValueError("Expected at least four non-collinear nodes")
    dists, _ = cKDTree(xy).query(xy, k=4)
    scale = dists[:, 3]
    edges = triangle_edges(Delaunay(xy).simplices)
    length = np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1)
    return edges[length <= ratio * np.maximum(scale[edges[:, 0]], scale[edges[:, 1]])]


def graph_metrics(n: int, edges: np.ndarray, boundary_ids: np.ndarray) -> dict[str, object]:
    row = np.r_[edges[:, 0], edges[:, 1]]
    col = np.r_[edges[:, 1], edges[:, 0]]
    matrix = coo_matrix((np.ones(len(row), dtype=np.int8), (row, col)), shape=(n, n)).tocsr()
    n_comp, labels = connected_components(matrix, directed=False)
    hops = np.full(n, -1, dtype=np.int32)
    hops[boundary_ids] = 0
    q = deque(int(i) for i in boundary_ids)
    while q:
        i = q.popleft()
        for j in matrix.indices[matrix.indptr[i] : matrix.indptr[i + 1]]:
            if hops[j] < 0:
                hops[j] = hops[i] + 1
                q.append(int(j))
    inside = np.ones(n, dtype=bool)
    inside[boundary_ids] = False
    interior_hops = hops[inside]
    return {
        "nodes": n,
        "edges": int(len(edges)),
        "components": int(n_comp),
        "largest_component_fraction": float(np.max(np.bincount(labels)) / n),
        "boundary_nodes": int(len(boundary_ids)),
        "interior_nodes": int(np.sum(inside)),
        "interior_unreachable": int(np.sum(interior_hops < 0)),
        "interior_hops_gt3_fraction": float(np.mean(interior_hops > 3)) if len(interior_hops) else None,
        "interior_hops_median": float(np.median(interior_hops)) if len(interior_hops) else None,
        "interior_hops_max": int(np.max(interior_hops)) if len(interior_hops) else None,
    }


def load_record(path: Path, index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    sol, bc_list = data[index]
    bc = np.concatenate(bc_list, axis=0)
    return np.asarray(sol[:, :2], dtype=np.float64), np.asarray(bc[:, :2], dtype=np.float64)


def audit_fixed_domain(root: Path, key: str, ratio: float) -> dict[str, object]:
    import matplotlib.tri as mtri

    xy, boundary_xy = load_record(root / FILES[key])
    boundary_ids, boundary_error = map_coordinates(boundary_xy, xy)
    edges = geometry_graph(xy, boundary_ids, ratio=ratio)
    mesh_xy, mesh_lines, mesh_tris = read_gmsh41(root / "mesh" / MESHES[key])
    to_record, mesh_error = map_coordinates(mesh_xy, xy)
    official_tris = to_record[mesh_tris]
    official_edges = triangle_edges(official_tris)
    official_boundary = canonical_edges(to_record[mesh_lines])
    expected_boundary = np.unique(official_boundary)
    if set(expected_boundary) != set(boundary_ids):
        raise ValueError(f"Boundary ID mismatch on {key}")
    official_set = set(map(tuple, official_edges))
    graph_set = set(map(tuple, edges))

    # An edge whose interior leaves the union of released FEM triangles is a
    # geometric shortcut through a hole or outside a concavity.
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=official_tris)
    finder = tri.get_trifinder()
    fractions = np.linspace(0.05, 0.95, 19)
    points = xy[edges[:, 0], None, :] * (1 - fractions[None, :, None]) + xy[edges[:, 1], None, :] * fractions[None, :, None]
    hit = np.asarray(finder(points[..., 0].ravel(), points[..., 1].ravel())).reshape(len(edges), len(fractions))
    outside = np.any(hit < 0, axis=1)
    # Official boundary edges lie exactly on FEM cell boundaries, which the
    # triangle finder may treat ambiguously; exempt only those certified edges.
    outside &= ~np.array([tuple(edge) in official_set for edge in edges])
    off_edges = edges[outside]
    distances = np.linalg.norm(xy[official_edges[:, 0]] - xy[official_edges[:, 1]], axis=1)
    official_sparse = coo_matrix(
        (
            np.r_[distances, distances],
            (np.r_[official_edges[:, 0], official_edges[:, 1]], np.r_[official_edges[:, 1], official_edges[:, 0]]),
        ),
        shape=(len(xy), len(xy)),
    ).tocsr()
    off_details = []
    for u, v in off_edges:
        geodesic = shortest_path(official_sparse, directed=False, indices=int(u))
        hops = shortest_path(official_sparse, directed=False, unweighted=True, indices=int(u))
        direct = float(np.linalg.norm(xy[u] - xy[v]))
        off_details.append(
            {
                "node_ids": [int(u), int(v)],
                "coordinates": xy[[u, v]].tolist(),
                "euclidean_length": direct,
                "official_hops": int(hops[v]),
                "official_path_length": float(geodesic[v]),
                "path_to_direct_ratio": float(geodesic[v] / direct),
                "both_boundary": bool(u in boundary_ids and v in boundary_ids),
            }
        )

    out = graph_metrics(len(xy), edges, boundary_ids)
    out.update(
        {
            "mesh_coordinate_max_error": mesh_error,
            "boundary_coordinate_max_error": boundary_error,
            "official_edges": int(len(official_edges)),
            "official_boundary_edges": int(len(official_boundary)),
            "official_edge_recall": len(graph_set & official_set) / len(official_set),
            "official_edge_precision": len(graph_set & official_set) / len(graph_set),
            "nonmesh_edges": len(graph_set - official_set),
            "off_domain_edges": int(np.sum(outside)),
            "off_domain_details": off_details,
            "official_graph": graph_metrics(len(xy), official_edges, boundary_ids),
        }
    )
    return out


def audit_sample(root: Path, key: str, ratio: float, n_samples: int) -> dict[str, object]:
    with (root / FILES[key]).open("rb") as f:
        records = pickle.load(f)
    first_by_geometry: dict[str, int] = {}
    repetitions: dict[str, int] = {}
    for index, (sol, _) in enumerate(records):
        digest = hashlib.sha1(np.ascontiguousarray(sol[:, :2]).tobytes()).hexdigest()
        first_by_geometry.setdefault(digest, index)
        repetitions[digest] = repetitions.get(digest, 0) + 1
    unique_ids = np.array(list(first_by_geometry.values()), dtype=int)
    ids = unique_ids[np.linspace(0, len(unique_ids) - 1, min(n_samples, len(unique_ids)), dtype=int)]
    per_graph = []
    for index in ids:
        sol, bc_list = records[int(index)]
        xy = np.asarray(sol[:, :2], dtype=np.float64)
        boundary_xy = np.concatenate(bc_list, axis=0)[:, :2]
        boundary_ids, error = map_coordinates(boundary_xy, xy)
        edges = geometry_graph(xy, boundary_ids, ratio=ratio)
        g = graph_metrics(len(xy), edges, boundary_ids)
        g.update({"record_index": int(index), "boundary_coordinate_max_error": error})
        per_graph.append(g)
    return {
        "record_count": len(records),
        "unique_geometry_count": len(unique_ids),
        "repetitions_per_geometry": sorted(set(repetitions.values())),
        "sample_strategy": "first record from each exact coordinate geometry; evenly spaced if sample-count is smaller",
        "sample_indices": ids.tolist(),
        "sample_count": len(per_graph),
        "connected_count": int(sum(g["components"] == 1 for g in per_graph)),
        "interior_nodes_total": int(sum(g["interior_nodes"] for g in per_graph)),
        "interior_gt3_total": int(sum((g["interior_hops_gt3_fraction"] or 0) * g["interior_nodes"] for g in per_graph)),
        "per_graph": per_graph,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=data_path('sni_laplace'))
    parser.add_argument("--out", type=Path, default=Path("results/diagnostics/sni_geometry_graph_audit_20260926.json"))
    parser.add_argument("--ratio", type=float, default=1.3)
    parser.add_argument("--sample-count", type=int, default=2000)
    args = parser.parse_args()
    data = {
        "rule": "Delaunay edges with distance <= ratio * max(endpoint 3rd-neighbor distances)",
        "ratio": args.ratio,
        "data_root": str(args.data_root),
        "fixed_domains": {key: audit_fixed_domain(args.data_root, key, args.ratio) for key in MESHES},
        "train_sample": audit_sample(args.data_root, "train", args.ratio, args.sample_count),
        "heldout_sample": audit_sample(args.data_root, "heldout", args.ratio, args.sample_count),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    for key, report in data["fixed_domains"].items():
        print(key, {k: report[k] for k in ("nodes", "edges", "components", "official_edge_recall", "official_edge_precision", "off_domain_edges", "interior_hops_gt3_fraction")})
    for key in ("train_sample", "heldout_sample"):
        report = data[key]
        print(key, {k: report[k] for k in ("sample_count", "connected_count", "interior_nodes_total", "interior_gt3_total")})
    print(args.out)


if __name__ == "__main__":
    main()
