"""Versioned inference on a fixed Loom layout for simple undirected graphs."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from threading import RLock

import torch

from .loom import Loom, compile_loom, normalized_neighbors


class LoomSession:
    """Maintain the normalized input view across feature and edge updates.

    Node IDs and their compiled addresses stay fixed. A version represents a
    committed batch, and reads return logits for the latest committed version.
    The session is for inference with a dedicated, fixed-weight model.
    """

    @torch.inference_mode()
    def __init__(self, model: Loom, x: torch.Tensor, edge_index: torch.Tensor,
                 mapping: torch.Tensor | None = None, *, retain: bool = True):
        if x.ndim != 2 or not x.is_floating_point() or not len(x):
            raise ValueError("x must be a nonempty floating-point matrix")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
            raise ValueError("edge_index must be a LongTensor with shape [2, E]")
        self._nodes = len(x)
        self._adjacency = [set() for _ in range(self._nodes)]
        for u, v in edge_index.T.tolist():
            self._check_node(u)
            self._check_node(v)
            if u == v:
                raise ValueError("LoomSession expects a simple graph without loops")
            self._adjacency[u].add(v)
            self._adjacency[v].add(u)
        canonical = self._edge_index(device=x.device)
        self._capacity = 1 << model.stages
        if self._nodes > self._capacity:
            raise ValueError("graph exceeds the model's wire capacity")
        if mapping is None:
            mapping, _ = compile_loom(canonical, self._nodes)
        if (mapping.shape != (self._nodes,) or mapping.dtype != torch.long
                or mapping.unique().numel() != self._nodes
                or mapping.min() < 0 or mapping.max() >= self._capacity):
            raise ValueError("mapping must assign each node a distinct in-range wire")
        self._mapping = mapping.to(x.device).clone()
        self._model = model.eval()
        self._x = x.clone()
        self._view = normalized_neighbors(self._x, canonical) if retain else None
        self._version = 0
        self._lock = RLock()

    def _check_node(self, node: int):
        if not 0 <= node < self._nodes:
            raise ValueError(f"node {node} is outside the fixed node set")

    def _edge_index(self, device):
        edges = [(u, v) for u, neighbors in enumerate(self._adjacency) for v in sorted(neighbors)]
        return torch.tensor(edges, dtype=torch.long, device=device).reshape(-1, 2).T.contiguous()

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @torch.inference_mode()
    def publish(self, *, insert: Iterable[tuple[int, int]] = (),
                delete: Iterable[tuple[int, int]] = (),
                features: Mapping[int, torch.Tensor] | None = None) -> int:
        """Apply deletions, insertions, then feature replacements as one batch.

        Returns the number of neighbor rows affected by the batch. A deletion
        must reference a present edge and an insertion an absent edge. Deleting
        and reinserting an edge in the same batch is allowed.
        """
        features = {} if features is None else dict(features)
        with self._lock:
            for node, value in features.items():
                self._check_node(node)
                if (value.shape != self._x.shape[1:] or value.dtype != self._x.dtype
                        or value.device != self._x.device):
                    raise ValueError("feature replacements must match the session input")
            staged = {}
            for edges, adding in ((delete, False), (insert, True)):
                for u, v in edges:
                    self._check_node(u)
                    self._check_node(v)
                    if u == v:
                        raise ValueError("self-loops are outside the session update contract")
                    for node in (u, v):
                        if node not in staged:
                            staged[node] = self._adjacency[node].copy()
                    if (v in staged[u]) == adding:
                        raise ValueError(f"invalid edge edit for ({u}, {v})")
                    if adding:
                        staged[u].add(v)
                        staged[v].add(u)
                    else:
                        staged[u].remove(v)
                        staged[v].remove(u)

            dirty = set()
            for node, neighbors in staged.items():
                old = self._adjacency[node]
                if old != neighbors:
                    dirty.add(node)
                if len(old) != len(neighbors):
                    dirty.update(neighbors)
            for node in features:
                dirty.update(staged.get(node, self._adjacency[node]))
            feature_ids = list(features)
            replacements = torch.stack(list(features.values())) if features else None
            rows = sorted(dirty)
            if self._view is not None and rows:
                patch = self._refresh(rows, staged, features)
                self._view[rows] = patch
            for node, neighbors in staged.items():
                self._adjacency[node] = neighbors
            if replacements is not None:
                self._x[feature_ids] = replacements
            self._version += 1
            return len(rows)

    def _refresh(self, rows, staged, features):
        dst, src, target_degree, source_degree = [], [], [], []
        for idx, node in enumerate(rows):
            neighbors = staged.get(node, self._adjacency[node])
            for other in sorted(neighbors):
                other_neighbors = staged.get(other, self._adjacency[other])
                dst.append(idx)
                src.append(other)
                target_degree.append(len(neighbors))
                source_degree.append(len(other_neighbors))
        if not src:
            return self._x.new_zeros((len(rows), self._x.shape[1]))
        sources = sorted(set(src))
        lookup = {node: i for i, node in enumerate(sources)}
        inputs = self._x[sources]
        for node, value in features.items():
            if node in lookup:
                inputs[lookup[node]] = value
        index = torch.tensor([dst, [lookup[node] for node in src]],
                             dtype=torch.long, device=self._x.device)
        weights = (self._x.new_tensor(target_degree) * self._x.new_tensor(source_degree)).rsqrt()
        sparse = torch.sparse_coo_tensor(index, weights, (len(rows), len(sources))).coalesce()
        return torch.sparse.mm(sparse, inputs)

    @torch.inference_mode()
    def read(self) -> tuple[int, torch.Tensor]:
        """Return the version and full Loom logits in original node order."""
        with self._lock:
            view = self._view
            if view is None:
                view = normalized_neighbors(self._x, self._edge_index(self._x.device))
            return self._version, self._model(self._x, view, self._mapping, padded=self._capacity)
