"""Fiedler ordering, Gray addresses, and Hamming-cost refinement."""

import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch
from numba import njit


def compiler_adjacency(adjacency):
    a = sp.csr_matrix(adjacency, dtype=bool)
    a = a.maximum(a.T).tocsr()
    a.setdiag(False)
    a.eliminate_zeros()
    a.sort_indices()
    return a


def hamming_cost(adjacency, mapping):
    a = compiler_adjacency(adjacency)
    u, v = sp.triu(a, k=1).nonzero()
    xor = (
        np.asarray(mapping, dtype=np.int64)[u] ^ np.asarray(mapping, dtype=np.int64)[v]
    )
    table = np.array([i.bit_count() for i in range(256)], dtype=np.int64)
    return int(sum(table[(xor >> bit) & 255].sum() for bit in range(0, 64, 8)))


@njit(cache=True)
def _popcount(value):
    result = 0
    while value:
        value &= value - 1
        result += 1
    return result


@njit(cache=True)
def _refine_native(
    indptr, indices, mapping, padded, seed, max_passes, random_candidates
):
    np.random.seed(seed)
    n = len(mapping)
    inverse = np.full(padded, -1, dtype=np.int64)
    for i in range(n):
        inverse[mapping[i]] = i
    stages = 0
    while (1 << stages) < padded:
        stages += 1
    swaps, moves, evaluated, passes = 0, 0, 0, 0
    history = np.zeros(max_passes * n, dtype=np.int64)
    used = 0
    for sweep in range(max_passes):
        occupied = np.sort(mapping.copy())
        visit = inverse[occupied[np.random.permutation(n)]].copy()
        changed = False
        for u in visit:
            old = mapping[u]
            if n <= 64:
                candidates = np.arange(padded)
            else:
                candidates = np.empty(stages + random_candidates, dtype=np.int64)
                for bit in range(stages):
                    candidates[bit] = old ^ (1 << bit)
                for j in range(random_candidates):
                    address = np.random.randint(padded - 1)
                    candidates[stages + j] = address + (address >= old)
                candidates = np.unique(candidates)
            best, best_delta = -1, 0
            for address in candidates:
                if address == old:
                    continue
                v = inverse[address]
                delta = 0
                for p in range(indptr[u], indptr[u + 1]):
                    neighbor = indices[p]
                    if neighbor != v:
                        w = mapping[neighbor]
                        delta += _popcount(address ^ w) - _popcount(old ^ w)
                if v >= 0:
                    for p in range(indptr[v], indptr[v + 1]):
                        neighbor = indices[p]
                        if neighbor != u:
                            w = mapping[neighbor]
                            delta += _popcount(old ^ w) - _popcount(address ^ w)
                evaluated += 1
                if delta < best_delta:
                    best, best_delta = address, delta
            if best >= 0:
                v = inverse[best]
                mapping[u] = best
                inverse[best] = u
                inverse[old] = v
                if v >= 0:
                    mapping[v] = old
                    swaps += 1
                else:
                    moves += 1
                history[used] = best_delta
                used += 1
                changed = True
        passes = sweep + 1
        if not changed:
            break
    return mapping, swaps, moves, evaluated, passes, history[:used]


def refine_mapping(
    adjacency, mapping, padded, seed=0, max_passes=4, random_candidates=16
):
    """Full compiler refinement including relocations into unused addresses."""
    a = compiler_adjacency(adjacency)
    mapping = np.asarray(mapping, dtype=np.int64).copy()
    n = a.shape[0]
    if (
        mapping.shape != (n,)
        or len(np.unique(mapping)) != n
        or padded < max(n, 1)
        or padded & (padded - 1)
        or np.any(mapping < 0)
        or np.any(mapping >= padded)
    ):
        raise ValueError("Expected injective addresses in a power-of-two capacity")
    initial = hamming_cost(a, mapping)
    mapped, swaps, moves, evaluated, passes, deltas = _refine_native(
        a.indptr, a.indices, mapping, padded, seed, max_passes, random_candidates
    )
    final = hamming_cost(a, mapped)
    if final != initial + int(deltas.sum()):
        raise AssertionError("Incremental compiler cost disagrees with full cost")
    return mapped, dict(
        initial_cost=initial,
        final_cost=final,
        accepted_swaps=int(swaps),
        accepted_moves=int(moves),
        evaluated_candidates=int(evaluated),
        passes=int(passes),
        cost_history=[initial] + (initial + np.cumsum(deltas)).tolist(),
    )


def _spectral_order(adjacency, allow_symmetric=False):
    """Seriate a connected graph; optionally resolve symmetric signs by index."""
    n = adjacency.shape[0]
    if n < 3:
        return np.arange(n, dtype=np.int64)
    count, _ = sp.csgraph.connected_components(adjacency, directed=False)
    if count != 1:
        raise ValueError("graph_spectral requires a connected graph")
    degree = np.asarray(adjacency.sum(1)).ravel()
    laplacian = sp.csgraph.laplacian(adjacency, normed=False).tocsr()
    initial = degree - degree.mean() + 0.01
    _, vectors = sla.eigsh(
        laplacian + 1e-3 * sp.eye(n, format="csr"),
        k=2,
        sigma=0.0,
        which="LM",
        tol=1e-6,
        maxiter=5000,
        v0=initial,
    )
    axis = vectors[:, 1]
    # The eigenvector sign is arbitrary. A topology-only moment fixes the
    # direction whenever the graph distinguishes its two ends.
    for anchor in (degree, adjacency @ degree, adjacency @ (adjacency @ degree)):
        centered = np.asarray(anchor).ravel() - np.mean(anchor)
        moment = float(axis @ centered)
        if abs(moment) > 1e-6 * np.linalg.norm(centered):
            if moment > 0:
                axis = -axis
            break
    else:
        if not allow_symmetric:
            raise ValueError("graph_spectral cannot orient this symmetric graph")
        # New layout's deterministic fallback; unresolved symmetries may
        # depend on input numbering, as recorded in the compiler specification.
        if axis[int(np.argmax(np.abs(axis)))] < 0:
            axis = -axis
    return np.argsort(axis, kind="stable")


def compile_graph(edge_index: torch.Tensor, num_nodes: int, seed: int = 0):
    """Fiedler ranks -> Gray addresses -> swaps and vacant-wire moves."""
    nodes = num_nodes
    if nodes < 1:
        raise ValueError("num_nodes must be positive")
    edges = edge_index.detach().cpu().numpy()
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    a = compiler_adjacency(
        sp.csr_matrix(
            (np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(nodes, nodes)
        )
    )
    count, labels = sp.csgraph.connected_components(a, directed=False)
    components = [np.flatnonzero(labels == i) for i in range(count)]
    components.sort(key=lambda ids: (-len(ids), int(ids[0])))
    orders = []
    for ids in components:
        sub = a[ids][:, ids].astype(float)
        if len(ids) <= 12000:
            order = _spectral_order(sub, allow_symmetric=True)
        else:
            # Iterative Fiedler solve avoids the fill-in of a large sparse LU.
            lap = sp.csgraph.laplacian(sub).tocsr()
            degree = np.asarray(sub.sum(1)).ravel()
            rng = np.random.default_rng(seed)
            x = rng.standard_normal((len(ids), 1))
            pre = sp.diags(1.0 / np.maximum(degree, 1.0))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                values, vectors, residuals = sla.lobpcg(
                    lap,
                    x,
                    Y=np.ones((len(ids), 1)),
                    M=pre,
                    largest=False,
                    tol=1e-5,
                    maxiter=1000,
                    retResidualNormsHistory=True,
                )
            axis = vectors[:, 0]
            if axis @ (degree - degree.mean()) > 0:
                axis = -axis
            order = np.argsort(axis, kind="stable")
        orders.append(ids[order])
    order = np.concatenate(orders)
    ranks = np.arange(nodes, dtype=np.int64)
    mapping = np.empty(nodes, dtype=np.int64)
    mapping[order] = ranks ^ (ranks >> 1)
    padded = 1 << max(0, (nodes - 1).bit_length())
    mapping, _ = refine_mapping(a, mapping, padded, seed=seed)
    return torch.from_numpy(mapping)
