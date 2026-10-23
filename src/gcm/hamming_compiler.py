"""Strictly descending address swaps for the unweighted compiler edge set."""
import numpy as np
import scipy.sparse as sp


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
    xor = np.asarray(mapping, dtype=np.int64)[u] ^ np.asarray(mapping, dtype=np.int64)[v]
    table = np.array([i.bit_count() for i in range(256)], dtype=np.int64)
    return int(sum(table[(xor >> bit) & 255].sum() for bit in range(0, 64, 8)))


def swap_delta(adjacency, mapping, u, v):
    """Exact new-minus-old cost; adjacency must be simple, symmetric CSR.

    Only edges incident to u or v change. Their mutual edge has the same
    XOR after swapping, so omit it from both incident lists.
    """
    if u == v:
        return 0
    wu, wv = int(mapping[u]), int(mapping[v])
    delta = 0
    for node, other, old, new in ((u, v, wu, wv), (v, u, wv, wu)):
        for neighbor in adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]:
            if neighbor != other:
                w = int(mapping[neighbor])
                delta += (new ^ w).bit_count() - (old ^ w).bit_count()
    return delta


def refine_hamming_mapping(adjacency, mapping, padded, seed=0,
                           max_passes=4, random_candidates=16):
    """Return an injective mapping and a reproducible descent audit.

    Visit nodes in seeded wire order. For <=64 nodes consider every partner;
    otherwise consider occupied one-bit wire neighbors and 16 seeded sampled
    partners. Apply the best strictly improving swap per visited node. Stop
    after a no-change sweep or four sweeps. Unoccupied addresses stay fixed.
    This bounded search makes no global or exhaustive local optimality claim.
    """
    a = compiler_adjacency(adjacency)
    mapping = np.asarray(mapping)
    n = a.shape[0]
    if (a.shape != (n, n) or mapping.shape != (n,)
            or not np.issubdtype(mapping.dtype, np.integer)
            or padded < max(1, n) or padded & (padded - 1)
            or len(np.unique(mapping)) != n
            or (mapping < 0).any() or (mapping >= padded).any()):
        raise ValueError('Expected a square graph and injective integer wire addresses')
    if max_passes < 0 or random_candidates < 0:
        raise ValueError('Search budgets must be nonnegative')
    mapping = mapping.astype(np.int64, copy=True)
    inverse = np.full(padded, -1, dtype=np.int64)
    inverse[mapping] = np.arange(n)
    occupied = np.sort(mapping)
    wire_rank = np.full(padded, -1, dtype=np.int64)
    wire_rank[occupied] = np.arange(n)
    rng = np.random.default_rng(seed)
    cost = hamming_cost(a, mapping)
    audit = dict(initial_cost=cost, final_cost=cost, accepted_swaps=0,
                 evaluated_swaps=0, passes=0, max_passes=max_passes,
                 random_candidates=random_candidates, cost_history=[cost])
    for sweep in range(max_passes if n > 1 else 0):
        changed = False
        # Sampling through wire order avoids using node IDs as search priorities.
        visit = inverse[occupied[rng.permutation(n)]].copy()
        for u in visit:
            if n <= 64:
                candidates = np.arange(n)
            else:
                candidates = [inverse[int(mapping[u]) ^ (1 << bit)]
                              for bit in range(padded.bit_length() - 1)]
                count = min(random_candidates, n - 1)
                indices = rng.choice(n - 1, size=count, replace=False)
                rank = int(wire_rank[mapping[u]])
                indices += indices >= rank
                candidates.extend(inverse[occupied[indices]])
            best, best_delta = None, 0
            for v in sorted(set(map(int, candidates)) - {-1, int(u)},
                            key=lambda node: int(mapping[node])):
                delta = swap_delta(a, mapping, int(u), v)
                audit['evaluated_swaps'] += 1
                if delta < best_delta:
                    best, best_delta = v, delta
            if best is not None:
                v = best
                wu, wv = int(mapping[u]), int(mapping[v])
                mapping[u], mapping[v] = wv, wu
                inverse[wu], inverse[wv] = v, u
                cost += best_delta
                audit['accepted_swaps'] += 1
                audit['cost_history'].append(cost)
                changed = True
        audit['passes'] = sweep + 1
        if not changed:
            break
    audit['final_cost'] = hamming_cost(a, mapping)
    if audit['final_cost'] != cost:
        raise AssertionError('Incremental and full Hamming costs disagree')
    return mapping, audit


# Native loops keep exact incident-edge deltas affordable on large graphs.
from numba import njit


@njit(cache=True)
def _popcount(value):
    result = 0
    while value:
        value &= value - 1
        result += 1
    return result


@njit(cache=True)
def _refine_native(indptr, indices, mapping, padded, seed, max_passes, random_candidates):
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


def refine_loom_mapping(adjacency, mapping, padded, seed=0, max_passes=4,
                        random_candidates=16):
    """Full compiler refinement including relocations into unused addresses."""
    a = compiler_adjacency(adjacency)
    mapping = np.asarray(mapping, dtype=np.int64).copy()
    n = a.shape[0]
    if (mapping.shape != (n,) or len(np.unique(mapping)) != n or
            padded < max(n, 1) or padded & (padded - 1) or
            np.any(mapping < 0) or np.any(mapping >= padded)):
        raise ValueError('Expected injective addresses in a power-of-two capacity')
    initial = hamming_cost(a, mapping)
    mapped, swaps, moves, evaluated, passes, deltas = _refine_native(
        a.indptr, a.indices, mapping, padded, seed, max_passes, random_candidates)
    final = hamming_cost(a, mapped)
    if final != initial + int(deltas.sum()):
        raise AssertionError('Incremental compiler cost disagrees with full cost')
    return mapped, dict(initial_cost=initial, final_cost=final,
                        accepted_swaps=int(swaps), accepted_moves=int(moves),
                        evaluated_candidates=int(evaluated), passes=int(passes),
                        cost_history=[initial] + (initial + np.cumsum(deltas)).tolist())
