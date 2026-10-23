import numpy as np
import scipy.sparse as sp
import torch

from gcm.architecture import compile_interfaces
from gcm.hamming_compiler import compiler_adjacency, hamming_cost, refine_hamming_mapping, swap_delta


def graph(n, edges):
    row, col = np.asarray(edges).T
    return compiler_adjacency(sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n)))


def test_exact_delta_matches_full_recomputation_including_shared_neighbors():
    rng = np.random.default_rng(8)
    for n in (4, 11, 23):
        a = compiler_adjacency(sp.csr_matrix(rng.random((n, n)) < .3))
        mapping = rng.choice(32, n, replace=False)
        before = hamming_cost(a, mapping)
        for u in range(n):
            for v in range(u + 1, n):
                moved = mapping.copy()
                moved[u], moved[v] = moved[v], moved[u]
                assert swap_delta(a, mapping, u, v) == hamming_cost(a, moved) - before


def test_user_example_and_tradeoff_with_other_incident_edges():
    mapping = np.arange(4)
    a = graph(4, [(1, 2)])
    assert swap_delta(a, mapping, 2, 3) == -1
    # Improving B-C alone can worsen total cost: A-C and B-D must count too.
    a = graph(4, [(1, 2), (0, 2), (1, 3)])
    assert swap_delta(a, mapping, 2, 3) == 1
    refined, audit = refine_hamming_mapping(graph(4, [(1, 2)]), mapping, 4)
    assert audit['final_cost'] == 1 < audit['initial_cost']
    assert sorted(refined) == list(mapping)


def test_descent_reproducibility_padding_and_duplicate_edge_semantics():
    rng = np.random.default_rng(3)
    a = compiler_adjacency(sp.csr_matrix(rng.random((80, 80)) < .025))
    initial = rng.choice(128, 80, replace=False)
    refined, stats = refine_hamming_mapping(a, initial, 128, seed=19)
    repeated, other = refine_hamming_mapping(a, initial, 128, seed=19)
    assert np.array_equal(refined, repeated) and stats == other
    assert np.array_equal(np.sort(initial), np.sort(refined))
    assert all(b < a for a, b in zip(stats['cost_history'], stats['cost_history'][1:]))
    assert stats['final_cost'] <= stats['initial_cost']
    assert hamming_cost(a * 3 + sp.eye(80), initial) == hamming_cost(a, initial)


def test_fiedler_pipeline_preserves_xor_schedule_and_improves_cost():
    n = 12
    edges = [(i, i + 1) for i in range(n - 1)] + [(0, 2), (1, 4), (3, 6), (6, 10)]
    edge_tensor = torch.tensor(edges).T
    before, _, pairs0 = compile_interfaces(edge_tensor, n, layout='graph_spectral')
    after, _, pairs = compile_interfaces(edge_tensor, n, layout='graph_spectral_hamming', seed=3)
    a = graph(n, edges)
    assert hamming_cost(a, after.numpy()) < hamming_cost(a, before.numpy())
    assert torch.equal(pairs, pairs0)
    for stage, pair in enumerate(pairs):
        assert torch.all((pair[0] ^ pair[1]) == 1 << stage)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(17))
    relabeled, _, _ = compile_interfaces(perm.argsort()[edge_tensor], n,
                                         layout='graph_spectral_hamming', seed=3)
    assert torch.equal(relabeled, after[perm])


def test_disconnected_symmetric_components_and_isolated_vertices():
    edges = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 0, 5]])
    mapping, _, pairs = compile_interfaces(edges, 8, layout='graph_spectral_hamming')
    assert sorted(mapping.tolist()) == list(range(8))
    assert pairs.shape == (3, 2, 4)
