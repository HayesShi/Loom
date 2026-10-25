import numpy as np
import scipy.sparse as sp
import torch
from loom import compile_graph, gray_grid
from loom.compiler import hamming_cost, refine_mapping


def test_refinement_preserves_injectivity_and_descends():
    rng = np.random.default_rng(4)
    adjacency = sp.csr_matrix(rng.random((13, 13)) < 0.12)
    initial = rng.choice(16, 13, replace=False)
    mapping, audit = refine_mapping(adjacency, initial, 16, seed=3)
    assert len(np.unique(mapping)) == 13
    assert audit["accepted_moves"] > 0
    assert (
        hamming_cost(adjacency, mapping) == audit["final_cost"] <= audit["initial_cost"]
    )
    assert all(b < a for a, b in zip(audit["cost_history"], audit["cost_history"][1:]))


def test_components_and_isolated_nodes():
    edges = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    first = compile_graph(edges, 7)
    second = compile_graph(edges.flip(0), 7)
    assert len(torch.unique(first)) == 7
    assert int(first.min()) >= 0 and int(first.max()) < 8
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        compile_graph(torch.empty(2, 0, dtype=torch.long), 1),
        torch.zeros(1, dtype=torch.long),
    )


def test_grid_neighbors_differ_in_one_bit():
    xy = torch.cartesian_prod(torch.arange(8), torch.arange(8))
    mapping = gray_grid(xy, bits=3).reshape(8, 8)
    assert len(torch.unique(mapping)) == 64
    for differences in [mapping[1:] ^ mapping[:-1], mapping[:, 1:] ^ mapping[:, :-1]]:
        assert all(int(value).bit_count() == 1 for value in differences.flatten())
