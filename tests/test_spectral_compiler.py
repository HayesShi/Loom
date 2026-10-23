import torch

from gcm.architecture import compile_interfaces


def test_spectral_order_follows_topology_after_relabeling():
    n = 12
    undirected = [(i, i + 1) for i in range(n - 1)]
    undirected += [(0, 2), (1, 4), (3, 6), (6, 10)]
    edges = torch.tensor(undirected, dtype=torch.long).T
    edges = torch.cat((edges, edges.flip(0)), dim=1)
    original, _, _ = compile_interfaces(edges, n, layout="graph_spectral")

    new_to_old = torch.randperm(n, generator=torch.Generator().manual_seed(17))
    old_to_new = new_to_old.argsort()
    relabeled_edges = old_to_new[edges]
    relabeled, _, _ = compile_interfaces(relabeled_edges, n,
                                         layout="graph_spectral")
    assert torch.equal(relabeled, original[new_to_old])
