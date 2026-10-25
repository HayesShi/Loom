import torch
from loom import ButterflyMixer, Loom, butterfly_pairs, normalized_neighbors


def test_grouped_mixing_matches_indexed_pairs_and_gradients():
    torch.manual_seed(7)
    mixer = ButterflyMixer(16, 2, 4).double()
    for pair in mixer.pairs:
        torch.nn.init.normal_(pair.gate.weight, std=0.1)
        torch.nn.init.normal_(pair.direction_gate.weight, std=0.1)
        pair.middle_delta.data.fill_(0.12)
    x = torch.randn(2, 16, 16, dtype=torch.double, requires_grad=True)
    actual = mixer(x)
    expected = x
    for stage, pair in enumerate(mixer.pairs):
        left, right = butterfly_pairs(16, stage)
        u, v = pair(expected[:, left], expected[:, right])
        expected = (
            torch.empty_like(expected).index_copy(1, left, u).index_copy(1, right, v)
        )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    params = [x] + list(mixer.parameters())
    ga = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), params)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


def test_embedding_and_persistent_padding():
    torch.manual_seed(3)
    x = torch.randn(5, 3)
    edges = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    mapping = torch.tensor([0, 3, 7, 2, 4])
    model = Loom(3, 2, 8, dim=8, heads=2, blocks=2, dropout=0.0)
    neighbors = normalized_neighbors(x, edges)
    degree = torch.bincount(edges[0], minlength=5).float()
    adjacency = torch.zeros(5, 5)
    adjacency[edges[1], edges[0]] = (degree[edges[0]] * degree[edges[1]]).rsqrt()
    h = model.embed_norm(
        model.self_projection(x) + adjacency @ model.graph_projection(x)
    )
    z = torch.zeros(8, 8).index_copy(0, mapping, h)
    for block in model.blocks:
        z = block(z)
    expected = model.head(model.head_norm(z[mapping]))
    torch.testing.assert_close(model(x, neighbors, mapping), expected)
    ids = torch.arange(2).repeat_interleave(5)
    actual = model(x.repeat(2, 1), neighbors.repeat(2, 1), mapping.repeat(2), ids, 8)
    torch.testing.assert_close(actual, expected.repeat(2, 1))


def test_directed_normalization_uses_out_and_in_degrees():
    edges = torch.tensor([[0, 0, 1], [1, 2, 2]])
    x = torch.eye(4)
    expected = torch.zeros(4, 4)
    expected[1, 0] = 2**-0.5
    expected[2, 0] = 0.5
    expected[2, 1] = 2**-0.5
    torch.testing.assert_close(normalized_neighbors(x, edges, directed=True), expected)


def test_graph_readout_pools_only_real_nodes():
    torch.manual_seed(11)
    model = Loom(3, 2, 4, dim=8, heads=2, blocks=1, dropout=0.0)
    x = torch.randn(5, 3)
    neighbors = torch.zeros_like(x)
    ids = torch.tensor([0, 0, 1, 1, 1])
    mapping = torch.tensor([0, 3, 2, 0, 1])
    h = model.encode(x, neighbors, mapping, ids, padded=4)
    pooled = torch.stack((h[:2].mean(0), h[2:].mean(0)))
    expected = model.head(model.head_norm(pooled))
    torch.testing.assert_close(
        model.forward_graph(x, neighbors, mapping, ids, 4), expected
    )
