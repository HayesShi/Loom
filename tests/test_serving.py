import pytest
import torch

from gcm import Loom, LoomSession, PolicyCost, normalized_neighbors


def edge_index(edges):
    return torch.tensor(edges + [(v, u) for u, v in edges], dtype=torch.long).reshape(-1, 2).T


def test_versioned_updates_match_full_aggregation():
    torch.manual_seed(4)
    model = Loom(3, 2, 8, dim=8, heads=2, blocks=1, dropout=0.).eval()
    x = torch.randn(5, 3)
    mapping = torch.tensor([5, 0, 2, 4, 3])
    initial = edge_index([(0, 1), (1, 2), (2, 3)])
    sessions = [LoomSession(model, x, initial, mapping, retain=retain) for retain in (True, False)]
    for session in sessions:
        with pytest.raises(ValueError):
            session.publish(delete=[(0, 1), (3, 4)])
        assert session.version == 0

    replacement = -x[3]
    for session in sessions:
        session.publish(delete=[(0, 1), (1, 2)], insert=[(0, 2)], features={3: replacement})
    x[3] = replacement
    edges = edge_index([(0, 2), (2, 3)])
    expected = model(x, normalized_neighbors(x, edges), mapping, padded=8)
    for session in sessions:
        version, logits = session.read()
        assert version == 1
        torch.testing.assert_close(logits, expected)
        session.publish(delete=[(0, 2), (2, 3)])
        torch.testing.assert_close(session.read()[1], model(x, torch.zeros_like(x), mapping, padded=8))


def test_lifecycle_crossover_accounts_for_build_and_strict_ties():
    baseline = PolicyCost(build=0, publish=1, read=5)
    retained = PolicyCost(build=12, publish=3, read=2)
    assert retained.crossover(baseline, versions=3) == 3
    assert retained.total(3, 2) == baseline.total(3, 2)
    assert retained.total(3, 3) < baseline.total(3, 3)
    assert PolicyCost(2, 2, 5).crossover(baseline, versions=1) is None
