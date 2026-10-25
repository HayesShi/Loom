import torch
from loom import ARCO
from loom.arco import rotate90


def test_fused_operator_matches_lifting_and_gradients():
    torch.manual_seed(17)
    pair = ARCO(16, 2).double()
    for parameter in pair.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    u = torch.randn(7, 16, dtype=torch.double, requires_grad=True)
    v = torch.randn_like(u, requires_grad=True)
    actual = pair(u, v)
    t, s = pair.coefficients(u, v)
    left, right = u.reshape(7, 2, 8), v.reshape(7, 2, 8)
    left = left + t * rotate90(right)
    right = right + s * rotate90(left)
    left = left + t * rotate90(right)
    expected = left.reshape_as(u), right.reshape_as(v)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    params = [u, v] + list(pair.parameters())
    ga = torch.autograd.grad(
        sum(a.square().sum() for a in actual), params, retain_graph=True
    )
    gb = torch.autograd.grad(sum(b.square().sum() for b in expected), params)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


def test_rotation_conserves_pair_energy():
    torch.manual_seed(5)
    pair = ARCO(12, 3).double()
    torch.nn.init.normal_(pair.gate.weight, std=0.1)
    torch.nn.init.normal_(pair.direction_gate.weight, std=0.1)
    u, v = torch.randn(2, 19, 12, dtype=torch.double)
    left, right = pair(u, v)
    torch.testing.assert_close(
        left.square().sum(-1) + right.square().sum(-1),
        u.square().sum(-1) + v.square().sum(-1),
    )


def test_zero_middle_coefficient_preserves_source():
    pair = ARCO(8, 2).double()
    with torch.no_grad():
        pair.middle_delta.copy_(torch.atanh(-pair.base.sin()))
    u, v = torch.randn(2, 5, 8, dtype=torch.double)
    left, right = pair(u, v)
    t, _ = pair.coefficients(u, v)
    expected = u.reshape(5, 2, 4) + 2 * t * rotate90(v.reshape(5, 2, 4))
    torch.testing.assert_close(right, v)
    torch.testing.assert_close(left, expected.reshape_as(u))
