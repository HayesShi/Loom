import math

import torch
from torch_geometric.data import Data

from gcm.layers import rotate90
from gcm.pure import PhasePair, PureCircuit, PureGCM


def test_zero_offset_lifting_matches_content_controlled_original_rotation():
    torch.manual_seed(17)
    original = PhasePair(12, 3, mode='original', angle=.2).double()
    lifting = PhasePair(12, 3, mode='lifting', angle=.2).double()
    with torch.no_grad():
        # The persistent phase buffer starts as float32 before .double().
        # Reset it at double precision to test the algebra rather than its
        # historical pi/2 rounding.
        original.phase.fill_(math.pi/2)
        original.gate.weight.normal_(std=.04)
        original.gate.bias.uniform_(-.1, .1)
        original.base.copy_(torch.tensor([.15, -.23, .4], dtype=torch.double))
        lifting.base.copy_(original.base)
        lifting.gate.weight.copy_(original.gate.weight)
        lifting.gate.bias.copy_(original.gate.bias)
    u = torch.randn(9, 12, dtype=torch.double)
    v = torch.randn_like(u)
    expected = original(u, v)
    actual = lifting(u, v)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, atol=2e-14, rtol=2e-14)


def test_lifting_can_preserve_either_source_while_transmitting_to_partner():
    torch.manual_seed(19)
    u = torch.randn(5, 8, dtype=torch.double)
    v = torch.randn_like(u)
    switch = PhasePair(8, 2, mode='lifting', angle=.2).double()
    with torch.no_grad():
        switch.base.fill_(.2)
        switch.middle_delta.fill_(math.atanh(-math.sin(.2)))
    out_u, out_v = switch(u, v)
    t = math.tan(.1)
    torch.testing.assert_close(out_u, u + 2*t*rotate90(v), atol=2e-15, rtol=2e-15)
    torch.testing.assert_close(out_v, v, atol=2e-15, rtol=2e-15)

    # t=0 leaves the left source untouched while the middle shear writes right.
    with torch.no_grad():
        switch.base.zero_()
        switch.middle_delta.fill_(.4)
    out_u, out_v = switch(u, v)
    torch.testing.assert_close(out_u, u)
    torch.testing.assert_close(out_v, v + math.tanh(.4)*rotate90(u))


def test_lifting_content_gate_and_independent_coefficients_receive_finite_gradients():
    torch.manual_seed(23)
    switch = PhasePair(8, 2, mode='lifting', angle=.2).double()
    u = torch.randn(7, 8, dtype=torch.double, requires_grad=True)
    v = torch.randn_like(u, requires_grad=True)
    out_u, out_v = switch(u, v)
    (out_u.square().mean() + .7*out_v.square().mean()).backward()
    for parameter in (switch.base, switch.middle_delta, switch.gate.weight,
                      switch.direction_gate.weight, u, v):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_lifting_runs_through_existing_butterfly_and_legacy_keys_remain_unchanged():
    ids = torch.arange(4)
    pairs = torch.stack([torch.stack([ids[(ids & (1 << stage)) == 0],
                                      ids[(ids & (1 << stage)) == 0] ^ (1 << stage)])
                         for stage in range(2)])
    circuit = PureCircuit(8, 2, pairs, mode='lifting', stage_ff=False).double()
    x = torch.randn(4, 8, dtype=torch.double, requires_grad=True)
    circuit(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert set(PhasePair(8, 2, mode='original').state_dict()) == {
        'base', 'phase', 'gate.weight', 'gate.bias'}


def test_same_seed_keeps_every_common_model_parameter_identical():
    ids = torch.arange(4)
    data = Data(x=torch.randn(4, 6), y=ids % 2,
                edge_index=torch.stack((ids, ids.roll(1))))
    options = dict(dim=8, heads=2, blocks=2, dropout=0.,
                   graph_embed=True, graph_refresh=True, seed=5)
    torch.manual_seed(37)
    original = PureGCM(data, mode='original', **options)
    torch.manual_seed(37)
    lifting = PureGCM(data, mode='lifting', **options)
    original_parameters = dict(original.named_parameters())
    lifting_parameters = dict(lifting.named_parameters())
    assert original_parameters.keys() <= lifting_parameters.keys()
    for name, parameter in original_parameters.items():
        assert torch.equal(parameter, lifting_parameters[name]), name
    assert any(name.endswith('direction_gate.weight') for name in lifting_parameters)
