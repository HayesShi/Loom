"""One existing J switch can be reused at every regular circuit stage."""

import torch
from torch_geometric.data import Batch, Data

from gcm.multigraph import MultiGraphGCM


def _graph(features: list[float]) -> Data:
    nodes = len(features)
    source = torch.arange(nodes)
    target = (source + 1) % nodes
    edges = torch.stack((torch.cat((source, target)),
                         torch.cat((target, source))))
    return Data(x=torch.tensor(features)[:, None], edge_index=edges)


def _model(repeat: bool | None, exchange_scale: float) -> MultiGraphGCM:
    options = dict(
        input_dim=1, edge_dim=None, output_dim=1, dim=8, heads=2,
        blocks=1, max_stages=3, mode="original", angle=.2,
        exchange_scale=exchange_scale, dropout=0., stage_ff=True,
        graph_embed=False, graph_refresh=False, layout="input_order",
    )
    if repeat is not None:
        options["repeat_pair_switch"] = repeat
    return MultiGraphGCM(**options)


def test_repeated_pair_switch_trains_on_small_then_upper_stage_of_large_graph():
    torch.manual_seed(43)
    default = _model(None, 0.)
    torch.manual_seed(43)
    explicit_false = _model(False, 0.)
    assert list(default.state_dict()) == list(explicit_false.state_dict())
    assert all(torch.equal(default.state_dict()[key], explicit_false.state_dict()[key])
               for key in default.state_dict())

    torch.manual_seed(43)
    repeated = _model(True, 0.)
    assert list(default.state_dict()) == list(repeated.state_dict())
    assert all(torch.equal(default.state_dict()[key], repeated.state_dict()[key])
               for key in default.state_dict())
    switches = repeated.blocks[0].circuit.switches
    assert len(switches) == 3
    assert len(list(switches[0].parameters())) == 3
    assert sum(p.numel() for p in repeated.parameters()) == sum(
        p.numel() for p in default.parameters())

    small = _graph([.2, -.7, 1.3, .4])
    large = _graph([.6, -1.1, .3, 1.7, -.4, .9, -.8, 1.2])
    mappings = [torch.arange(4), torch.arange(8)]
    control_batch = Batch.from_data_list([small, large])
    repeated_batch = Batch.from_data_list([small, large])
    control_batch.x.requires_grad_()
    repeated_batch.x.requires_grad_()
    control = default.predict_nodes(control_batch, mappings=mappings)
    candidate = repeated.predict_nodes(repeated_batch, mappings=mappings)
    torch.testing.assert_close(candidate, control, rtol=0., atol=0.)
    weights = torch.linspace(-1., 1., control.numel()).reshape_as(control)
    control_grad = torch.autograd.grad((control * weights).sum(), control_batch.x)[0]
    candidate_grad = torch.autograd.grad(
        (candidate * weights).sum(), repeated_batch.x)[0]
    torch.testing.assert_close(candidate_grad, control_grad, rtol=0., atol=0.)

    repeated.exchange_scale = 1.
    optimizer = torch.optim.SGD(switches.parameters(), lr=.05)
    small_batch = Batch.from_data_list([small])
    small_target = torch.tensor([[.5], [-1.], [.3], [2.]])
    small_prediction = repeated.predict_nodes(
        small_batch, mappings=[torch.arange(4)])
    (small_prediction * small_target).sum().backward()
    assert switches[0].base.grad.abs().sum() > 0
    assert switches[0].gate.weight.grad.abs().sum() > 0
    assert all(p.grad is None for switch in switches[1:]
               for p in switch.parameters())
    before_step = switches[0].base.detach().clone()
    optimizer.step()
    assert not torch.equal(switches[0].base, before_step)

    optimizer.zero_grad()
    large_batch = Batch.from_data_list([large])
    # Zero angle at lower stages makes the highest stage the only source of
    # switch-parameter gradients; it was never active on the small graph.
    upper_only = torch.zeros(3, 4)
    upper_only[2] = 1.
    large_prediction = repeated.predict_nodes(
        large_batch, mappings=[torch.arange(8)],
        pair_angle_scales=upper_only)
    large_target = torch.tensor([[.3], [-1.], [.8], [.2], [1.1], [-.5], [.7], [-.4]])
    (large_prediction * large_target).sum().backward()
    assert switches[0].base.grad.abs().sum() > 0
    assert switches[0].gate.weight.grad.abs().sum() > 0
    assert all(p.grad is None for switch in switches[1:]
               for p in switch.parameters())
