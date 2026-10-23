import torch
from torch_geometric.data import Batch, Data

from gcm.multigraph import MultiGraphGCM, _VariableSizeCircuit


def _batch():
    return Batch.from_data_list([
        Data(x=torch.arange(3, dtype=torch.float32)[:, None],
             edge_index=torch.empty((2, 0), dtype=torch.long)),
        Data(x=torch.arange(5, dtype=torch.float32)[:, None],
             edge_index=torch.empty((2, 0), dtype=torch.long)),
    ])


def test_empty_wire_relays_signal_through_original_j_switch():
    circuit = _VariableSizeCircuit(
        dim=2, heads=1, max_stages=2, mode="original", angle=.2,
        stage_ff=False,
    )
    z = torch.zeros((1, 4, 2))
    z[0, 2, 0] = 1.
    real_mask = torch.tensor([[True, True, True, False]])
    relay_mask = torch.ones_like(real_mask)

    legacy = circuit(z, real_mask)
    relayed = circuit(z, real_mask, relay_mask=relay_mask)
    zero_exchange = circuit(z, real_mask, pair_scale=0.,
                            relay_mask=relay_mask)

    # Wire 2 can reach wire 1 only by using empty wire 3 between stages.
    assert torch.equal(legacy[0, 1], torch.zeros(2))
    assert relayed[0, 1].norm() > 1e-4
    assert torch.equal(zero_exchange, z)


def test_relay_stays_inside_each_graph_and_zero_exchange_matches_legacy():
    torch.manual_seed(3)
    args = dict(input_dim=1, edge_dim=None, output_dim=1, dim=8,
                heads=2, blocks=2, max_stages=3, mode="phase",
                exchange_scale=0., graph_embed=False, graph_refresh=False,
                stage_ff=True, dropout=0.)
    legacy = MultiGraphGCM(**args)
    relay = MultiGraphGCM(**args, relay_padding=True)
    relay.load_state_dict(legacy.state_dict())
    batch = _batch()
    mappings = [torch.arange(3), torch.arange(5)]

    old_nodes, old_diag = legacy.encode_nodes(
        batch, mappings, return_diagnostics=True)
    new_nodes, new_diag = relay.encode_nodes(
        batch, mappings, return_diagnostics=True)

    torch.testing.assert_close(new_nodes, old_nodes, rtol=0., atol=0.)
    torch.testing.assert_close(old_diag["active_pairs_per_stage"],
                               torch.tensor([[1., 1., 0.], [2., 2., 1.]]))
    torch.testing.assert_close(new_diag["active_pairs_per_stage"],
                               torch.tensor([[2., 2., 0.], [4., 4., 4.]]))
