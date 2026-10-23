"""The optional interstage wiring changes no learned switch or pair geometry."""

from argparse import Namespace
import math
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from gcm.multigraph import MultiGraphGCM, _VariableSizeCircuit
from scripts.diagnose_lrim_interstage_head_shuffle import _frozen_kernels
from scripts.run_lrim import _config, _make_model, _parameter_digest


def _graph(nodes: int) -> Data:
    source = torch.arange(nodes)
    target = (source + 1) % nodes
    edges = torch.stack((torch.cat((source, target)),
                         torch.cat((target, source))))
    return Data(x=torch.arange(nodes, dtype=torch.float32)[:, None] / nodes,
                edge_index=edges)


def _model(scale: float, reverse: bool = False,
           exempt: int = 0) -> MultiGraphGCM:
    return MultiGraphGCM(
        input_dim=1, edge_dim=None, output_dim=1, dim=16, heads=4,
        blocks=2, max_stages=3, mode="original", exchange_scale=scale,
        dropout=0., layout="input_order", reverse_odd_blocks=reverse,
        interstage_head_shuffle=True, shuffle_exempt_heads=exempt,
    ).eval()


def test_default_circuit_and_explicit_false_are_bitwise_identical():
    torch.manual_seed(17)
    default = _VariableSizeCircuit(16, 4, 3, "original", .2, False).eval()
    torch.manual_seed(17)
    explicit = _VariableSizeCircuit(
        16, 4, 3, "original", .2, False,
        interstage_head_shuffle=False).eval()
    state = torch.randn(2, 8, 16)
    mask = torch.ones(2, 8, dtype=torch.bool)
    assert set(default.state_dict()) == set(explicit.state_dict())
    assert all(torch.equal(default.state_dict()[key], explicit.state_dict()[key])
               for key in default.state_dict())
    assert torch.equal(default(state, mask), explicit(state, mask))
    assert torch.equal(default(state, mask, pair_scale=0.),
                       explicit(state, mask, pair_scale=0.))


def test_bypassed_shuffled_circuit_and_complete_zero_arm_are_exactly_legacy():
    torch.manual_seed(19)
    circuit = _VariableSizeCircuit(
        16, 4, 3, "original", .2, False,
        interstage_head_shuffle=True).eval()
    state = torch.randn(3, 8, 16)
    mask = torch.ones(3, 8, dtype=torch.bool)
    active = torch.tensor([3, 2, 1])
    assert torch.equal(circuit(state, mask, pair_scale=0.,
                               active_stages=active), state)
    assert torch.equal(circuit(state, mask, pair_scale=0.), state)

    torch.manual_seed(20)
    legacy = MultiGraphGCM(
        input_dim=1, edge_dim=None, output_dim=1, dim=16, heads=4,
        blocks=2, max_stages=3, mode="original", exchange_scale=0.,
        dropout=0., layout="input_order").eval()
    torch.manual_seed(20)
    shuffled = _model(0.)
    assert all(torch.equal(legacy.state_dict()[key], shuffled.state_dict()[key])
               for key in legacy.state_dict())
    batch = Batch.from_data_list([_graph(4), _graph(8)])
    mappings = [torch.arange(4), torch.arange(8)]
    with torch.no_grad():
        expected = legacy.predict_nodes(batch, mappings=mappings)
        actual = shuffled.predict_nodes(batch, mappings=mappings)
    assert torch.equal(expected, actual)
    legacy.predict_nodes(batch, mappings=mappings).square().sum().backward()
    shuffled.predict_nodes(batch, mappings=mappings).square().sum().backward()
    for (legacy_name, legacy_parameter), (new_name, new_parameter) in zip(
            legacy.named_parameters(), shuffled.named_parameters()):
        assert legacy_name == new_name
        if legacy_parameter.grad is None:
            assert new_parameter.grad is None
        else:
            assert torch.equal(legacy_parameter.grad, new_parameter.grad)


@pytest.mark.parametrize("dim,heads", [(16, 4), (8, 4), (16, 1)])
def test_permutation_keeps_real_imaginary_pairs_intact(dim, heads):
    circuit = _VariableSizeCircuit(
        dim, heads, 3, "original", .2, False,
        interstage_head_shuffle=True)
    pairs = circuit.interstage_shuffle.reshape(-1, 2)
    assert torch.equal(torch.sort(circuit.interstage_shuffle).values,
                       torch.arange(dim))
    assert torch.equal(pairs[:, 1], pairs[:, 0] + 1)
    assert torch.equal(pairs[:, 0] % 2, torch.zeros(dim // 2, dtype=torch.long))
    if heads > 1:
        origin_head = pairs[:, 0] // (dim // heads)
        destination_head = torch.arange(dim // 2) // (dim // (2 * heads))
        assert bool((origin_head != destination_head).any())


def test_carrier_head_stays_fixed_while_other_complex_pairs_visit_heads():
    circuit = _VariableSizeCircuit(
        64, 4, 10, "original", .2, False,
        interstage_head_shuffle=True, shuffle_exempt_heads=1)
    permutation = circuit.interstage_shuffle
    pairs = permutation.reshape(-1, 2)
    assert torch.equal(permutation[:16], torch.arange(16))
    assert torch.equal(torch.sort(permutation).values, torch.arange(64))
    assert torch.equal(pairs[:, 1], pairs[:, 0] + 1)
    assert torch.equal(pairs[:, 0] % 2, torch.zeros(32, dtype=torch.long))
    assert bool((pairs[8:, 0] // 16 != torch.arange(8, 32) // 8).any())
    # The fixed carrier has the same head assignment at every J resolution.
    cumulative = torch.arange(64)
    for _ in range(9):
        cumulative = cumulative[permutation]
        assert torch.equal(cumulative[:16], torch.arange(16))


def test_ten_stage_carrier_retains_alpha_over_32_long_range_amplitude():
    alpha = .25
    def circuit(exempt: int) -> _VariableSizeCircuit:
        result = _VariableSizeCircuit(
            8, 4, 10, "original", .2, False,
            interstage_head_shuffle=True,
            shuffle_exempt_heads=exempt).double().eval()
        with torch.no_grad():
            for switch in result.switches:
                switch.base.copy_(torch.tensor(
                    [math.pi / 4, .2, .2, .2], dtype=torch.float64))
        return result

    anchored = circuit(1)
    all_shuffled = circuit(0)
    last_anchored = None
    with torch.no_grad():
        for stages in range(1, 11):
            nodes = 1 << stages
            state = torch.zeros(1, nodes, 8, dtype=torch.float64)
            state[0, 0, 0] = 1.
            mask = torch.ones(1, nodes, dtype=torch.bool)
            output = anchored(state, mask)
            far_pair = output[0, -1, :2]
            amplitude = alpha * far_pair.norm().item()
            assert amplitude == pytest.approx(alpha * 2 ** (-stages / 2),
                                              rel=1e-12, abs=1e-14)
            last_anchored = amplitude
        shuffled_output = all_shuffled(state, mask)
        shuffled_amplitude = alpha * shuffled_output[0, -1, :2].norm().item()
    assert last_anchored == pytest.approx(alpha / 32, rel=1e-12)
    assert shuffled_amplitude < last_anchored / 10


def test_carrier_exemption_leaves_zero_exchange_output_and_state_dict_exact():
    torch.manual_seed(47)
    legacy = _model(0., exempt=0)
    torch.manual_seed(47)
    preserved = _model(0., exempt=1)
    assert set(legacy.state_dict()) == set(preserved.state_dict())
    assert all(torch.equal(legacy.state_dict()[key], preserved.state_dict()[key])
               for key in legacy.state_dict())
    batch = Batch.from_data_list([_graph(4), _graph(8)])
    mappings = [torch.arange(4), torch.arange(8)]
    with torch.no_grad():
        assert torch.equal(legacy.predict_nodes(batch, mappings=mappings),
                           preserved.predict_nodes(batch, mappings=mappings))


@pytest.mark.parametrize("exempt,shuffle", [(1, False), (-1, True),
                                             (4, True), (True, True)])
def test_invalid_carrier_exemption_is_rejected(exempt, shuffle):
    with pytest.raises(ValueError, match="shuffle_exempt_heads"):
        MultiGraphGCM(input_dim=1, edge_dim=None, output_dim=1, dim=8,
                      heads=4, interstage_head_shuffle=shuffle,
                      shuffle_exempt_heads=exempt)


def test_same_initialized_parameters_for_zero_and_j_arms_and_runner_config():
    args = Namespace(
        dim=16, heads=4, blocks=2, dropout=0., angle=.2,
        layout="torus_gray", tie_pair_gate=False,
        pair_distance_angle=False, interstage_head_shuffle=True,
        magnetization_control=False, batch_size=4, lr=.001,
        weight_decay=1e-5, clip_norm=1., epochs=2, patience=1,
    )
    torch.manual_seed(23)
    zero = _make_model(args, 16, 0, 0.)
    torch.manual_seed(23)
    exchanged = _make_model(args, 16, 0, .25)
    assert _parameter_digest(zero) == _parameter_digest(exchanged)
    assert sum(p.numel() for p in zero.parameters()) == sum(
        p.numel() for p in exchanged.parameters())
    for scale in (0., .25):
        config = _config(args, "lrim_16_0.6_10k", 16, Path("."), 0, scale)
        assert "complete_real_imag_pair" in config["model"]["interstage_head_shuffle"]
    args.interstage_head_shuffle = False
    legacy = _config(args, "lrim_16_0.6_10k", 16, Path("."), 0, .25)
    assert "interstage_head_shuffle" not in legacy["model"]


@pytest.mark.parametrize("scale", [0., .25])
@pytest.mark.parametrize("reverse", [False, True])
def test_short_graph_output_is_independent_of_larger_graph_in_batch(scale, reverse):
    torch.manual_seed(31)
    model = _model(scale, reverse)
    small = _graph(4)
    large = _graph(8)
    with torch.no_grad():
        alone = model.predict_nodes(Batch.from_data_list([small]),
                                    mappings=[torch.arange(4)])
        together = model.predict_nodes(
            Batch.from_data_list([small, large]),
            mappings=[torch.arange(4), torch.arange(8)])[:4]
    torch.testing.assert_close(alone, together, rtol=1e-6, atol=1e-6)


def test_shuffled_j_remains_equivariant_when_compiled_wires_follow_relabeling():
    torch.manual_seed(37)
    model = _model(.25)
    graph = _graph(8)
    permutation = torch.tensor([5, 1, 7, 2, 0, 6, 3, 4])
    relabeled_x = torch.empty_like(graph.x)
    relabeled_x[permutation] = graph.x
    relabeled = Data(x=relabeled_x, edge_index=permutation[graph.edge_index])
    relabeled_mapping = torch.empty(8, dtype=torch.long)
    relabeled_mapping[permutation] = torch.arange(8)
    with torch.no_grad():
        expected = model.predict_nodes(Batch.from_data_list([graph]),
                                       mappings=[torch.arange(8)])
        actual = model.predict_nodes(Batch.from_data_list([relabeled]),
                                     mappings=[relabeled_mapping])
    torch.testing.assert_close(expected, actual[permutation],
                               rtol=1e-6, atol=1e-6)


def test_interstage_shuffle_does_not_activate_stage_ff():
    circuit = _VariableSizeCircuit(16, 4, 3, "original", .2, False,
                                   interstage_head_shuffle=True)
    assert not circuit.stage_ff
    assert not any("gains" in key or "biases" in key
                   for key in circuit.state_dict())
    with pytest.raises(ValueError, match="stage_ff=False"):
        _VariableSizeCircuit(16, 4, 3, "original", .2, True,
                             interstage_head_shuffle=True)


@pytest.mark.parametrize("shuffled", [False, True])
def test_frozen_operator_probe_matches_actual_j_circuit(shuffled):
    circuit = _VariableSizeCircuit(
        16, 4, 3, "original", .2, False,
        interstage_head_shuffle=shuffled).double().eval()
    angles = torch.tensor([[.13, .19, .23, .31],
                           [.11, .17, .29, .37],
                           [.07, .21, .33, .41]], dtype=torch.float64)
    scales = torch.tensor([[1., .8, .7, .6],
                           [.9, .5, .8, .7],
                           [.4, .6, .9, 1.]], dtype=torch.float64)
    with torch.no_grad():
        for level, switch in enumerate(circuit.switches):
            switch.base.copy_(angles[level])
            switch.phase.fill_(math.pi / 2)
    pair_order = (circuit.interstage_shuffle[::2] // 2).numpy()
    kernels, _ = _frozen_kernels(angles.numpy(), scales.numpy(),
                                 pair_order, shuffled, False)
    generator = torch.Generator().manual_seed(43)
    for pair in (0, 5):
        input_real = torch.randn(8, generator=generator, dtype=torch.float64)
        input_imag = torch.randn(8, generator=generator, dtype=torch.float64)
        state = torch.zeros(1, 8, 16, dtype=torch.float64)
        state[0, :, 2 * pair] = input_real
        state[0, :, 2 * pair + 1] = input_imag
        with torch.no_grad():
            actual = circuit(state, torch.ones(1, 8, dtype=torch.bool),
                             pair_angle_scales=scales)[0, :, 2 * pair:2 * pair + 2]
        expected = kernels[pair] @ (input_real.numpy() + 1j * input_imag.numpy())
        torch.testing.assert_close(actual[:, 0], torch.from_numpy(expected.real),
                                   rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(actual[:, 1], torch.from_numpy(expected.imag),
                                   rtol=1e-11, atol=1e-11)
