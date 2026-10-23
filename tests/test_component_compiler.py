import torch

from gcm.architecture import compile_interfaces


def test_component_layout_keeps_grid_tiles_in_64_wire_groups():
    # A 64x64 grid with the links between 8x8 tiles removed has 64 separate
    # 64-node components. The first six circuit stages should stay in a tile.
    edges = []
    tiles = [[] for _ in range(64)]
    for row in range(64):
        for column in range(64):
            node = row * 64 + column
            tile = (row // 8) * 8 + column // 8
            tiles[tile].append(node)
            if row % 8 != 7:
                edges.append((node, node + 64))
            if column % 8 != 7:
                edges.append((node, node + 1))

    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    node_to_wire, _, _ = compile_interfaces(edge_index, 4096, layout="graph_components")
    tile_groups = []
    for tile in tiles:
        groups = torch.unique(node_to_wire[tile] // 64)
        assert groups.numel() == 1
        tile_groups.append(groups.item())
    assert len(set(tile_groups)) == 64


def test_component_layout_agrees_with_graph_layout_on_complete_graph():
    source, target = torch.triu_indices(16, 16, offset=1)
    edge_index = torch.stack((source, target))
    regular = compile_interfaces(edge_index, 16, layout="graph")
    component_aware = compile_interfaces(edge_index, 16, layout="graph_components")
    for actual, expected in zip(component_aware, regular):
        torch.testing.assert_close(actual, expected)


def test_component_layout_preserves_padding_and_wires():
    # Both components fit whole into one 64-wire half despite their unequal
    # sizes; the remaining 58 slots are dummy wires.
    edges = [(node, node + 1) for node in range(63)]
    edges += [(node, node + 1) for node in range(64, 69)]
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    node_to_wire, roles, pairs = compile_interfaces(edge_index, 70, layout="graph_components")

    assert node_to_wire.shape == (70,)
    assert torch.unique(node_to_wire).numel() == 70
    assert node_to_wire.min() >= 0
    assert node_to_wire.max() < 128
    assert roles.shape == (70, 3)
    assert torch.isfinite(roles).all()
    assert pairs.shape == (7, 2, 64)
    assert torch.unique(node_to_wire[:64] // 64).numel() == 1
    assert torch.unique(node_to_wire[64:] // 64).numel() == 1
    assert (node_to_wire[0] // 64) != (node_to_wire[64] // 64)
    for left, right in pairs:
        torch.testing.assert_close(torch.sort(torch.cat((left, right))).values,
                                   torch.arange(128))
