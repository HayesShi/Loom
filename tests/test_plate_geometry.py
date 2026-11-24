"""Focused checks for Plate's label-free, fixed-grid wire compiler."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from gcm.multigraph import MultiGraphGCM
from gcm.plate_geometry import (WIRES, _binary_wire_grid_coordinates,
                                _interleaved_binary, _interleaved_gray,
                                _wire_grid_coordinates,
                                compile_plate_binary_contact,
                                compile_plate_gray_contact,
                                compile_plate_reference,
                                compile_plate_gray_spaced_contact,
                                compile_plate_gray_distance_angle_scales)


SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_plate_deformation.py"
SPEC = importlib.util.spec_from_file_location("run_plate_deformation_geometry", SOURCE)
assert SPEC is not None and SPEC.loader is not None
plate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plate)


def _dummy_plate(shift: float = 0.) -> Data:
    grid = torch.tensor([(x, y) for x in range(29) for y in range(29)],
                        dtype=torch.float32)
    mesh_pos = torch.cat((5. * grid, torch.zeros((841, 1))), dim=1)
    force_pos = torch.tensor([[55. + shift, 56., 0.],
                              [110., 35. + shift, 0.]])
    pos = torch.cat((mesh_pos, force_pos), dim=0)
    node_type = torch.zeros(843, dtype=torch.long)
    node_type[torch.arange(841)[(grid[:, 0] == 0) |
                                  (grid[:, 0] == 28) |
                                  (grid[:, 1] == 0) |
                                  (grid[:, 1] == 28)]] = 2
    node_type[841:] = 1
    props = torch.zeros((843, 6))
    props[torch.arange(843), node_type] = 1.
    props[841, 3] = 1.
    props[842, 4] = 1.
    # Bidirectional grid edges and directed force-to-mesh contacts.
    pairs = []
    for x in range(29):
        for y in range(29):
            source = 29 * x + y
            if x < 28:
                pairs.extend(((source, source + 29), (source + 29, source)))
            if y < 28:
                pairs.extend(((source, source + 1), (source + 1, source)))
    for force in (841, 842):
        near = torch.cdist(pos[force:force + 1, :2], mesh_pos[:, :2])
        for vertex in near[0].topk(5, largest=False).indices.tolist():
            pairs.append((force, vertex))
    edge_index = torch.tensor(pairs, dtype=torch.long).T.contiguous()
    u0 = torch.cat((pos, torch.zeros_like(pos)), dim=1)
    return Data(x=props.clone(), pos=pos, u0=u0, properties=props,
                edge_index=edge_index,
                edge_attr=torch.zeros((edge_index.shape[1], 9)),
                node_type=node_type, loss_mask=node_type != 1,
                y=u0.clone(), mesh_node_count=841)


def test_mesh_wires_are_stable_and_adjacent_grid_vertices_pair() -> None:
    graph = _dummy_plate()
    other = _dummy_plate(7.)
    mapping = compile_plate_gray_contact(graph)
    other_mapping = compile_plate_gray_contact(other)
    assert torch.equal(compile_plate_reference(graph), mapping)
    assert torch.equal(compile_plate_reference(other), other_mapping)
    assert len(mapping) == graph.num_nodes
    assert mapping.unique().numel() == graph.num_nodes
    assert bool(((mapping >= 0) & (mapping < WIRES)).all())
    torch.testing.assert_close(mapping[:841], other_mapping[:841])
    for x in range(29):
        for y in range(29):
            node = 29 * x + y
            if x < 28:
                assert (int(mapping[node]) ^ int(mapping[node + 29])).bit_count() == 1
            if y < 28:
                assert (int(mapping[node]) ^ int(mapping[node + 1])).bit_count() == 1


def test_distinguishable_forces_and_mesh_are_node_relabel_invariant() -> None:
    graph = _dummy_plate()
    original = compile_plate_gray_contact(graph)
    permutation = torch.randperm(graph.num_nodes,
                                 generator=torch.Generator().manual_seed(17))
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(graph.num_nodes)
    relabeled = graph.clone()
    for key in ("x", "pos", "u0", "properties", "node_type",
                "loss_mask", "y"):
        setattr(relabeled, key, getattr(graph, key)[permutation])
    relabeled.edge_index = inverse[graph.edge_index]
    compiled = compile_plate_gray_contact(relabeled)
    torch.testing.assert_close(compiled[inverse], original)


def test_spaced_gray_uses_full_axis_and_keeps_most_mesh_edges_direct() -> None:
    graph = _dummy_plate()
    mapping = compile_plate_gray_spaced_contact(graph)
    assert mapping.numel() == graph.num_nodes
    assert mapping.unique().numel() == graph.num_nodes
    assert bool(((mapping >= 0) & (mapping < WIRES)).all())

    q = [round(i * 31 / 28) for i in range(29)]
    assert sorted(set(range(32)) - set(q)) == [5, 15, 26]
    direct_edges = 0
    for x in range(29):
        for y in range(29):
            node = 29 * x + y
            wire = int(mapping[node])
            assert wire == _interleaved_gray(q[x], q[y])
            assert _wire_grid_coordinates(wire) == (q[x], q[y])
            if x < 28:
                direct_edges += (wire ^ int(mapping[node + 29])).bit_count() == 1
            if y < 28:
                direct_edges += (wire ^ int(mapping[node + 1])).bit_count() == 1
    assert direct_edges == 1450
    assert direct_edges / (2 * 29 * 28) == pytest.approx(25 / 28)

    # The original gauge and its 100% adjacent-edge pairing are untouched.
    original = compile_plate_gray_contact(graph)
    assert original[:841].unique().numel() == 841
    assert not torch.equal(original[:841], mapping[:841])


def test_spaced_gray_force_placement_is_valid_and_relabel_invariant() -> None:
    graph = _dummy_plate()
    mapping = compile_plate_gray_spaced_contact(graph)
    torch.testing.assert_close(
        compile_plate_gray_contact(graph, mesh_layout="gray_spaced"), mapping)
    assert set(mapping[841:].tolist()).isdisjoint(set(mapping[:841].tolist()))
    assert all(0 <= int(wire) < WIRES for wire in mapping[841:])
    torch.testing.assert_close(compile_plate_gray_spaced_contact(graph), mapping)

    relabeled = graph.clone()
    permutation = torch.randperm(graph.num_nodes,
                                 generator=torch.Generator().manual_seed(29))
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(graph.num_nodes)
    for key in ("x", "pos", "u0", "properties", "node_type",
                "loss_mask", "y"):
        setattr(relabeled, key, getattr(graph, key)[permutation])
    relabeled.edge_index = inverse[graph.edge_index].flip(0).flip(1)
    relabeled.y = torch.randn_like(relabeled.y)
    torch.testing.assert_close(
        compile_plate_gray_spaced_contact(relabeled)[inverse], mapping)


def test_spaced_gray_free_wire_positions_use_rescaled_physical_axis() -> None:
    graph = _dummy_plate()
    graph.pos[841, :2] = torch.tensor([95., 0.])
    nearest = compile_plate_gray_contact(
        graph, mesh_layout="gray_spaced", force_layout="nearest")
    assert _wire_grid_coordinates(int(nearest[841])) == (26, 0)


def test_binary_mesh_gauge_is_fixed_injective_and_label_free() -> None:
    graph = _dummy_plate()
    shifted = _dummy_plate(7.)
    mapping = compile_plate_binary_contact(graph)
    changed_forces = compile_plate_binary_contact(shifted)
    assert mapping.numel() == graph.num_nodes
    assert mapping.unique().numel() == graph.num_nodes
    assert bool(((mapping >= 0) & (mapping < WIRES)).all())
    torch.testing.assert_close(mapping[:841], changed_forces[:841])
    for x in range(29):
        for y in range(29):
            wire = _interleaved_binary(x, y)
            assert int(mapping[29 * x + y]) == wire
            assert _binary_wire_grid_coordinates(wire) == (x, y)
    graph.y = torch.randn_like(graph.y)
    torch.testing.assert_close(compile_plate_binary_contact(graph), mapping)
    del graph.y
    torch.testing.assert_close(compile_plate_binary_contact(graph), mapping)


def test_binary_compiler_is_invariant_to_node_relabeling() -> None:
    graph = _dummy_plate()
    original = compile_plate_binary_contact(graph)
    permutation = torch.randperm(graph.num_nodes,
                                 generator=torch.Generator().manual_seed(51))
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(graph.num_nodes)
    relabeled = graph.clone()
    for key in ("x", "pos", "u0", "properties", "node_type",
                "loss_mask", "y"):
        setattr(relabeled, key, getattr(graph, key)[permutation])
    relabeled.edge_index = inverse[graph.edge_index]
    torch.testing.assert_close(
        compile_plate_binary_contact(relabeled)[inverse], original)


def test_binary_butterfly_levels_are_exact_physical_scales() -> None:
    for level in range(10):
        stride = 1 << level
        count = 0
        for pair_index in range(WIRES // 2):
            left = (pair_index // stride) * (2 * stride) + pair_index % stride
            right = left + stride
            x0, y0 = _binary_wire_grid_coordinates(left)
            x1, y1 = _binary_wire_grid_coordinates(right)
            if max(x0, y0, x1, y1) >= 29:
                continue
            expected = 1 << (level // 2)
            assert (abs(x1 - x0), abs(y1 - y0)) == (
                (expected, 0) if level % 2 == 0 else (0, expected))
            assert 5 * (abs(x1 - x0) + abs(y1 - y0)) == 5 * expected
            count += 1
        assert count > 0


def test_binary_runner_dispatch_check_only_and_gray_weight_compatibility(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "prepared"
    small_splits = {"train": (0, 1), "val": (2,), "test": (3,)}
    monkeypatch.setattr(plate, "SPLIT_IDS", small_splits)
    for split, ids in (("train", (0, 1)), ("val", (2,))):
        (root / split).mkdir(parents=True)
        for sim_id in ids:
            graph = _dummy_plate(float(sim_id))
            graph.x = torch.cat((graph.u0, graph.properties), dim=1)
            torch.save(graph, root / split / f"Sim{sim_id:03d}.pt")
    (root / "metadata.json").write_text(json.dumps({
        "splits": {split: list(ids) for split, ids in small_splits.items()}
    }), encoding="utf-8")
    monkeypatch.setattr(plate.sys, "argv", [
        "plate", "--device", "cpu", "--layout", "plate_binary_contact",
        "--arm", "j", "--check-only", "--dim", "8", "--heads", "2",
        "--blocks", "1", "--threads", "1",
    ])
    args = plate._parse_args()
    assert args.layout == "plate_binary_contact" and args.check_only
    datasets, info = plate._load_data(root, layout=args.layout,
                                      include_test=False)
    assert set(datasets) == {"train", "val"}
    graph = datasets["train"][0]
    torch.testing.assert_close(graph.plate_node_to_wire,
                               compile_plate_binary_contact(graph))
    torch.set_num_threads(1)
    plate._check_only(args, datasets, info)
    assert '"check": "passed"' in capsys.readouterr().out

    gray_args = argparse.Namespace(**vars(args))
    gray_args.layout = "plate_gray_contact"
    gray_config = plate._config(gray_args, info, seed=0, scale=.25)
    binary_config = plate._config(args, info, seed=0, scale=.25)
    assert {**gray_config["model"], "layout": args.layout} == binary_config["model"]
    assert "plate_distance_angle_prior" not in gray_config["model"]
    torch.manual_seed(28)
    gray = plate._make_model(gray_args, info, seed=0, scale=.25)
    torch.manual_seed(28)
    binary = plate._make_model(args, info, seed=0, scale=.25)
    assert gray.state_dict().keys() == binary.state_dict().keys()
    for name in gray.state_dict():
        torch.testing.assert_close(gray.state_dict()[name],
                                   binary.state_dict()[name], atol=0., rtol=0.)


def test_scaffold_component_layout_keeps_mesh_wires_across_forcing() -> None:
    graph = _dummy_plate()
    shifted = _dummy_plate(7.)
    first = compile_plate_gray_contact(graph, mesh_layout="scaffold_components")
    second = compile_plate_gray_contact(
        shifted, mesh_layout="scaffold_components")
    assert first.unique().numel() == graph.num_nodes
    torch.testing.assert_close(first[:841], second[:841])
    # This comparison isolates a stable scaffold from Gray's local pairing.
    assert not torch.equal(first[:841], compile_plate_gray_contact(graph)[:841])

    permutation = torch.randperm(graph.num_nodes,
                                 generator=torch.Generator().manual_seed(18))
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(graph.num_nodes)
    relabeled = graph.clone()
    for key in ("x", "pos", "u0", "properties", "node_type",
                "loss_mask", "y"):
        setattr(relabeled, key, getattr(graph, key)[permutation])
    relabeled.edge_index = inverse[graph.edge_index]
    compiled = compile_plate_gray_contact(
        relabeled, mesh_layout="scaffold_components")
    torch.testing.assert_close(compiled[inverse], first)


def test_gray_nearest_uses_initial_xy_and_dataset_dispatch(tmp_path: Path) -> None:
    graph = _dummy_plate()
    contact = compile_plate_gray_contact(graph)
    nearest = compile_plate_gray_contact(graph, force_layout="nearest")
    torch.testing.assert_close(nearest[:841], contact[:841])
    assert nearest.unique().numel() == graph.num_nodes
    assert [_wire_grid_coordinates(int(wire)) for wire in nearest[841:]] == [
        (11, 29), (29, 7)]

    # Change only the force contact edges.  They are valid input and may set
    # canonical tie order, but cannot change the nearest-placement cost.
    changed = graph.clone()
    force_edges = changed.edge_index[0] >= 841
    changed.edge_index[1, force_edges] = (
        changed.edge_index[1, force_edges] + 400) % 841
    changed.y = torch.randn_like(graph.y)
    torch.testing.assert_close(
        compile_plate_gray_contact(changed, force_layout="nearest"), nearest)

    prepared = tmp_path / "train"
    prepared.mkdir()
    torch.save(graph, prepared / "Sim000.pt")
    dataset = plate.PlateDataset(tmp_path, "train", (0,),
                                 layout="plate_gray_nearest")
    torch.testing.assert_close(dataset[0].plate_node_to_wire, nearest)


def test_targets_are_not_read_and_dataset_caches_mapping(tmp_path: Path) -> None:
    graph = _dummy_plate()
    mapping = compile_plate_gray_contact(graph)
    graph.y = torch.randn_like(graph.y)
    torch.testing.assert_close(compile_plate_gray_contact(graph), mapping)
    del graph.y
    torch.testing.assert_close(compile_plate_gray_contact(graph), mapping)

    prepared = tmp_path / "train"
    prepared.mkdir()
    torch.save(graph, prepared / "Sim000.pt")
    dataset = plate.PlateDataset(tmp_path, "train", (0,),
                                 layout="plate_gray_contact")
    first = dataset[0]
    torch.testing.assert_close(first.plate_node_to_wire, mapping)
    assert first.x.shape == (843, 6)


def test_zero_exchange_is_layout_independent_and_batched_raw_uses_mapping() -> None:
    graph = _dummy_plate()
    other = _dummy_plate(7.)
    graphs = (graph, other)
    for item in graphs:
        item.plate_node_to_wire = compile_plate_gray_contact(item)
    batch = Batch.from_data_list(list(graphs))
    torch.manual_seed(3)
    model = MultiGraphGCM(input_dim=6, edge_dim=9, output_dim=3,
                          dim=8, heads=2, blocks=1, max_stages=10,
                          mode="original", exchange_scale=0., dropout=0.,
                          graph_embed=True, graph_refresh=True,
                          relay_padding=True,
                          edge_weight_mode="directed_symmetric").eval()
    with torch.no_grad():
        mapped = model.predict_nodes(batch, mappings=[
            item.plate_node_to_wire for item in graphs])
        arbitrary = model.predict_nodes(batch, mappings=[
            torch.arange(item.num_nodes) for item in graphs])
        raw = plate._predict_raw(model, batch)
    torch.testing.assert_close(mapped, arbitrary, atol=1e-6, rtol=1e-6)
    expected = batch.u0 + torch.cat((mapped, mapped), dim=1) * (
        batch.node_type == 0).unsqueeze(-1)
    torch.testing.assert_close(raw, expected, atol=1e-6, rtol=1e-6)


def test_gray_distance_prior_matches_butterfly_pairs_and_preserves_zero() -> None:
    scales = compile_plate_gray_distance_angle_scales(heads=3,
                                                       carrier_heads=1)
    assert scales.shape == (10, WIRES // 2, 3)
    torch.testing.assert_close(scales[:, :, 0], torch.ones((10, WIRES // 2)))
    # Stage two pairs (0,4) and (1,5): Gray x-coordinates are (0,3)
    # and (1,2), respectively. Both use the same stage median.
    stage = 2
    stride = 1 << stage
    pair_indices = torch.arange(WIRES // 2)
    left = (pair_indices // stride) * (2 * stride) + pair_indices % stride
    right = left + stride
    coordinates = torch.tensor([_wire_grid_coordinates(wire)
                                for wire in range(WIRES)])
    mesh_pairs = ((coordinates[left] < 29).all(1) &
                  (coordinates[right] < 29).all(1))
    distances = torch.linalg.vector_norm(
        (coordinates[left[mesh_pairs]] -
         coordinates[right[mesh_pairs]]).float() * 5., dim=1)
    median = torch.quantile(distances, .5).item()
    for pair_index, distance in ((0, 15.), (1, 5.)):
        expected = 2 * median / (median + distance)
        torch.testing.assert_close(
            scales[stage, pair_index, 1:],
            torch.full((2,), expected), atol=1e-6, rtol=1e-6)
    assert scales[stage, 1, 1] > scales[stage, 0, 1]
    torch.testing.assert_close(
        scales[stage, ~mesh_pairs],
        torch.ones_like(scales[stage, ~mesh_pairs]))

    graph = _dummy_plate()
    graph.plate_node_to_wire = compile_plate_gray_contact(graph)
    batch = Batch.from_data_list([graph])
    args = argparse.Namespace(
        layout="plate_gray_contact", independent_six_state=False,
        dim=8, heads=2, blocks=1, dropout=0., angle=.2,
        tie_pair_gate=False, relay_padding=True, balanced_carrier_heads=1,
        plate_distance_angle_prior=False, freeze_boundary=True,
        lr=1e-3, weight_decay=0., batch_size=1, epochs=1,
        patience=1, clip_norm=1.,
    )
    info = {
        "max_stages": 10, "benchmark": "fixture",
        "metadata_sha256": "fixture", "prepared_root": "fixture",
        "split_simulation_ids": {}, "input_processing": {},
        "normalization_gap": "fixture",
    }
    default_config = plate._config(args, info, seed=0, scale=0.)
    assert "plate_distance_angle_prior" not in default_config["model"]
    assert "prior_only_angle_scale" not in default_config["model"]
    del args.plate_distance_angle_prior
    assert plate._config(args, info, seed=0, scale=0.) == default_config
    torch.manual_seed(19)
    plain = plate._make_model(args, info, seed=0, scale=0.).eval()
    assert not plain.prior_only_angle_scale
    assert all(not block.circuit.prior_only_angle_scale
               for block in plain.blocks)
    args.plate_distance_angle_prior = True
    assert plate._config(args, info, seed=0, scale=0.)["model"][
        "plate_distance_angle_prior"] is True
    assert plate._config(args, info, seed=0, scale=0.)["model"][
        "prior_only_angle_scale"] is True
    torch.manual_seed(19)
    prior = plate._make_model(args, info, seed=0, scale=0.).eval()
    assert prior.prior_only_angle_scale
    assert all(block.circuit.prior_only_angle_scale
               for block in prior.blocks)
    assert plain.state_dict().keys() == prior.state_dict().keys()
    assert all(torch.equal(plain.state_dict()[key], prior.state_dict()[key])
               for key in plain.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(plate._predict_raw(prior, batch),
                                   plate._predict_raw(plain, batch),
                                   atol=0., rtol=0.)
