"""Focused CPU checks for the Plate Deformation paired runner."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data


SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_plate_deformation.py"
SPEC = importlib.util.spec_from_file_location("run_plate_deformation", SOURCE)
assert SPEC is not None and SPEC.loader is not None
plate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plate)


def _graph(shift: float) -> Data:
    pos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
                        [3., 0., 0.], [1.5, 0., 1.]], dtype=torch.float32)
    pos[:, 1] += shift
    node_type = torch.tensor([0, 0, 2, 0, 1])
    loss_mask = node_type != 1
    u0 = torch.cat((pos, torch.zeros_like(pos)), dim=1)
    y = u0.clone()
    y[:4, 3:] = torch.tensor([[1., 0., 0.], [2., 0., 0.],
                              [3., 0., 0.], [4., 0., 0.]])
    props = torch.zeros((5, 6))
    props[torch.arange(5), node_type] = 1
    props[4, 5] = 400
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 4, 4],
                               [1, 0, 2, 1, 3, 2, 0, 1, 2]])
    edge_attr = torch.zeros((edge_index.shape[1], 9))
    return Data(x=torch.cat((u0, props), dim=1), u0=u0, pos=pos,
                properties=props, edge_index=edge_index, edge_attr=edge_attr,
                y=y, node_type=node_type, loss_mask=loss_mask)


def _args(output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output=output, resume=False, arm="paired", dim=8, heads=2, blocks=1,
        dropout=0., angle=.2, tie_pair_gate=False, relay_padding=True,
        balanced_carrier_heads=1, independent_six_state=False,
        freeze_boundary=True, output_scale="none",
        layout="graph_components", lr=1e-3,
        weight_decay=0., batch_size=2, epochs=1, patience=1,
        clip_norm=1., print_every=1,
    )


def test_carrier_preserving_shuffle_runner_config_and_cli(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    info = {
        "benchmark": "fixture", "metadata_sha256": "fixture",
        "prepared_root": str(tmp_path), "split_simulation_ids": {},
        "input_processing": {}, "normalization_gap": "fixture", "max_stages": 3,
    }
    args = _args(tmp_path)
    default_config = plate._config(args, info, 0, .25)
    assert "shuffle_exempt_heads" not in default_config["model"]
    args.interstage_head_shuffle = True
    args.shuffle_exempt_heads = 1
    preserved_config = plate._config(args, info, 0, .25)
    assert preserved_config["model"]["shuffle_exempt_heads"] == 1
    assert preserved_config["model"]["interstage_head_shuffle"] == (
        "complete_real_imag_pair")
    model = plate._make_model(args, info, 0, .25)
    assert all(block.circuit.shuffle_exempt_heads == 1
               for block in model.blocks)

    monkeypatch.setattr(plate.sys, "argv", [
        "plate", "--device", "cpu", "--balanced-carrier-heads", "1",
        "--interstage-head-shuffle", "--shuffle-exempt-heads", "1"])
    assert plate._parse_args().shuffle_exempt_heads == 1
    for options in (
            ["--shuffle-exempt-heads", "1"],
            ["--interstage-head-shuffle", "--shuffle-exempt-heads", "1"],
            ["--balanced-carrier-heads", "1", "--interstage-head-shuffle",
             "--shuffle-exempt-heads", "4"]):
        monkeypatch.setattr(plate.sys, "argv", ["plate", "--device", "cpu",
                                               *options])
        with pytest.raises(SystemExit):
            plate._parse_args()


def test_gate_anchor_runner_config_and_zero_exchange(tmp_path: Path) -> None:
    info = {
        "benchmark": "fixture", "metadata_sha256": "fixture",
        "prepared_root": str(tmp_path), "split_simulation_ids": {},
        "input_processing": {}, "normalization_gap": "fixture", "max_stages": 3,
    }
    args = _args(tmp_path)
    default_config = plate._config(args, info, 0, 0.)
    assert "gate_anchor" not in default_config["model"]
    args.gate_anchor = 0.
    assert plate._config(args, info, 0, 0.) == default_config
    torch.manual_seed(31)
    default = plate._make_model(args, info, 0, 0.).eval()
    args.gate_anchor = .7
    anchored_config = plate._config(args, info, 0, 0.)
    assert anchored_config["model"]["gate_anchor"] == .7
    torch.manual_seed(31)
    anchored = plate._make_model(args, info, 0, 0.).eval()
    assert set(default.state_dict()) == set(anchored.state_dict())
    assert all(torch.equal(default.state_dict()[key], anchored.state_dict()[key])
               for key in default.state_dict())
    from torch_geometric.data import Batch
    graph = _graph(0.)
    graph.x = graph.properties.clone()
    batch = Batch.from_data_list([graph])
    mapping = [torch.arange(graph.num_nodes)]
    with torch.no_grad():
        torch.testing.assert_close(
            default.predict_nodes(batch, mappings=mapping),
            anchored.predict_nodes(batch, mappings=mapping),
            atol=0., rtol=0.)


def test_reverse_odd_blocks_config_and_zero_exchange(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plate.sys, "argv", ["plate", "--device", "cpu",
                                           "--reverse-odd-blocks"])
    assert plate._parse_args().reverse_odd_blocks is True
    info = {
        "benchmark": "fixture", "metadata_sha256": "fixture",
        "prepared_root": str(tmp_path), "split_simulation_ids": {},
        "input_processing": {}, "normalization_gap": "fixture", "max_stages": 3,
    }
    args = _args(tmp_path)
    args.blocks = 2
    default_config = plate._config(args, info, 0, 0.)
    assert "reverse_odd_blocks" not in default_config["model"]
    torch.manual_seed(37)
    default = plate._make_model(args, info, 0, 0.).eval()

    args.reverse_odd_blocks = False
    assert plate._config(args, info, 0, 0.) == default_config
    torch.manual_seed(37)
    explicit_false = plate._make_model(args, info, 0, 0.).eval()
    assert all(torch.equal(default.state_dict()[key], explicit_false.state_dict()[key])
               for key in default.state_dict())
    assert not explicit_false.blocks[1].circuit.reverse_stages

    args.reverse_odd_blocks = True
    assert plate._config(args, info, 0, 0.)["model"]["reverse_odd_blocks"] is True
    torch.manual_seed(37)
    reversed_model = plate._make_model(args, info, 0, 0.).eval()
    assert not reversed_model.blocks[0].circuit.reverse_stages
    assert reversed_model.blocks[1].circuit.reverse_stages
    assert all(torch.equal(default.state_dict()[key], reversed_model.state_dict()[key])
               for key in default.state_dict())

    from torch_geometric.data import Batch
    graph = _graph(0.)
    graph.x = graph.properties.clone()
    batch = Batch.from_data_list([graph])
    mapping = [torch.arange(graph.num_nodes)]
    with torch.no_grad():
        expected = default.predict_nodes(batch, mappings=mapping)
        torch.testing.assert_close(
            explicit_false.predict_nodes(batch, mappings=mapping),
            expected, atol=0., rtol=0.)
        torch.testing.assert_close(
            reversed_model.predict_nodes(batch, mappings=mapping),
            expected, atol=0., rtol=0.)


def test_masked_six_state_metric_and_paired_cpu_epoch(tmp_path: Path) -> None:
    graphs = [_graph(0.), _graph(1.)]
    for index, graph in enumerate(graphs):
        plate._validate_graph(graph, tmp_path / f"Sim{index:03d}.pt")
    # The initial state has only displacement error: squared magnitudes
    # 1+4+9+16 over four scored nodes. Force node targets never contribute.
    from torch_geometric.data import Batch
    batch = Batch.from_data_list(graphs)
    accumulator = plate._MetricAccumulator(distance_bins=True)
    accumulator.update(batch.u0, batch)
    metrics = accumulator.result()
    assert math.isclose(metrics["state_MSE"], 7.5)
    assert math.isclose(metrics["displacement_MSE"], 2.5)
    assert metrics["scored_nodes"] == 8
    assert sum(b["nodes"] for b in metrics["force_distance_bins"].values()) == 8

    # Official logging averages masked loss per batch, including a short
    # final batch; this differs from a global node-weighted average.
    short = _graph(0.)
    short.y[0, 3] = 10.
    batch_mean = plate._MetricAccumulator(distance_bins=False)
    batch_mean.update(batch.u0, batch)
    from torch_geometric.data import Batch as PyGBatch
    short_batch = PyGBatch.from_data_list([short])
    batch_mean.update(short_batch.u0, short_batch)
    assert math.isclose(batch_mean.result()["state_MSE"], (7.5 + 32.25) / 2)
    assert math.isclose(batch_mean.result()["state_MSE_node_weighted"],
                        (60 + 129) / 12)

    root = tmp_path / "prepared"
    for split, ids in (("train", (0, 1)), ("val", (2,)), ("test", (3,))):
        (root / split).mkdir(parents=True)
        for sim_id in ids:
            torch.save(graphs[sim_id % 2], root / split / f"Sim{sim_id:03d}.pt")
    dataset = {
        split: plate.PlateDataset(root, split, ids)
        for split, ids in (("train", (0, 1)), ("val", (2,)), ("test", (3,)))
    }
    coordinate_dataset = plate.PlateDataset(root, "train", (0,),
                                            coordinate_input=True)
    coordinate_graph = coordinate_dataset[0]
    assert coordinate_graph.x.shape == (5, 8)
    torch.testing.assert_close(coordinate_graph.x[:, :6], graphs[0].properties)
    torch.testing.assert_close(coordinate_graph.x[:, 6:],
                               graphs[0].pos[:, :2] / 140.)
    info = {
        "benchmark": "synthetic CPU path only", "metadata_sha256": "fixture",
        "prepared_root": str(root), "split_simulation_ids": {},
        "input_processing": {}, "normalization_gap": "fixture", "max_stages": 3,
    }
    args = _args(tmp_path / "results")
    initialized = plate._make_model(args, info, 0, .25)
    assert initialized.output_dim == 3
    batch.x = batch.properties.clone()
    with torch.no_grad():
        physical = plate._predict_raw(initialized.eval(), batch)
    assert torch.allclose(physical[:, :3] - batch.u0[:, :3],
                          physical[:, 3:] - batch.u0[:, 3:], atol=1e-6)
    assert torch.equal(physical[batch.node_type != 0],
                       batch.u0[batch.node_type != 0])
    assert all(math.isclose(float(switch.base[0]), math.pi / 4, rel_tol=1e-6)
               and math.isclose(float(switch.base[1]), .2, rel_tol=1e-6)
               for block in initialized.blocks for switch in block.circuit.switches)
    torch.set_num_threads(1)
    zero = plate._run_arm(args, dataset, info, 0, 0., torch.device("cpu"))
    exchange = plate._run_arm(args, dataset, info, 0, .25, torch.device("cpu"))
    assert zero["initial_parameter_digest"] == exchange["initial_parameter_digest"]
    assert math.isfinite(zero["test"]["state_MSE"])
    assert math.isfinite(exchange["test"]["state_MSE"])
    assert zero["selected_epoch"] == exchange["selected_epoch"] == 1
    args.resume = True
    assert (plate._run_arm(args, dataset, info, 0, .25,
                           torch.device("cpu"))["test"]
            == exchange["test"])


def test_directed_force_message_survives_plate_local_normalization() -> None:
    from gcm.multigraph import _aggregate, _edge_weights

    graph = _graph(0.)
    source, target = graph.edge_index
    force_edges = source == 4
    assert bool(force_edges.any())
    symmetric = _edge_weights(graph.edge_index, graph.num_nodes, "symmetric")
    incoming = _edge_weights(graph.edge_index, graph.num_nodes,
                             "directed_symmetric")
    assert torch.equal(symmetric[force_edges], torch.zeros_like(symmetric[force_edges]))
    assert bool((incoming[force_edges] > 0).all())

    values = torch.zeros((graph.num_nodes, 1))
    values[4, 0] = 1.
    old_message = _aggregate(values, graph.edge_index, symmetric, None,
                             graph.num_nodes)
    new_message = _aggregate(values, graph.edge_index, incoming, None,
                             graph.num_nodes)
    assert torch.equal(old_message, torch.zeros_like(old_message))
    assert bool((new_message[target[force_edges]] > 0).all())

    reciprocal = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    torch.testing.assert_close(
        _edge_weights(reciprocal, 3, "directed_symmetric"),
        _edge_weights(reciprocal, 3, "symmetric"),
    )

    # Caching fixed input-graph weights must preserve model predictions.
    graph.x = graph.properties.clone()
    model = plate._make_model(_args(Path("unused")), {"max_stages": 3},
                              seed=0, scale=.25).eval()
    from torch_geometric.data import Batch
    plain = Batch.from_data_list([graph])
    graph.directed_edge_weight = incoming
    cached = Batch.from_data_list([graph])
    with torch.no_grad():
        direct_prediction = model.predict_nodes(plain, mappings=[torch.arange(5)])
        cached_prediction = model.predict_nodes(cached, mappings=[torch.arange(5)])
    torch.testing.assert_close(cached_prediction, direct_prediction, atol=0., rtol=0.)


def test_train_only_output_std_scales_three_channel_head_and_paired_run(
        tmp_path: Path) -> None:
    train_graphs = [_graph(0.), _graph(1.)]
    for graph in train_graphs:
        graph.y[:4, 3:6] = torch.tensor([
            [1., 2., 3.], [2., 4., 6.],
            [3., 6., 9.], [4., 8., 12.],
        ])
        graph.y[:4, :3] = graph.u0[:4, :3] + graph.y[:4, 3:6]
        # Force-node targets are excluded by the same official score mask.
        graph.y[4, 3:6] = 1_000_000.
    other = _graph(2.)
    other.y[:4, 3:6] = 10_000.
    root = tmp_path / "prepared"
    for split, entries in (("train", ((0, train_graphs[0]),
                                      (1, train_graphs[1]))),
                           ("val", ((2, other),)),
                           ("test", ((3, other),))):
        (root / split).mkdir(parents=True)
        for sim_id, graph in entries:
            torch.save(graph, root / split / f"Sim{sim_id:03d}.pt")
    datasets = {
        split: plate.PlateDataset(root, split, ids)
        for split, ids in (("train", (0, 1)), ("val", (2,)), ("test", (3,)))
    }
    args = _args(tmp_path / "scaled-results")
    args.output_scale = "train_std"
    stats = plate._output_scale_stats(args, datasets)
    assert stats is not None
    assert stats["source_split"] == "train"
    assert stats["graph_count"] == 2
    assert stats["scored_node_count"] == 8
    torch.testing.assert_close(torch.tensor(stats["mean"]),
                               torch.tensor([2.5, 5., 7.5]))
    torch.testing.assert_close(torch.tensor(stats["std"]),
                               torch.tensor([1., 2., 3.]) * math.sqrt(1.25))

    info = {
        "benchmark": "synthetic CPU path only", "metadata_sha256": "fixture",
        "prepared_root": str(root), "split_simulation_ids": {},
        "input_processing": {}, "normalization_gap": "fixture", "max_stages": 3,
    }
    default_args = _args(tmp_path / "unused")
    assert "output_scale" not in plate._config(default_args, info, 0, .25)["model"]
    scaled_config = plate._config(args, info, 0, .25, stats)
    assert scaled_config["model"]["output_scale"] == {
        "mode": "train_std", "training_stats": stats,
    }
    assert scaled_config["training"]["loss"] == (
        "raw six-channel sum-of-squares per NORMAL/BOUNDARY node")

    from torch_geometric.data import Batch
    batch = Batch.from_data_list(train_graphs)
    batch.x = batch.properties.clone()
    model = plate._make_model(args, info, 0, .25).eval()
    with torch.no_grad():
        unscaled = plate._predict_raw(model, batch, freeze_boundary=False)
        scaled = plate._predict_raw(
            model, batch, freeze_boundary=False,
            output_std=torch.tensor(stats["std"], dtype=torch.float32))
    factor = torch.tensor(stats["std"] * 2, dtype=torch.float32)
    torch.testing.assert_close(scaled - batch.u0,
                               (unscaled - batch.u0) * factor,
                               atol=2e-5, rtol=2e-5)

    torch.set_num_threads(1)
    development = {split: datasets[split] for split in ("train", "val")}
    args.arm = "j"
    validated_j = plate._run_seed(args, development, info, 0,
                                   torch.device("cpu"), stats)
    assert validated_j["status"] == "validated_no_test"
    assert "test" not in validated_j
    j_folder = args.output / "seed_0" / "scale_0p25"
    assert (j_folder / "validation_only.json").exists()
    assert (j_folder / "best.pt").exists()
    assert (j_folder / "latest.pt").exists()
    assert not (j_folder / "result.json").exists()
    assert not (args.output / "seed_0" / "paired.json").exists()
    # An interrupted validation-only run resumes from latest.pt without test.
    (j_folder / "validation_only.json").unlink()
    args.resume = True
    restored = plate._run_seed(args, development, info, 0,
                                torch.device("cpu"), stats)
    assert restored["status"] == "validated_no_test"
    assert "test" not in restored
    assert not (j_folder / "result.json").exists()
    assert (plate._run_seed(args, development, info, 0,
                            torch.device("cpu"), stats)["val"]
            == restored["val"])
    args.arm = "paired"
    with pytest.raises(RuntimeError, match="Both arms must finish validation"):
        plate._run_seed(args, datasets, info, 0, torch.device("cpu"), stats)
    assert not (j_folder / "result.json").exists()

    args.arm = "zero"
    args.resume = False
    validated_zero = plate._run_seed(args, development, info, 0,
                                      torch.device("cpu"), stats)
    assert "test" not in validated_zero
    args.arm = "paired"
    args.resume = True
    paired = plate._run_seed(args, datasets, info, 0,
                             torch.device("cpu"), stats)
    zero = json.loads((args.output / "seed_0" / "scale_0" /
                       "result.json").read_text(encoding="utf-8"))
    exchange = json.loads((j_folder / "result.json").read_text(encoding="utf-8"))
    assert zero["initial_parameter_digest"] == exchange["initial_parameter_digest"]
    assert zero["train_output_scale_stats"] == exchange["train_output_scale_stats"] == stats
    assert zero["selected_epoch"] == exchange["selected_epoch"] == 1
    assert math.isfinite(exchange["test"]["state_MSE"])
    assert paired["zero"]["test"] == zero["test"]
    assert paired["exchange_0p25"]["test"] == exchange["test"]
    assert plate._run_arm(args, datasets, info, 0, .25,
                          torch.device("cpu"), stats)["test"] == exchange["test"]
    args.output_scale = "none"
    with pytest.raises(ValueError, match="incompatible"):
        plate._run_arm(args, datasets, info, 0, .25, torch.device("cpu"))


def test_single_arm_data_load_does_not_open_test_graphs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    small_splits = {"train": (0, 1), "val": (2,), "test": (3,)}
    monkeypatch.setattr(plate, "SPLIT_IDS", small_splits)
    root = tmp_path / "prepared"
    for split, ids in (("train", (0, 1)), ("val", (2,))):
        (root / split).mkdir(parents=True)
        for sim_id in ids:
            torch.save(_graph(float(sim_id)),
                       root / split / f"Sim{sim_id:03d}.pt")
    (root / "metadata.json").write_text(json.dumps({
        "splits": {split: list(ids) for split, ids in small_splits.items()}
    }), encoding="utf-8")
    datasets, info = plate._load_data(root, include_test=False)
    assert set(datasets) == {"train", "val"}
    assert info["split_simulation_ids"]["test"] == [3, 3]
    with pytest.raises(FileNotFoundError, match="test"):
        plate._load_data(root, include_test=True)


def test_force_feature_scale_changes_only_model_force_input(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    small_splits = {"train": (0,), "val": (1,), "test": (2,)}
    monkeypatch.setattr(plate, "SPLIT_IDS", small_splits)
    root = tmp_path / "prepared"
    for split, ids in (("train", (0,)), ("val", (1,))):
        (root / split).mkdir(parents=True)
        torch.save(_graph(float(ids[0])), root / split / f"Sim{ids[0]:03d}.pt")
    (root / "metadata.json").write_text(json.dumps({
        "splits": {split: list(ids) for split, ids in small_splits.items()}
    }), encoding="utf-8")

    original, original_info = plate._load_data(root, include_test=False)
    scaled, scaled_info = plate._load_data(
        root, include_test=False, force_feature_scale=400.)
    raw = original["train"][0]
    modified = scaled["train"][0]
    torch.testing.assert_close(raw.x, raw.properties, atol=0., rtol=0.)
    torch.testing.assert_close(modified.x[:, :3], raw.x[:, :3], atol=0., rtol=0.)
    torch.testing.assert_close(modified.x[:, 3:6], raw.x[:, 3:6] / 400.,
                               atol=0., rtol=0.)
    for key in ("properties", "u0", "pos", "edge_index", "edge_attr", "y"):
        torch.testing.assert_close(getattr(modified, key), getattr(raw, key),
                                   atol=0., rtol=0.)
    assert "force_feature_scale" not in original_info["input_processing"]
    assert scaled_info["input_processing"]["force_feature_scale"] == 400.


def test_force_feature_scale_config_default_and_validation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    info = {
        "benchmark": "fixture", "metadata_sha256": "fixture",
        "prepared_root": str(tmp_path), "split_simulation_ids": {},
        "input_processing": {}, "normalization_gap": "fixture", "max_stages": 3,
    }
    args = _args(tmp_path)
    original = plate._config(args, info, 0, .25)
    assert "force_feature_scale" not in original["model"]
    args.force_feature_scale = 1.
    assert plate._config(args, info, 0, .25) == original
    args.force_feature_scale = 400.
    assert plate._config(args, info, 0, .25)["model"]["force_feature_scale"] == 400.
    monkeypatch.setattr(plate.sys, "argv", ["plate", "--device", "cpu",
                                           "--force-feature-scale", "400"])
    assert plate._parse_args().force_feature_scale == 400.
    for invalid in ("0", "-1", "nan", "inf"):
        monkeypatch.setattr(plate.sys, "argv", ["plate", "--device", "cpu",
                                               "--force-feature-scale", invalid])
        with pytest.raises(SystemExit):
            plate._parse_args()


def test_full_original_j_scale_keeps_default_pair_and_defers_test(
        tmp_path: Path) -> None:
    root = tmp_path / "prepared"
    for split, ids in (("train", (0, 1)), ("val", (2,)), ("test", (3,))):
        (root / split).mkdir(parents=True)
        for sim_id in ids:
            torch.save(_graph(float(sim_id)), root / split / f"Sim{sim_id:03d}.pt")
    datasets = {split: plate.PlateDataset(root, split, ids)
                for split, ids in (("train", (0, 1)),
                                   ("val", (2,)), ("test", (3,)))}
    info = {
        "benchmark": "synthetic CPU path only", "metadata_sha256": "fixture",
        "prepared_root": str(root), "split_simulation_ids": {},
        "input_processing": {}, "normalization_gap": "fixture",
        "max_stages": 3,
    }
    args = _args(tmp_path / "results")
    assert plate._paired_scales(args) == (0., .25)
    assert plate._config(args, info, 0, 0.)["model"]["exchange_scale"] == 0.
    args.j_exchange_scale = 1.
    assert plate._paired_scales(args) == (0., 1.)
    with pytest.raises(ValueError, match="j_exchange_scale"):
        plate._paired_scales(argparse.Namespace(j_exchange_scale=0.))

    development = {split: datasets[split] for split in ("train", "val")}
    torch.set_num_threads(1)
    args.arm = "j"
    j = plate._run_seed(args, development, info, 0, torch.device("cpu"))
    assert j["status"] == "validated_no_test"
    assert j["config"]["model"]["exchange_scale"] == 1.
    assert (args.output / "seed_0" / "scale_1" / "validation_only.json").exists()
    assert not (args.output / "seed_0" / "scale_1" / "result.json").exists()
    args.arm = "paired"
    args.resume = True
    with pytest.raises(RuntimeError, match="Both arms must finish validation"):
        plate._run_seed(args, datasets, info, 0, torch.device("cpu"))

    args.arm = "zero"
    args.resume = False
    zero = plate._run_seed(args, development, info, 0, torch.device("cpu"))
    args.arm = "paired"
    args.resume = True
    paired = plate._run_seed(args, datasets, info, 0, torch.device("cpu"))
    assert paired["exchange_1"]["test"]["state_MSE"] >= 0.
    assert paired["initial_parameter_digest"] == j["initial_parameter_digest"]
    assert paired["initial_parameter_digest"] == zero["initial_parameter_digest"]
    assert (args.output / "seed_0" / "paired_scale_1.json").exists()
    assert not (args.output / "seed_0" / "paired.json").exists()
