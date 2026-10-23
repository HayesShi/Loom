"""Protocol checks for the fixed-mesh Magnetostatics paired runner."""

from __future__ import annotations

import importlib.util
import math
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_hodge_magnetostatics.py"
SPEC = importlib.util.spec_from_file_location("run_hodge_magnetostatics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_saved_public_split_is_exact_and_disjoint() -> None:
    if not RUN.DEFAULT_SPLIT.is_file():
        pytest.skip("Generated official data are unavailable on this host")
    indices = RUN._official_splits(3000, RUN.DEFAULT_SPLIT)
    assert [len(indices[key]) for key in ("train", "val", "test")] == [2040, 360, 600]
    assert len(np.unique(np.concatenate(tuple(indices.values())))) == 3000


def test_three_component_paper_formula_is_three_times_public_code_formula() -> None:
    score = RUN._Metrics()
    prediction = torch.tensor([[1., 2., 3.], [0., -1., 4.]])
    target = torch.zeros_like(prediction)
    score.update(prediction, target, graphs=1)
    result = score.result()
    assert result["normalized_scalar_MSE"] == pytest.approx(31 / 6)
    assert result["normalized_nodewise_vector_norm_SSE"] == pytest.approx(31 / 2)


def test_fixed_compiler_ignores_edge_order_and_is_injective() -> None:
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                       [0., 0., 1.], [2., 2., 2.]], dtype=np.float32)
    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
                      [1, 4], [2, 4], [3, 4]], dtype=np.int64)
    first = RUN.compile_fixed_mesh(points, edges)
    second = RUN.compile_fixed_mesh(points, edges[::-1])
    assert torch.equal(first, second)
    assert len(first.unique()) == len(points)
    assert int(first.max()) < 8


def test_normalization_uses_trainval_and_never_test_target(tmp_path: Path) -> None:
    samples, nodes = 10, 4
    source = np.ones((samples, nodes), dtype=np.float32)
    target = np.ones((samples, nodes, 3), dtype=np.float32) * 2
    split = RUN._official_splits(samples)
    source[split["test"]] = 1000
    target[split["test"]] = 2000
    points = np.eye(4, 3, dtype=np.float32)
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    path = tmp_path / "official_shape_fixture.pkl"
    with path.open("wb") as handle:
        pickle.dump({"X_data": source, "Y_data": target,
                     "nodes": points, "elements": elements}, handle)
    datasets, info, mapping = RUN.load_official(path, strict_size=False)
    assert info["x_scale_trainval_maxabs"] == pytest.approx(1.)
    assert info["y_scale_trainval_maxabs"] == pytest.approx(2.)
    assert datasets["train"][0].x.shape == (4, 4)
    assert datasets["train"][0].y.shape == (4, 3)
    assert len(mapping.unique()) == nodes


def test_existing_j_carrier_is_matched_across_arms_and_can_be_disabled() -> None:
    args = SimpleNamespace(dim=64, heads=4, blocks=3, dropout=0., angle=.2,
                           balanced_carrier_heads=1)
    info = {"max_stages": 12}
    RUN._seed(7)
    zero = RUN._model(args, info, seed=7, scale=0.)
    RUN._seed(7)
    exchange = RUN._model(args, info, seed=7, scale=.25)
    assert RUN._parameter_digest(zero) == RUN._parameter_digest(exchange)
    assert sum(p.numel() for p in zero.parameters()) == sum(
        p.numel() for p in exchange.parameters())
    for model in (zero, exchange):
        for block in model.blocks:
            assert len(block.circuit.switches) == 12
            for switch in block.circuit.switches:
                assert float(switch.base[0]) == pytest.approx(math.pi / 4)
                assert torch.allclose(switch.base[1:],
                                      torch.full((3,), .2))
    args.balanced_carrier_heads = 0
    RUN._seed(7)
    no_carrier = RUN._model(args, info, seed=7, scale=.25)
    assert RUN._parameter_digest(no_carrier) != RUN._parameter_digest(exchange)
    assert sum(p.numel() for p in no_carrier.parameters()) == sum(
        p.numel() for p in exchange.parameters())
    assert all(torch.allclose(switch.base, torch.full((4,), .2))
               for block in no_carrier.blocks
               for switch in block.circuit.switches)
