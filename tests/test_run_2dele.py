"""One focused check of 2Dele representation and official score semantics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_2dele.py"
SPEC = importlib.util.spec_from_file_location("run_2dele", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_lossless_primal_dual_input_and_official_primal_score() -> None:
    p_x = torch.arange(105 * 5, dtype=torch.float32).reshape(105, 5) / 100
    d_x = torch.arange(173 * 5, dtype=torch.float32).reshape(173, 5) / 70
    p_y = torch.arange(105 * 4, dtype=torch.float32).reshape(105, 4) / 40
    triangles = torch.arange(173 * 3).reshape(173, 3) % 105
    primal_edges = torch.arange(277 * 2).reshape(277, 2) % 105
    dual_edges = torch.arange(277 * 2).reshape(277, 2) % 173
    adjs = [None] * 9
    adjs[6], adjs[7], adjs[8] = primal_edges, triangles, dual_edges
    boundary = torch.zeros(105, 1, dtype=torch.bool)
    boundary[:35] = True
    graph = SimpleNamespace(p_x=p_x, d_x=d_x, p_y=p_y, adjs=adjs,
                            bdry_markers=[boundary, None,
                                          torch.zeros(173, 1, dtype=torch.bool)],
                            poses=[torch.rand(105, 2), torch.rand(173, 2)],
                            vols=[None, torch.ones(173, 1), None,
                                  torch.ones(105)])

    data = runner._prepare_graph(graph, "primal_dual")
    assert data.x.shape == (278, 11)
    assert torch.equal(data.x[:105, :5], p_x)
    assert torch.equal(data.x[105:, 5:10], d_x)
    assert int(data.score_mask.sum()) == 105
    assert int(data.boundary_mask.sum()) == 35

    prediction = torch.zeros_like(data.y)
    prediction[105:] = 1e6  # face outputs must never enter the score
    metrics = runner._Metrics()
    metrics.update(prediction, data)
    observed = metrics.result()
    official = p_y.double().square().sum() / 105
    assert abs(observed["score"] - float(official)) < 1e-5
    assert observed["scored_nodes"] == 105
    assert observed["regions"]["interior"]["nodes"] == 70
    assert observed["regions"]["boundary"]["nodes"] == 35

    # Changing the targets cannot alter either model inputs or compilation.
    graph.p_y = graph.p_y + 99
    changed = runner._prepare_graph(graph, "primal_dual")
    assert torch.equal(data.x, changed.x)
    assert torch.equal(data.edge_index, changed.edge_index)

    geometry = runner._prepare_graph(graph, "primal_dual", geometry_input=True)
    graph.p_y = graph.p_y - 7
    geometry_changed = runner._prepare_graph(graph, "primal_dual",
                                             geometry_input=True)
    assert geometry.x.shape == (278, 15)
    assert geometry.edge_attr.shape == (2146, 6)
    assert torch.equal(geometry.x, geometry_changed.x)
    assert torch.equal(geometry.edge_attr, geometry_changed.edge_attr)
