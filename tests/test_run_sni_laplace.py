"""Checks for released SNI meshes whose nodes are all Dirichlet boundary."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_sni_laplace import _graph_relative_errors  # noqa: E402


def test_all_boundary_graph_still_contributes_to_all_node_metric() -> None:
    boundary_only = Data(
        x=torch.zeros((2, 1)),
        y=torch.tensor([1., 2.]),
        boundary_mask=torch.tensor([True, True]),
    )
    with_interior = Data(
        x=torch.zeros((3, 1)),
        y=torch.tensor([1., 2., 3.]),
        boundary_mask=torch.tensor([True, False, False]),
    )
    batch = Batch.from_data_list([boundary_only, with_interior])
    prediction = torch.tensor([1., 2., 1., 2.1, 2.7])
    all_rel, interior_rel = _graph_relative_errors(prediction, batch)
    torch.testing.assert_close(all_rel[0], torch.tensor(0.))
    torch.testing.assert_close(all_rel[1], torch.sqrt(torch.tensor(.1 / 14.)))
    assert interior_rel.shape == (1,)
    torch.testing.assert_close(interior_rel[0],
                               torch.sqrt(torch.tensor(.1 / 13.)))


def test_all_boundary_batch_has_empty_interior_metric() -> None:
    batch = Batch.from_data_list([
        Data(x=torch.zeros((2, 1)), y=torch.tensor([1., 2.]),
             boundary_mask=torch.tensor([True, True])),
    ])
    all_rel, interior_rel = _graph_relative_errors(batch.y.clone(), batch)
    assert all_rel.shape == (1,)
    assert interior_rel.shape == (0,)
