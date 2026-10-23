"""Focused invariants for the optional coordinate-address SNI compiler."""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_sni_laplace import (  # noqa: E402
    SniDataset,
    _normalized_xy,
    _spatial_morton_anchor_mapping,
    _spatial_morton_mapping,
)
import run_sni_laplace  # noqa: E402


@pytest.mark.parametrize("nodes", [30, 88, 159, 549, 1038])
def test_anchored_compiler_is_unique_and_relabeling_equivariant(nodes: int) -> None:
    rng = np.random.default_rng(nodes)
    xy = _normalized_xy(rng.uniform(-1, 1, size=(nodes, 2)))
    mapping = _spatial_morton_anchor_mapping(xy)
    padded = 1 << (nodes - 1).bit_length()
    assert mapping.dtype == torch.int64
    assert len(mapping.unique()) == nodes
    assert int(mapping.min()) >= 0
    assert int(mapping.max()) < padded

    permutation = rng.permutation(nodes)
    torch.testing.assert_close(
        _spatial_morton_anchor_mapping(xy[permutation]), mapping[permutation]
    )
    torch.testing.assert_close(_spatial_morton_anchor_mapping(xy), mapping)


def test_anchored_compiler_ignores_translation_and_uniform_scale() -> None:
    rng = np.random.default_rng(17)
    xy = rng.integers(-50, 51, size=(73, 2)).astype(np.float64)
    original = _spatial_morton_anchor_mapping(_normalized_xy(xy))
    transformed = _spatial_morton_anchor_mapping(_normalized_xy(4 * xy + 200))
    torch.testing.assert_close(transformed, original)


def test_existing_rank_layout_keeps_its_reference_wires() -> None:
    xy = np.array([[-.5, -.5], [.5, -.5], [-.5, .5], [.5, .5], [0., 0.]])
    torch.testing.assert_close(
        _spatial_morton_mapping(xy), torch.tensor([0, 4, 5, 7, 2])
    )


def test_dataset_dispatches_to_optional_anchor_layout(tmp_path, monkeypatch) -> None:
    xy = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [.5, .5]])
    sol = np.column_stack((xy, np.zeros(len(xy))))
    boundary = np.column_stack((xy[:4], np.zeros(4), np.zeros(4)))
    source = tmp_path / "one_mesh.pkl"
    with source.open("wb") as handle:
        pickle.dump([(sol, [boundary])], handle)
    monkeypatch.setitem(run_sni_laplace.COUNTS, "train", 1)
    graph = SniDataset(source, "train", 1.3, "spatial_morton_anchor")[0]
    torch.testing.assert_close(
        graph.wire_mapping,
        _spatial_morton_anchor_mapping(_normalized_xy(xy)),
    )
