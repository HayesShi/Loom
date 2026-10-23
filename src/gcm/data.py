from __future__ import annotations

from pathlib import Path
from gcm.paths import data_path

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


DEFAULT_DATA_ROOT = data_path('planetoid')


def load_planetoid(name: str, root: str | Path = DEFAULT_DATA_ROOT):
    canonical = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[name.lower()]
    dataset = Planetoid(root=str(root), name=canonical, split="public", transform=NormalizeFeatures())
    return dataset, dataset[0]
