"""One training step on a node dataset, without saving outputs."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected

from loom import Loom, compile_graph, normalized_neighbors


def load_data(name, root, split):
    planetoid = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
    if name in planetoid:
        return Planetoid(
            str(root / "planetoid"),
            planetoid[name],
            split="public",
            transform=NormalizeFeatures(),
        )[0]
    path = root / "node_benchmarks" / name / "raw" / f"{name}.npz"
    with np.load(path, allow_pickle=False) as raw:
        x = torch.from_numpy(raw["node_features"].copy()).float()
        y = torch.from_numpy(raw["node_labels"].copy()).long()
        edges = torch.from_numpy(raw["edges"].copy()).long()
        if edges.shape[0] != 2:
            edges = edges.T
        data = Data(x=x, y=y, edge_index=to_undirected(edges, num_nodes=len(x)))
        for key in ("train", "val", "test"):
            mask = torch.from_numpy(raw[f"{key}_masks"].copy()).bool()
            if mask.ndim == 2:
                mask = mask.T if mask.shape[0] == len(x) else mask
                mask = mask[split]
            setattr(data, f"{key}_mask", mask)
    training = data.x[data.train_mask]
    data.x = (data.x - training.mean(0)) / training.std(0).clamp_min(1.0)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    data = load_data(args.dataset, args.data_root, args.split)
    mapping = compile_graph(data.edge_index, data.num_nodes).to(args.device)
    data = data.to(args.device)
    neighbors = normalized_neighbors(data.x, data.edge_index)
    model = Loom(
        data.x.shape[1],
        int(data.y.max()) + 1,
        data.num_nodes,
        dim=32,
        heads=4,
        blocks=2,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    logits = model(data.x, neighbors, mapping)
    loss = torch.nn.functional.cross_entropy(
        logits[data.train_mask], data.y[data.train_mask]
    )
    loss.backward()
    optimizer.step()
    print("Forward and backward passes completed.")


if __name__ == "__main__":
    main()
