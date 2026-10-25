# Loom

Core implementation of **Compile Graphs, Compute Globally**.
Loom compiles graph topology into node placement on a fixed butterfly network
and exchanges information through adaptive rotary coupling (ARCO).

## Install

Use Python 3.10+ and a [PyTorch](https://pytorch.org/get-started/locally/)
build suitable for your hardware.

```bash
git clone https://github.com/HayesShi/Loom.git
cd Loom
python -m pip install -e ".[test,data]"
```

## Core modules

| Module | Manuscript component |
|---|---|
| `src/loom/compiler.py` | Fiedler ordering, Gray addresses, and Hamming-cost refinement |
| `src/loom/reference.py` | Reference grid addresses and contact-based Plate placement |
| `src/loom/arco.py` | Content controller and two-output rotary coupling |
| `src/loom/butterfly.py` | Fixed XOR schedule and grouped parallel execution |
| `src/loom/model.py` | Graph embedding, persistent wire states, residual blocks, and prediction head |

The compiler uses at most four refinement sweeps. Every block runs the full
butterfly schedule; padding is initialized to zero and remains active during
mixing. ARCO uses the algebraically fused form of the three additive updates.
Parameters are shared across pairs within a stage and separate across stages
and blocks. Classification returns logits for use with cross-entropy.

```python
from loom import Loom, compile_graph, normalized_neighbors

mapping = compile_graph(edge_index, num_nodes=x.shape[0])
neighbors = normalized_neighbors(x, edge_index)
model = Loom(x.shape[1], num_classes, max_nodes=x.shape[0])
logits = model(x, neighbors, mapping)
```

For undirected inputs, provide both directions of each edge. Node features,
neighbor features, mappings, and the model should be on the same device.
Cache the layout and neighbor features when the graph and inputs are static.
For reference layouts with a larger wire range, pass its capacity as `padded`.

## Data

```bash
python scripts/download_data.py --list
python scripts/download_data.py --only node-benchmarks
```

Omit `--only` to download all dataset families. Data are distributed through
the [Datasets release](https://github.com/HayesShi/Loom/releases/tag/datasets-v1).
See [dataset instructions](datasets/README.md) for sources, formats, and splits.

## Example and checks

```bash
python examples/node_classification.py --dataset cora
python -m pytest -q
```

The example performs one training step without saving measurements or weights.
Tests cover the coupling equations, conservation and directional transfer,
compiler refinement, pair scheduling, and persistent-state execution.
