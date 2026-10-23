# Loom

Loom is a graph neural network that compiles graph topology into node placement on a fixed butterfly network. Its parallel, multiscale schedule exchanges information through an adaptive rotary coupling operator (ARCO), while retaining a separate state for each node.

## Installation

Use Python 3.10 or newer. Install a [PyTorch](https://pytorch.org/get-started/locally/) build appropriate for your hardware, then run:

```bash
git clone https://github.com/HayesShi/Loom.git
cd Loom
python -m pip install -e ".[test,data]"
python examples/quickstart.py
```

The quickstart compiles a small graph and performs one optimization step on CPU. The implementation was checked with PyTorch 2.5.1 and PyG 2.7.0. Experiment training and timing through the campaign runner require an NVIDIA GPU and CUDA-enabled PyTorch. Custom CUDA kernels additionally require a compatible CuPy installation; they are optional for the ordinary PyTorch model.

## Experiments

Run commands from this directory. A single-task run is:

```bash
python scripts/run_loom_campaign.py --tasks cora --seed 0 --output results/cora_s0
```

The runner selects configurations and epochs on validation data and saves run records, checkpoints, and timing measurements. Use a different output directory for each seed. See [experiment instructions](docs/experiments.md) and [dataset setup](docs/datasets.md) before running the full campaign.

The complete experiment inputs are available in the [dataset release](https://github.com/HayesShi/Loom/releases/tag/datasets-v1). Download and restore them with:

```bash
git clone https://github.com/Supradax/Higher-Order-Differential-Topology-aware-GNN external/Higher-Order-Differential-Topology-aware-GNN
python scripts/download_datasets.py
```

Clone the 2Dele source once before restoring all data; skip that clone if it already exists. The downloader uses GitHub CLI (`gh`) for authenticated access to a private repository. Use `--list` to see the packages or `--only node-benchmarks` to restore one family. Original Plate trajectories and the prepared static graphs are both included; the downloader reassembles the large original archive automatically.

## Code map

| Location | Purpose |
| --- | --- |
| `src/gcm/loom.py` | Loom model, layout compilation, and rotary coupling |
| `src/gcm/hamming_compiler.py` | Topology-aware placement optimization |
| `src/gcm/loom_cuda.py` | Optional CUDA execution |
| `scripts/run_loom_campaign.py` | Main Loom experiment entry point |
| `scripts/loom_graph_campaign.py` | Graph-level and physical-field task integration |
| `scripts/benchmark_scaling.py` | Inference scaling measurements |
| `tests/` | Operator, compiler, adapter, and execution tests |

The Python package retains the name `gcm` for compatibility with existing imports and checkpoints. Supporting modules and task adapters include controls used by the experiments; use `run_loom_campaign.py` to train Loom itself.

## Tests

```bash
python -m pytest -q
```

CUDA-specific tests are skipped when their required hardware or dependencies are unavailable. Large datasets are stored as Release assets rather than Git blobs. Model checkpoints, generated reports, and third-party code repositories are not bundled. See [third-party sources](docs/third_party.md).
