# Dataset setup

The [datasets-v1 release](https://github.com/HayesShi/Loom/releases/tag/datasets-v1) contains the complete inputs used by the manuscript experiments. Large files are stored as GitHub Release assets; the Git repository tracks the exact file inventory and download instructions. Model checkpoints, predictions and compiler caches are separate from this data release.

## Download and restore

Install [GitHub CLI](https://cli.github.com/) and authenticate with an account that can access the repository while it is private:

```bash
gh auth login
git clone https://github.com/Supradax/Higher-Order-Differential-Topology-aware-GNN external/Higher-Order-Differential-Topology-aware-GNN
python scripts/download_datasets.py --list
python scripts/download_datasets.py
```

Clone the 2Dele source before installing that package, unless it already exists. Its legacy data files depend on the upstream Python classes. This clone is unnecessary when selecting only other packages. If the data were restored first, move the `data/2Dele` subdirectory temporarily aside, clone into the empty upstream destination, and restore that subdirectory afterwards.

The complete download is about 3.49 GiB. The script restores the `data/` and `external/.../data/2Dele/` directories below the repository root. It verifies each asset, then reassembles Plate's six binary parts into the original 2.97 GB ZIP. Prepared Plate graphs are supplied separately, so training does not require extracting all original trajectory frames again.

Select a family when only part of the release is needed:

```bash
python scripts/download_datasets.py --only node-benchmarks paris
python scripts/download_datasets.py --only graphland lrim echo-sssp
python scripts/download_datasets.py --only plate-prepared 2dele hodge laplace
```

Use `--from-local /path/to/downloaded/assets` to restore assets downloaded manually. Assets are cached in `.dataset-downloads/` by default. `--root /path/to/checkout` changes the installation root; it creates `data/` beneath that directory. Restoring a package writes the release versions to the paths in its manifest.

Run experiments from the repository root. Loaders using `gcm.paths.data_path` default to `data/`; set `LOOM_DATA_ROOT` to relocate that data directory. Upstream loaders and adapters with explicit `--data-dir` or `--data-root` options retain their own conventions. GraphLand uses repository-relative `data/`, and 2Dele uses its upstream repository directory.

| Task family | Expected input |
| --- | --- |
| Cora, CiteSeer, PubMed | PyG Planetoid download under `data/planetoid` |
| Other node benchmarks | `data/node_benchmarks/<task>/raw/<task>.npz` |
| GraphLand | Official loader at `external/GraphLand/dataset.py`; official data under repository-relative `data/` |
| Paris | PyG CityNetwork download under `data/city_networks` |
| LRIM | Official loader at `external/LRIM/example-setup/lrim_loader.py`; data under `data/lrim` |
| ECHO SSSP | Official loader at `external/ECHO/PyG_implementation/ECHOBenchmarkDataset.py`; data under `data/echo_benchmark/official` |
| Plate | Prepared graphs and metadata under `data/plate_deformation/processed_static`; see `prepare_plate_deformation.py --help` |
| 2Dele | Upstream `external/Higher-Order-Differential-Topology-aware-GNN/data/2Dele` |
| Hodge | `data/hodge_magnetostatics/official_cpu_seed42_3000/flux_field_dataset.pkl` and `split_indices_public_code_seed42.npz` |
| Laplace | Released SNI data under `data/sni_laplace`; exact filenames are in `run_sni_laplace.py::FILES` |

The NPZ node loader expects `node_features` (N by F), `node_labels` (N), `edges` (2 by E or E by 2), and `train_masks`, `val_masks`, `test_masks`. Masks may be one-dimensional or contain multiple splits along either axis. Preserve the original benchmark split when preparing these files. The loader makes edges undirected, coalesces them, and normalizes features using training-node statistics, with standard deviations clamped to at least 1. The release includes the complete NPZ inputs with their saved masks; these are experiment-format graphs, not model outputs. Original Planetoid files are also supplied for the standard Cora, CiteSeer and PubMed loader. Installing the Python package alone does not download data; run the command above.

## Release contents

| Package | Complete contents |
| --- | --- |
| `node-benchmarks` | Cora, CiteSeer, PubMed, DBLP, WikiCS, ogbn-arxiv, Coauthor-CS, Coauthor-Physics, Roman Empire, Amazon Ratings, Chameleon-filtered, Minesweeper, Tolokers and Questions; 24 original Planetoid files |
| `graphland` | Roads-M, Tolokers-2, Artnet-exp, Artnet-views, City-reviews, Twitch-views, plus Roads-L for scaling |
| `paris` | Original node/edge features, augmented features, topology, labels and masks |
| `lrim` | Four 10,000-graph variants: side lengths 16 and 32, exponents 0.6 and 1.5 |
| `echo-sssp` | Original train/validation/test files covering all six topology families |
| `2dele` | Original 120/15/15 training/validation/test partitions |
| `hodge` | All 3,000 samples generated with the upstream magnetostatics solver, seed 42; saved 2,040/360/600 split and generation provenance |
| `laplace` | 20,000 training and 2,500 validation examples; 100 test examples in each of the three transfer domains; reference meshes |
| `plate-prepared` | All 1,000 prepared static graphs, 800/100/100 split and preparation metadata |
| `plate-original` | Byte-for-byte original trajectory ZIP, split into six upload assets and reassembled by the downloader |

The million-node synthetic scaling inputs are generated by `scripts/benchmark_scaling.py` with its recorded seed. They are not a separately sourced raw dataset. Unrelated earlier experiments and duplicate PyG/compilation caches are not part of this release.

## Sources and attribution

The inventory in [`datasets/manifest.json`](../datasets/manifest.json) records upstream sources, archive sizes, digests, file paths and whether a package is original, prepared, or generated with an upstream solver. Data retain the original authors' rights and citation requirements; this collection does not assign them a new license. The source-code license notices available in the workspace are preserved in [`datasets/notices/`](../datasets/notices/); those notices do not replace dataset-specific terms.

Use the upstream projects linked in [third_party.md](third_party.md) to install the external loaders before running GraphLand, LRIM, ECHO or 2Dele. The release supplies their data, while their code remains in the original repositories. Existing generated Hodge samples and prepared Plate graphs are identified explicitly; they are inputs to the experiments, not fabricated model results.
