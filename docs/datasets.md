# Dataset setup

The repository excludes raw data and generated caches. Run experiments from the repository root. Loaders using `gcm.paths.data_path` default to `data/`; set the environment variable `LOOM_DATA_ROOT` to relocate that root. Upstream loaders and adapters with explicit `--data-dir` or `--data-root` options retain their own path conventions.

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

The NPZ node loader expects `node_features` (N by F), `node_labels` (N), `edges` (2 by E or E by 2), and `train_masks`, `val_masks`, `test_masks`. Masks may be one-dimensional or contain multiple splits along either axis. Preserve the original benchmark split when preparing these files. The loader makes edges undirected, coalesces them, and normalizes features using training-node statistics, with standard deviations clamped to at least 1. The processed NPZ files and physical-field datasets must be obtained or prepared separately; installing this package does not download them all.

Use the upstream projects linked in [third_party.md](third_party.md) for their data releases and preparation instructions. Data and split provenance are required to compare new runs with existing results; the repository does not replace missing observations with generated scores.
