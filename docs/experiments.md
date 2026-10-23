# Running experiments

Run from the repository root after installing the package. The main runner requires CUDA, including when CUDA graph capture is disabled.

```bash
python scripts/run_loom_campaign.py --help
python scripts/run_loom_campaign.py --tasks cora citeseer pubmed --seed 0 --output results/classic_s0
python scripts/run_loom_campaign.py --tasks roman_empire amazon_ratings chameleon_filtered minesweeper tolokers questions --seed 0 --output results/heterophily_s0
python scripts/run_loom_campaign.py --tasks city-roads-M tolokers-2 artnet-exp artnet-views city-reviews twitch-views paris --seed 0 --output results/applications_s0
python scripts/run_loom_campaign.py --tasks echo_sssp plate 2dele hodge laplace --seed 0 --output results/fields_s0
```

Dataset setup is described in [datasets.md](datasets.md). LRIM tasks are `lrim_16_0.6_10k`, `lrim_32_0.6_10k`, `lrim_16_1.5_10k`, and `lrim_32_1.5_10k`. Synthetic communication studies are `reuse_1`, `reuse_4`, `reuse_16`, `transfer_1`, `transfer_8`, `transfer_32`, `scale_511`, `scale_512`, `scale_513`, and `scale_8192`; their data are generated locally.

Omitting `--tasks` runs the full campaign. Task configurations are defined in `configs()` in `run_loom_campaign.py` and `configurations()` in `loom_graph_campaign.py`. The node runner uses split 0; `--seed` controls training randomness, not dataset split selection. Keep seeds in separate output directories to avoid reusing another seed's cached results. `--configs` limits the configuration search; leave it at its default for the full search. `--no-cuda-graph` disables CUDA graph capture for timing.

`finalize_loom_campaign.py` refreshes timing records for the complete campaign in `results/loom`. `verify_loom_campaign.py` audits its complete seed-0 records, including 42 reported task/domain outcomes. These utilities are for a finished full campaign, not a partial run.

For a synthetic scaling measurement:

```bash
python scripts/benchmark_scaling.py --nodes 4096 --method loom --repeats 10
```

The `loom_cuda` and `loom_cuda_gc` modes require CuPy; SGFormer modes require the upstream implementation in `external/SGFormer/medium` and its dependencies. Compare measurements on the same device with the same inputs, and retain compilation time separately from inference time. Individual task adapters retain their original control models and diagnostic options; they are also used as data loaders by the main Loom runner.
