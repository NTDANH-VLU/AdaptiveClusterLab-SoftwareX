# AdaptiveClusterLab

AdaptiveClusterLab is an open-source Python research workbench for studying clustering decisions as incoming data evolve. It includes temporal data adapters, a train/update/reuse controller, reference K-Means baselines, and scripts for evaluating each decision on held-out observations.

The experimental controller uses tabular Q-learning and a clustering autoencoder. It is a **research implementation**, not a claim that an adaptive policy is consistently better than full retraining, and it does not reproduce the original published neural Hybrid-RL or CM-HRL controllers.

## Install

Python 3.11 was used for the reproducibility checks. Install [uv](https://docs.astral.sh/uv/) and clone this standalone repository:

```bash
git clone https://github.com/NTDANH-VLU/AdaptiveClusterLab-SoftwareX.git
cd AdaptiveClusterLab-SoftwareX
uv sync --python 3.11 --all-extras --locked
uv run --all-extras --locked python -m pytest -q
uv run --all-extras --locked python scripts/reproduce.py demo --quick
```

The synthetic demo needs no external dataset. To install with pip instead of uv, run `python -m pip install -e ".[dev,research]"` from the checkout.

## Audited source and reproduction results

The original `v1.0.0` release is retained as a historical baseline and does **not** contain the updated Q-action coverage experiment. The paper's audited revision is the `main` source with its permanently committed [five-seed evaluation CSVs and checksums](reproducibility/audited-2026-11-17/PROVENANCE.json), validated by [GitHub Actions run 44039942375](https://github.com/NTDANH-VLU/AdaptiveClusterLab-SoftwareX/actions/runs/44039942375). Cite a full Git commit hash of this revision for exact reproduction; `main` can move.

## Reproduce the paper experiments

The SoftwareX experiment protocol uses independent seeds 42–46, an operation-count reward (not elapsed CPU time), five Q-learning development episodes, two full autoencoder epochs and one update epoch:

```bash
python scripts/reproduce.py retail --download --cost-mode operation_counts --seeds 42 43 44 45 46
python scripts/reproduce.py damadics --download --cost-mode operation_counts --seeds 42 43 44 45 46
python scripts/reproduce.py electricity --download --cost-mode operation_counts --seeds 42 43 44 45 46
```

- **Online Retail II**: 24 monthly batches; 90-day customer RFM features; 12 development and 12 evaluation months, with up to 160 customers in each model partition.
- **Reduced DAMADICS / FURAKI**: 3,600 rows, 32 feature columns and eight blocks in source order; four development blocks and four evaluation blocks; cap 110 rows per partition. The reduced file order is **not** the original industrial time sequence.
- **Electricity Load Diagrams**: UCI dataset 321, 24 months from 2012–2013 with 34 engineered consumption features; 12 development and 12 evaluation months; cap 110 active consumers per partition. First-month scaling is transductive.

Downloads use upstream public dataset providers; raw datasets and unpublished notebooks are **not** stored in Git. The original datasets may have separate licenses.

Every run saves batch-level results to `results/<dataset>/evaluation.csv`, summaries to `summary.json`, and the independent audit to `audit_report.json`. Silhouette reporting separates valid scores from invalid single-cluster outputs (with an explicit degeneracy count), while the legacy penalized score is retained only as a reward diagnostic. The CI workflow `five-seed.yml` reruns all three studies. A stable snapshot of the audited CSVs is stored under `reproducibility/audited-2026-11-17/` in Git, with SHA-256 checksums.

## Methods and limitations

The evaluated policies include `always_retrain`, `always_update`, `reuse_when_available`, `q_learning`, `q_learning_gated`, `q_learning_fixed`, `q_learning_covered` and `q_learning_fixed_covered`. `q_learning_covered` permits only actions whose exact Q state–action pair was updated in policy development and otherwise explicitly falls back to retraining. `q_learning_fixed_covered` additionally holds K=3 and d=8. The controller logs full fits, updates, reuses, tie-breaking, selected-action Q visits, and fallback events. Its action cost is a chosen proxy: one unit per full AE fit, 0.25 unit per update and zero units per reuse. It is not a validated measurement of energy or overall runtime.

The audited five-seed studies demonstrate traceable adaptive decisions on three datasets. The covered controller avoids 45.0%, 20.0%, and 18.3% of full AE fits on Retail II, DAMADICS and Electricity, respectively, while attaining lower mean valid Silhouette than always retraining. These reductions are counts, **not measured runtime or energy savings**. The detailed [Q-policy audit](docs/Q_POLICY_AUDIT_2026_11_17.md) reports fallback frequencies and clustering degeneracy.

The [results summary](docs/THREE_DATASET_FIVE_SEED_VALIDATION.md) and [dataset protocols](docs/ELECTRICITY_PIPELINE.md) describe the experiments in more detail.

## Package contents

- `src/adaptiveclusterlab/`: installable controller, autoencoder, reference K-Means and data loader
- `scripts/`: dataset preparation and the reproduction entry point
- `experiments/controlled/`: three-action runner and controlled A–B–A example
- `benchmarks/`: independent K-Means comparisons
- `tests/`: automated validation
- `docs/`: preparation protocols and observed results
- `.github/workflows/`: continuous integration and reproducibility

## Citation and license

The source code is available under the [MIT license](LICENSE). The tagged `v1.0.0` remains an earlier historical release; see the audited Git commit and archived CSVs for the revised manuscript results. See [CITATION.cff](CITATION.cff) for author metadata. No software DOI has been assigned.
