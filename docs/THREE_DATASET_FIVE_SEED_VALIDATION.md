# Five-seed controller validation across three datasets

These are the results reported in the SoftwareX manuscript. Every controller experiment uses seeds **42, 43, 44, 45, 46**, five policy-development episodes, two full AE training epochs, and the `operation_counts` reward. The evaluation split is held out from action selection. The five-seed runs are short-budget software evaluations, not comparisons with fully trained published Hybrid-RL or CM-HRL neural policies.

| Dataset | GitHub Actions run | Held-out batches per method | Development / evaluation | Per-partition cap |
| --- | --- | ---: | --- | ---: |
| Online Retail II | [43694292478](https://github.com/NTDANH-VLU/AdaptiveClusterLab-SoftwareX/actions/runs/43694292478) | 60 | 12 / 12 months | 160 |
| Reduced DAMADICS | [43694295584](https://github.com/NTDANH-VLU/AdaptiveClusterLab-SoftwareX/actions/runs/43694295584) | 20 | 4 / 4 source-order blocks | 110 |
| Electricity | [43692866424](https://github.com/NTDANH-VLU/AdaptiveClusterLab-SoftwareX/actions/runs/43692866424) | 60 | 12 / 12 months | 110 |

## Controller results

Each policy's Silhouette and Davies–Bouldin index (DBI) is the mean over its held-out batches. For DBI, undefined values are excluded. The action counts show **full AE fits / updates / whole-model reuses** across five seeds.

| Dataset | Policy | Silhouette ↑ | DBI ↓ | Fits / updates / reuses |
| --- | --- | ---: | ---: | --- |
| Retail II | Always retrain | 0.28207 | 1.34157 | 60 / 0 / 0 |
| Retail II | Always update | 0.25795 | 1.34332 | 0 / 60 / 0 |
| Retail II | Always reuse | 0.13548 | 1.57977 | 0 / 0 / 60 |
| Retail II | Q-learning | 0.25900 | 1.43406 | 49 / 6 / 5 |
| Retail II | Q-learning + gate | 0.26218 | 1.41749 | 35 / 16 / 9 |
| DAMADICS | Always retrain | 0.70952 | 0.50884 | 20 / 0 / 0 |
| DAMADICS | Always update | 0.40945 | 1.20998 | 0 / 20 / 0 |
| DAMADICS | Always reuse | −0.60733 | 1.97495 | 0 / 0 / 20 |
| DAMADICS | Q-learning | 0.62112 | 0.70925 | 16 / 4 / 0 |
| DAMADICS | Q-learning + gate | 0.62112 | 0.70925 | 16 / 4 / 0 |
| Electricity | Always retrain | 0.35490 | 1.50083 | 60 / 0 / 0 |
| Electricity | Always update | 0.27387 | 1.39393 | 0 / 60 / 0 |
| Electricity | Always reuse | 0.26972 | 1.43093 | 0 / 0 / 60 |
| Electricity | Q-learning | 0.29996 | 1.35829 | 25 / 6 / 29 |
| Electricity | Q-learning + gate | 0.28844 | 1.54943 | 28 / 27 / 5 |

The controller makes real reuse decisions, but this does not mean overall quality or runtime improves. Mean Silhouette is lower than always retraining for all three domains. Reuse is especially harmful on the reduced DAMADICS dataset. The DAMADICS controller results include **14 undefined DBI values among 100 evaluations**; Electricity has 5 among 300, and Retail II has none. The DAMADICS row order is not the original timestamped industrial sequence.

## K-Means reference baselines

These are separate models, each assessed over five seeds and two cluster counts (K = 3, 5).

| Dataset | Method | Held-out evaluations | Silhouette ↑ | DBI ↓ | Additional measure |
| --- | --- | ---: | ---: | ---: | --- |
| Retail II | Frozen | 120 | 0.11500 | 1.33304 | Temporal ARI 0.29132 |
| Retail II | Refit | 120 | 0.30021 | 1.11551 | Temporal ARI 0.10552 |
| Retail II | Memory gate | 120 | 0.31671 | 1.09288 | Temporal ARI 0.13522 |
| DAMADICS | Frozen | 40 | 0.23723 | 1.74192 | Fault ARI 0.14101 |
| DAMADICS | Refit | 40 | 0.58755 | 0.59078 | Fault ARI 0.33960 |
| DAMADICS | Memory gate | 40 | 0.63196 | 0.58928 | Fault ARI 0.34082 |
| Electricity | Frozen | 120 | 0.30942 | 1.08626 | Temporal ARI 0.89854 |
| Electricity | Refit | 120 | 0.32012 | 1.16495 | Temporal ARI 0.79490 |
| Electricity | Memory gate | 120 | 0.32607 | 1.16538 | Temporal ARI 0.81724 |

The baseline source runs are [Retail II #43694628784](https://github.com/NTDANH-VLU/AdaptiveClusterLab-SoftwareX/actions/runs/43694628784), [DAMADICS #43694631491](https://github.com/NTDANH-VLU/AdaptiveClusterLab-SoftwareX/actions/runs/43694631491), and [Electricity #43692866424](https://github.com/NTDANH-VLU/AdaptiveClusterLab-SoftwareX/actions/runs/43692866424).

The memory-gate baseline still fits a candidate at every evaluation step: 120 times on Retail II, 40 on DAMADICS, and 120 on Electricity. Its Silhouette scores therefore should not be described as evidence of lower K-Means training cost.

The output CSVs, per-run manifests, and provenance are bundled with [GitHub Release v1.0.0](https://github.com/NTDANH-VLU/AdaptiveClusterLab-SoftwareX/releases/tag/v1.0.0). The original experiment runs are retained as links for traceability.
