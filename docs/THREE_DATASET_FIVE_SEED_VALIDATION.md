> **HISTORICAL v1.0.0 RESULTS — NOT the revised SoftwareX manuscript protocol.**
> The old table below uses penalized Silhouette averages and deterministic index-based Q ties.
> For the audited, corrected five-seed study (including valid Silhouette, Q-action coverage and matched K/d),
> use [the new audit report](Q_POLICY_AUDIT_2026_11_17.md) and
> [committed CSVs](../reproducibility/audited-2026-11-17/PROVENANCE.json).

# Five-seed validation reference

This document preserves an earlier five-seed evaluation for historical comparison. **Its controller table is not the final SoftwareX manuscript Table 3.** The submitted manuscript uses valid-Silhouette and action-coverage results in [Q_POLICY_AUDIT_2026_11_17.md](Q_POLICY_AUDIT_2026_11_17.md) and the [audited results archive](../reproducibility/audited-2026-11-17/PROVENANCE.json). The standalone repository contains the source code, the complete reproduction workflow and independently verified outputs. **Verified on the clean repository:** [GitHub Actions run 43808723741](https://github.com/NTDANH-VLU/AdaptiveClusterLab-SoftwareX/actions/runs/43808723741) successfully completed all three independent five-seed controllers and baseline evaluations. The downloaded new evaluation and baseline CSVs reproduce the means and operation counts reported below.

All methods used seeds 42–46, five policy-development episodes, two full autoencoder epochs, one update epoch and the operation-count reward. Test data were held out of controller action selection.

| Dataset | Development / evaluation | Held-out evaluations per method | Sample cap per partition |
| --- | --- | ---: | ---: |
| UCI Online Retail II | 12 / 12 monthly batches | 60 | 160 |
| Reduced FURAKI DAMADICS | 4 / 4 blocks in file order | 20 | 110 |
| UCI Electricity | 12 / 12 monthly batches | 60 | 110 |

## Historical controller results (superseded; do not cite as the final Table 3)

The action counts are full AE fits / model updates / real model reuses.

| Dataset | Method | Mean Silhouette (higher) | Mean DBI (lower) | Fits / updates / reuses |
| --- | --- | ---: | ---: | --- |
| Retail II | Always retrain | 0.28207 | 1.34157 | 60 / 0 / 0 |
| Retail II | Always update | 0.25795 | 1.34332 | 0 / 60 / 0 |
| Retail II | Always reuse | 0.13548 | 1.57977 | 0 / 0 / 60 |
| Retail II | Q-learning | 0.25900 | 1.43406 | 49 / 6 / 5 |
| Retail II | Q-learning + gate | 0.26218 | 1.41749 | 35 / 16 / 9 |
| DAMADICS | Always retrain | 0.70952 | 0.50884 | 20 / 0 / 0 |
| DAMADICS | Always update | 0.40945 | 1.20998 | 0 / 20 / 0 |
| DAMADICS | Always reuse | -0.60733 | 1.97495 | 0 / 0 / 20 |
| DAMADICS | Q-learning | 0.62112 | 0.70925 | 16 / 4 / 0 |
| DAMADICS | Q-learning + gate | 0.62112 | 0.70925 | 16 / 4 / 0 |
| Electricity | Always retrain | 0.35490 | 1.50083 | 60 / 0 / 0 |
| Electricity | Always update | 0.27387 | 1.39393 | 0 / 60 / 0 |
| Electricity | Always reuse | 0.26972 | 1.43093 | 0 / 0 / 60 |
| Electricity | Q-learning | 0.29996 | 1.35829 | 25 / 6 / 29 |
| Electricity | Q-learning + gate | 0.28844 | 1.54943 | 28 / 27 / 5 |

Unreported DBI values are omitted from mean DBI: 14 of 100 reduced DAMADICS controller evaluations and 5 of 300 Electricity evaluations have undefined DBI. Reuse reduces full-fit counts but does not establish savings in measured runtime or superior overall clustering geometry.

## Reference K-Means baselines

| Dataset | Method | Held-out runs | Silhouette | DBI |
| --- | --- | ---: | ---: | ---: |
| Retail II | Frozen | 120 | 0.11500 | 1.33304 |
| Retail II | Refit | 120 | 0.30021 | 1.11551 |
| Retail II | Memory gate | 120 | 0.31671 | 1.09288 |
| DAMADICS | Frozen | 40 | 0.23723 | 1.74192 |
| DAMADICS | Refit | 40 | 0.58755 | 0.59078 |
| DAMADICS | Memory gate | 40 | 0.63196 | 0.58928 |
| Electricity | Frozen | 120 | 0.30942 | 1.08626 |
| Electricity | Refit | 120 | 0.32012 | 1.16495 |
| Electricity | Memory gate | 120 | 0.32607 | 1.16538 |

The memory gate still fits a new K-Means candidate on every evaluation block. The measured Silhouette therefore should not be used to claim reduced K-Means training cost.

## Regeneration

Run `python scripts/reproduce.py DATASET --download --cost-mode operation_counts --seeds 42 43 44 45 46` with `DATASET` equal to `retail`, `damadics`, or `electricity`. Alternatively, launch the GitHub Actions workflow `Five-seed SoftwareX reproducibility` in this repository. Data preparation details are in the dataset-specific documents.
