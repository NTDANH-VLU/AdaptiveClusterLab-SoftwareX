# DAMADICS external baseline check (2026-11-15)

## Dataset and provenance

Source: [MPiangerelli/FURAKI data/DAMADICS.csv](https://github.com/MPiangerelli/FURAKI/blob/main/data/DAMADICS.csv), a **reduced 3,600-record subset**, NOT the original ~2.25 million record industrial time series. Source: FURAKI repository README and data file.

The file has 32 numeric features and one additional `drifting` fault/category label. This label is **excluded** from K-Means fitting, preprocessing and method-selection. It is only used for external ARI evaluation. Temporal block order of the reduced dataset must **NOT** be presented as original timestamp continuity.

Protocol: eight consecutive reduced-dataset blocks of 450 records; four development blocks and four evaluation blocks. This historical diagnostic uses three seeds 42/43/44 (distinct from the five-seed 42–46 audited SoftwareX manuscript study); K=3/5; stable row-offset train/test splits per block; first-block train-only StandardScaler; input-space held-out metrics. This is a cross-domain **software baseline reproducibility check**, not an RL model test or original production-time drift validation.

## Verified GitHub Actions execution

Workflow: `.github/workflows/damadics-benchmark.yml`. Run ID: `43676744668`, **completed successfully**. CSV and stdout summary are preserved in workflow artifact `damadics-baseline-results`. See GitHub Actions run (earlier reference result).

Results are means across **24 held-out block × K × seed evaluation combinations per method**:

| Method | Silhouette ↑ | DBI ↓ | External fault ARI ↑ | K-Means fits in eval | Memory acceptance |
|---|---:|---:|---:|---:|---:|
| Frozen | 0.224725 | 1.767719 | 0.128218 | 0 | 0 |
| Refit | 0.588532 | 0.597178 | 0.336117 | 24 | 0 |
| Memory gate | 0.637227 | 0.570256 | 0.336101 | 24 | 6 |
| Always reuse | 0.224725 | 1.767719 | 0.128218 | 0 | 24 |

Memory-gated clustering improves geometric criteria in this diagnostic comparison, **not** external fault ARI. It still incurs **24 fresh K-Means fits**, so no measured retraining avoidance can be claimed. Frozen and always-reuse have identical predictions because always-reuse prototypes are never updated.

**No claims of statistical significance, true temporal adaptation, learned RL policy, or cost reduction** are supported. FURAKI data are reduced and chronologically ambiguous; full-data verification is a separate task.

To reproduce:

```bash
curl -L -o DAMADICS.csv https://raw.githubusercontent.com/MPiangerelli/FURAKI/main/data/DAMADICS.csv
python benchmarks/damadics_baselines.py --input DAMADICS.csv \
  --output results/damadics_baselines.csv --seeds 42 43 44 --ks 3 5
```

**Historical note:** this earlier exploratory check predates the eight-policy audited five-seed SoftwareX evaluation. Current Electricity and five-seed results are in [the audit report](Q_POLICY_AUDIT_2026_11_17.md); do not conflate these protocols.
