# UCI Electricity: 34-feature monthly clustering protocol

UCI ElectricityLoadDiagrams20112014, ID 321:
https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
(DOI: 10.24432/C58C86).

**Reproduction:** The five-seed protocol described here was used for the manuscript. Use the clean-repository workflow to independently rerun and verify the results.
[Reproduction workflows](https://github.com/NTDANH-VLU/AdaptiveClusterLab-SoftwareX/actions/workflows/five-seed.yml);
[full five-seed findings](ELECTRICITY_FIVE_SEED_RESULTS.md).

The source `LD2011_2014.txt` is a semicolon-delimited, comma-decimal
text file with 370 source consumer columns and 15-minute kW observations.
The adapter streams Jan 2012–Dec 2013 into 24 monthly batches, including
321–349 active customers/month. Only customers with positive consumption
in a month are included. The 34 features per active consumer are:
24 hourly energy shares, seven weekday energy shares, log1p(total monthly
kWh), log1p(peak kW), and load factor (mean/peak). kW is converted to
kWh by dividing each 15-minute reading by 4. Original source local-time
conventions are kept; no DST resampling is performed.

The scaler is fitted once on the first development month's full active
customer set and then frozen. This is transductive first-month
normalization, **not strictly per-seed train-ID-only scaling**.
No later months enter its fit.

The first 12 months are used for controller development and the last
12 for held-out evaluation. Each seed assigns customer IDs consistently
to 60/20/20 train/validation/test partitions. At most **110 consumers
per partition per month** are used. The five-seed protocol uses
five developmental episodes, 2 full AE epochs, 1 update epoch and
the `operation_counts` reward (1 per full AE fit, 0.25 per AE update,
0 per reuse). These are subjective proxy weights, not calibrated costs.

## Reproduction

```bash
python -m pip install -e ".[dev,research]"
python -m pytest -q tests/test_electricity.py
python scripts/reproduce.py electricity --download \
  --cost-mode operation_counts --seeds 42 43 44 45 46
```

Outputs under `results/electricity/` include
`batches/manifest.csv`, `evaluation.csv`, `summary.json`, and
`temporal_baselines.csv`. The user must archive these outputs because
GitHub Actions artifacts expire. No public electricity-cluster ground-truth
classes or downstream business improvement is established.

The experiments verify real policy actions, including failures. Five-seed
ungated Q-learning held-out mean Silhouette 0.3000 was below always-retrain
0.3549, and its training fit counts do not prove net speedup.
