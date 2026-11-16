# Online Retail II — research pipeline v0.1

## Source

Daqing Chen (2012). **Online Retail II**. UCI Machine Learning Repository,
DOI: https://doi.org/10.24432/C5CG6D (CC BY 4.0).
Download the official `online_retail_II.xlsx` from
https://archive.ics.uci.edu/dataset/502/online+retail+ii

The source workbook contains two yearly sheets. **Use both sheets for the paper.**
Do not use the 2010–2011-only Online Retail dataset (UCI ID 352) by mistake.

## Setup & command

```bash
python -m pip install -e ".[dev]"
python scripts/prepare_retail_ii.py \
  --input data/raw/online_retail_II.xlsx \
  --out results/retail_ii --history-days 90 --min-customers 15
pytest -q tests/test_retail_ii.py
```

For a quick local check, `--max-rows-per-sheet 10000` is allowed,
but **never report a sample run as a full-dataset result**.

## Methodological choices

1. Resolve both `Price` and `UnitPrice` column names, along with invoice and customer aliases.
2. Exclude cancellations (invoice begins with C), non-positive quantity/price,
   missing customer IDs or dates. This intentionally studies *positive purchases*,
   not net revenue after returns. Report this exclusion in the paper.
3. Calendar-month batches: retain customers with purchases during the calendar month.
4. Compute features from those customers' purchases within the preceding 90 days,
   strictly earlier than the following month's first timestamp:
   recency in days, distinct invoices, monetary, monetary per invoice, purchased quantity.
5. Apply `log1p` and freeze a `StandardScaler` fitted only on the FIRST eligible month.
   Never fit feature transformations on future batches.
6. Keep customer IDs alongside feature arrays to align overlapping customers across time.

### Important cautions

- This preprocessing **differs** from the archived CM-HRL notebook, which builds
  ~400-transaction batches and manually generated phase/drift labels. Retrain and
  rerun *all* compared models under this common preprocessing. Do not carry over
  old results or claim predefined drift ground truth.
- Month-end segmentation is calendar-based; seasonality and natural drift do not
  imply verified distribution-shift ground-truth labels.
- Some months can be skipped if fewer than `min_customers` are active.
- Between-month ARI is meaningful only for corresponding customer IDs; calculate
  it only on matched overlapping customers with adequate sample size.
- The reference baseline in `benchmark.py` still refits K-Means on each batch.
  It is NOT a frozen static-clustering baseline and does NOT establish H1.
- The dataset does not automatically validate customer-level intervention
  decisions or business action effectiveness.

## Next integration

Import published Hybrid-RL / CM-HRL functions through the shared interface;
measure cluster quality in the same common evaluation representation; add
prequential scoring, frozen-vs-adaptive baselines, latency breakdown, and
group-level temporal stability. Use >= 3 independent seeds and disclose runtime
hardware and all configuration choices. No model results have been established
by this pipeline alone.
