"""Prepare calendar-month retail batches from official Online Retail II workbook.

Example:
python scripts/prepare_retail_ii.py --input data/raw/online_retail_II.xlsx --out results/retail_ii
"""
from __future__ import annotations
import argparse
from adaptiveclusterlab.data.retail_ii import (
    clean_transactions, read_retail_ii, monthly_rfm_batches, export_batches,
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Official UCI Online Retail II workbook")
    parser.add_argument("--out", default="results/retail_ii")
    parser.add_argument("--history-days", type=int, default=90)
    parser.add_argument("--min-customers", type=int, default=15)
    parser.add_argument("--max-rows-per-sheet", type=int, default=None,
                        help="Smoke test ONLY; paper experiments must omit this")
    args = parser.parse_args()
    transactions = clean_transactions(read_retail_ii(args.input, max_rows_per_sheet=args.max_rows_per_sheet))
    batches = monthly_rfm_batches(
        transactions, history_days=args.history_days, min_customers=args.min_customers,
    )
    summary = export_batches(batches, args.out)
    print(f"Clean purchase lines: {len(transactions):,}; months: {len(summary)}")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
