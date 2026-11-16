"""Leakage-aware Online Retail II monthly customer RFM pipeline.

This is a NEW standardized data adapter, NOT a modification of the archived
CM-HRL notebook. The original notebook uses row-count batches and different
features; resulting metrics must not be compared without rerunning both methods.

Dataset: Daqing Chen (2012), UCI Online Retail II, doi:10.24432/C5CG6D.
License: CC BY 4.0. Download data separately; never commit raw workbook.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ("recency_days", "frequency", "monetary", "avg_invoice_value", "quantity")
ALIASES = {
    "invoice": "invoice", "invoiceno": "invoice", "invoice_no": "invoice",
    "invoicedate": "date", "invoice_date": "date", "date": "date",
    "price": "price", "unitprice": "price", "unit_price": "price",
    "customer id": "customer_id", "customerid": "customer_id",
    "customer_id": "customer_id", "quantity": "quantity",
}
REQUIRED = ("invoice", "date", "price", "customer_id", "quantity")

@dataclass(frozen=True)
class RetailBatch:
    """A calendar-month batch of active customers with trailing history features."""
    month: str
    customer_ids: NDArray
    raw_features: NDArray[np.float64]
    features: NDArray[np.float64]
    cutoff: pd.Timestamp

def read_retail_ii(path: str | Path, *, max_rows_per_sheet: int | None = None) -> pd.DataFrame:
    """Load all original workbook sheets (2009-10 and 2010-11) by default.

    max_rows_per_sheet is ONLY for local smoke tests and is NOT a valid
    replacement for the complete research dataset.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Online Retail II workbook not found: {path}")
    if path.suffix.lower() not in (".xlsx", ".xls"):
        raise ValueError("Expected the original UCI Excel workbook (.xlsx/.xls)")
    sheets = pd.read_excel(path, sheet_name=None, nrows=max_rows_per_sheet)
    if not sheets:
        raise ValueError("Workbook has no sheets")
    return pd.concat(sheets.values(), ignore_index=True)

def clean_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Select positive non-cancelled purchases with known customers.

    Returned fields: invoice, date, customer_id, quantity, price, amount.
    Cancellation/returns are EXCLUDED, and not counted as negative spending.
    """
    cols = {c: ALIASES.get(str(c).strip().lower().replace(" ", ""), None)
            for c in frame.columns}
    # Handle legacy 'Customer ID' after removing spaces via the customerid alias.
    cols = {old: new for old, new in cols.items() if new}
    data = frame.rename(columns=cols).copy()
    if not set(REQUIRED).issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(set(REQUIRED)-set(data.columns))}")
    data["invoice"] = data["invoice"].astype("string").str.strip()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    cust = pd.to_numeric(data["customer_id"], errors="coerce")
    data["customer_id"] = cust.astype("Int64").astype("string")
    valid = (
        data["date"].notna() & cust.notna() &
        (data["quantity"] > 0) & (data["price"] > 0) &
        data["invoice"].notna() &
        ~data["invoice"].str.upper().str.startswith("C", na=False)
    )
    data = data.loc[valid, list(REQUIRED)].copy()
    data["amount"] = data["quantity"] * data["price"]
    return data.sort_values("date", kind="stable").reset_index(drop=True)

def _transform_log(raw: NDArray[np.float64]) -> NDArray[np.float64]:
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("RFM values must be finite and non-negative")
    return np.log1p(raw)

def monthly_rfm_batches(
    transactions: pd.DataFrame, *,
    history_days: int = 90,
    min_customers: int = 15,
) -> list[RetailBatch]:
    """Build monthly batches using ONLY purchases before each month-end cutoff.

    At each calendar month end:
    - score only customers active during that month;
    - compute R/F/M and two extra quantities from a trailing history window;
    - end the history *strictly* before the following month's start;
    - fit the scaler once on the FIRST eligible monthly batch and reuse it.

    This uses a calendar time axis, not the original notebook's row-count batches.
    """
    if history_days < 1 or min_customers < 2:
        raise ValueError("history_days must be >= 1 and min_customers >= 2")
    if transactions.empty:
        raise ValueError("No valid purchase transactions")
    data = transactions.sort_values("date", kind="stable").copy()
    months = pd.period_range(data["date"].min().to_period("M"),
                             data["date"].max().to_period("M"), freq="M")
    scaler = StandardScaler()
    batches: list[RetailBatch] = []
    for period in months:
        start = period.start_time
        cutoff = (period + 1).start_time
        current = data.loc[(data["date"] >= start) & (data["date"] < cutoff)]
        active = current["customer_id"].dropna().unique()
        if len(active) < min_customers:
            continue
        history = data.loc[
            (data["date"] >= cutoff - pd.Timedelta(days=history_days)) &
            (data["date"] < cutoff) & data["customer_id"].isin(active)
        ]
        grouped = history.groupby("customer_id", sort=True)
        stats = grouped.agg(
            last_date=("date", "max"),
            frequency=("invoice", "nunique"),
            monetary=("amount", "sum"),
            quantity=("quantity", "sum"),
        )
        stats["recency_days"] = (cutoff - stats["last_date"]).dt.total_seconds() / 86400
        stats["avg_invoice_value"] = stats["monetary"] / stats["frequency"]
        stats = stats.loc[:, list(FEATURE_NAMES)]
        raw = stats.to_numpy(dtype=np.float64, copy=True)
        logged = _transform_log(raw)
        if not batches:
            scaler.fit(logged)
        scaled = scaler.transform(logged).astype(np.float64)
        batches.append(RetailBatch(
            month=str(period), customer_ids=stats.index.to_numpy(),
            raw_features=raw, features=scaled, cutoff=cutoff,
        ))
    if not batches:
        raise ValueError("No eligible month; reduce min_customers or inspect source")
    return batches

def export_batches(batches: Iterable[RetailBatch], output_dir: str | Path) -> pd.DataFrame:
    """Export aligned features and customer IDs, with a reproducible manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch in batches:
        path = out / f"retail_{batch.month}.npz"
        np.savez_compressed(path, X=batch.features, raw_X=batch.raw_features,
                            customer_ids=batch.customer_ids.astype(str),
                            feature_names=np.asarray(FEATURE_NAMES))
        rows.append(dict(month=batch.month, cutoff=str(batch.cutoff),
                         n_customers=len(batch.customer_ids),
                         n_features=batch.features.shape[1], file=path.name))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)
    return manifest
