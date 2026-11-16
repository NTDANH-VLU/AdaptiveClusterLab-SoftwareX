"""Stream the original UCI Electricity Load Diagrams into monthly 34-feature batches.

Original file: LD2011_2014.txt (semicolon separated; comma decimals).
The 370 client columns hold kW observations every 15 minutes. This adapter
does not claim that its 34 engineered features are supplied by UCI.

Features: 24 hourly energy shares, 7 weekday energy shares, log1p(monthly
energy in kWh), log1p(peak kW), and monthly load factor (mean/peak).
One scaler is fitted on the FIRST DEVELOPMENT MONTH only. Like the Retail
adapter, this is unsupervised first-batch / transductive normalization, not
a per-seed train-only scaler. No future month enters scaler fitting.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = tuple(
    [f"hour_share_{h:02d}" for h in range(24)] +
    [f"weekday_share_{d}" for d in range(7)] +
    ["log_monthly_kwh", "log_peak_kw", "load_factor"]
)


def _flush(month, totals, hourly, weekday, peak, count, client_ids):
    """Return unscaled features of all consumers with nonzero monthly energy."""
    active = totals > 0
    if not np.any(active):
        raise ValueError(f"No consuming clients in month {month}")
    t = totals[active]
    features = np.column_stack((
        hourly[:, active].T / t[:, None],
        weekday[:, active].T / t[:, None],
        np.log1p(t),
        np.log1p(peak[active]),
        (totals[active] * 4.0 / count) / np.maximum(peak[active], 1e-12),
    ))
    if features.shape[1] != 34 or not np.isfinite(features).all():
        raise ValueError(f"Invalid 34D engineered matrix for {month}")
    return client_ids[active], features.astype(np.float64)


def monthly_electricity_batches(
    source: str | Path,
    *,
    start: str = "2012-01",
    months: int = 24,
    chunksize: int = 2048,
):
    """Read chunks, keeping RAM bounded; yield (month, IDs, raw features).

    Requires sequential timestamps. The source timestamps are used as stated;
    DST-specific source conventions are not reconstructed or corrected.
    """
    if months < 2 or chunksize < 1:
        raise ValueError("months must be >=2 and chunksize >=1")
    start_period = pd.Period(start, freq="M")
    end_period = start_period + months
    reader = pd.read_csv(
        source, sep=";", decimal=",", chunksize=chunksize,
        index_col=False, low_memory=False,
    )
    first = True
    current = None
    total = hourly = weekday = peak = None
    count = 0
    emitted = 0
    previous = None
    clients = None

    for frame in reader:
        if first:
            if frame.shape[1] < 3:
                raise ValueError("Expected timestamp and at least two client columns")
            clients = np.asarray([str(c) for c in frame.columns[1:]], dtype=str)
            if len(set(clients)) != len(clients):
                raise ValueError("Client IDs must be unique")
            first = False
        stamps = pd.to_datetime(frame.iloc[:, 0], format="%Y-%m-%d %H:%M:%S", errors="raise")
        values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
        if values.shape[1] != len(clients) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Expected nonnegative finite kW readings for consistent clients")
        for i, stamp in enumerate(stamps):
            if previous is not None and stamp < previous:
                raise ValueError("Electricity timestamps are not in chronological order")
            previous = stamp
            period = stamp.to_period("M")
            if period < start_period:
                continue
            if period >= end_period:
                break
            if current is None:
                current = period
                total = np.zeros(len(clients))
                hourly = np.zeros((24, len(clients)))
                weekday = np.zeros((7, len(clients)))
                peak = np.zeros(len(clients))
            if period != current:
                ids, features = _flush(current, total, hourly, weekday, peak, count, clients)
                yield str(current), ids, features
                emitted += 1
                if period != current + 1:
                    raise ValueError(f"Missing calendar month after {current}")
                current = period
                total.fill(0)
                hourly.fill(0)
                weekday.fill(0)
                peak.fill(0)
                count = 0
            kw = values[i]
            kwh = kw / 4.0
            total += kwh
            hourly[stamp.hour] += kwh
            weekday[stamp.dayofweek] += kwh
            np.maximum(peak, kw, out=peak)
            count += 1
        if previous is not None and previous.to_period("M") >= end_period:
            break
    if current is not None:
        ids, features = _flush(current, total, hourly, weekday, peak, count, clients)
        yield str(current), ids, features
        emitted += 1
    if emitted != months:
        raise ValueError(f"Requested {months} monthly batches; found {emitted}")


def export_electricity_batches(
    source: str | Path,
    output: str | Path,
    *,
    start: str = "2012-01",
    months: int = 24,
    development_months: int = 12,
    chunksize: int = 2048,
):
    if not 1 < development_months < months:
        raise ValueError("development_months must be between 2 and months-1")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    manifest = []
    for index, (month, ids, raw) in enumerate(
        monthly_electricity_batches(source, start=start, months=months, chunksize=chunksize)
    ):
        if index == 0:
            scaler.fit(raw)
        x = scaler.transform(raw).astype(np.float32)
        path = output / f"electricity_{month}.npz"
        np.savez_compressed(
            path, X=x, raw_X=raw, customer_ids=ids,
            feature_names=np.asarray(FEATURE_NAMES, dtype=str),
        )
        manifest.append(dict(
            month=month, file=path.name, clients=len(ids),
            features=x.shape[1],
            phase="development" if index < development_months else "evaluation",
        ))
    pd.DataFrame(manifest).to_csv(output / "manifest.csv", index=False)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/electricity/batches")
    parser.add_argument("--start", default="2012-01")
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--development-months", type=int, default=12)
    parser.add_argument("--chunksize", type=int, default=2048)
    a = parser.parse_args()
    manifest = export_electricity_batches(
        a.input, a.output, start=a.start, months=a.months,
        development_months=a.development_months, chunksize=a.chunksize,
    )
    print(f"Prepared {len(manifest)} electricity monthly batches in {a.output}")


if __name__ == "__main__":
    main()
