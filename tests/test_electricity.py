"""Offline synthetic fixture tests for the UCI Electricity adapter.

These tests check formatting and temporal engineering; they are not evidence
of measured performance on the full UCI source.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scripts.prepare_electricity import (
    FEATURE_NAMES, monthly_electricity_batches, export_electricity_batches,
)


def sample_txt(path: Path):
    # Three 15-minute readings per month, 90 clients, comma decimal format.
    dates = ["2012-01-03 00:00:00", "2012-01-03 00:15:00",
             "2012-02-02 01:00:00", "2012-02-02 01:15:00",
             "2012-03-02 02:00:00", "2012-03-02 02:15:00"]
    ids = [f"MT_{i:03d}" for i in range(90)]
    with path.open("w", encoding="utf-8") as f:
        f.write(";".join([""] + ids) + "\n")
        for index, date in enumerate(dates):
            values = [str(1 + (i + index) % 9 / 10).replace(".", ",") for i in range(90)]
            f.write(";".join([date] + values) + "\n")
    return ids


def test_streamed_electricity_34_features_and_no_future_scaler(tmp_path):
    source = tmp_path / "LD2011_2014.txt"
    ids = sample_txt(source)
    batches = list(monthly_electricity_batches(source, start="2012-01", months=3, chunksize=2))
    assert len(FEATURE_NAMES) == 34
    assert [m for m, _, _ in batches] == ["2012-01", "2012-02", "2012-03"]
    assert all(x.shape == (90, 34) for _, _, x in batches)
    assert all(np.isfinite(x).all() for _, _, x in batches)
    assert all(list(client_ids) == ids for _, client_ids, _ in batches)
    assert np.allclose(batches[0][2][:, :24].sum(axis=1), 1)
    assert np.allclose(batches[0][2][:, 24:31].sum(axis=1), 1)
    out = tmp_path / "batches"
    manifest = export_electricity_batches(source, out, months=3,
                                          development_months=2, chunksize=2)
    assert len(manifest) == 3
    assert manifest[2]["phase"] == "evaluation"
    with np.load(out / "electricity_2012-03.npz", allow_pickle=False) as data:
        assert data["X"].shape == (90, 34)
        assert len(data["feature_names"]) == 34


def test_reject_missing_month(tmp_path):
    source = tmp_path / "LD2011_2014.txt"
    sample_txt(source)
    rows = source.read_text().splitlines()
    source.write_text("\n".join([row for row in rows if "2012-02" not in row]) + "\n")
    with pytest.raises(ValueError, match="Missing calendar month"):
        list(monthly_electricity_batches(source, start="2012-01", months=3, chunksize=3))


def test_reject_negative_energy(tmp_path):
    source = tmp_path / "LD2011_2014.txt"
    sample_txt(source)
    content = source.read_text()
    source.write_text(content.replace("1,0", "-1,0", 1))
    with pytest.raises(ValueError, match="nonnegative"):
        list(monthly_electricity_batches(source, start="2012-01", months=3, chunksize=2))
