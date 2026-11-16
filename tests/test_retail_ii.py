import numpy as np
import pandas as pd
import pytest
from adaptiveclusterlab.data.retail_ii import clean_transactions, monthly_rfm_batches, export_batches

def small_transactions():
    return pd.DataFrame({
        "Invoice": ["1","2","C3","4","5","6","7","8","9"],
        "InvoiceDate": [
            "2010-01-02","2010-01-03","2010-01-03","2010-02-01",
            "2010-02-03","2010-02-05","2010-01-11","2010-02-09","2010-02-10",
        ],
        "Customer ID": [1,2,1,1,2,3,3,4,4],
        "Quantity": [2,1,-1,1,1,1,2,1,1],
        "Price": [10,20,10,15,9,8,5,5,5],
    })

def test_cancellation_and_negative_quantity_excluded():
    x = clean_transactions(small_transactions())
    assert len(x) == 8
    assert "C3" not in x["invoice"].to_list()
    assert x["amount"].sum() == pytest.approx(92.0)

def test_future_transactions_cannot_change_past_month():
    source = clean_transactions(small_transactions())
    first = monthly_rfm_batches(source, history_days=90, min_customers=2)[0]
    future = pd.DataFrame([{
        "invoice":"100","date":pd.Timestamp("2010-03-10"),
        "customer_id":"1","quantity":100,"price":1000,"amount":100000,
    }])
    altered = pd.concat([source, future], ignore_index=True)
    still_first = monthly_rfm_batches(altered, history_days=90, min_customers=2)[0]
    np.testing.assert_allclose(first.raw_features, still_first.raw_features)
    np.testing.assert_allclose(first.features, still_first.features)

def test_order_and_reproducibility_and_export(tmp_path):
    tx = clean_transactions(small_transactions())
    a = monthly_rfm_batches(tx, history_days=90, min_customers=2)
    b = monthly_rfm_batches(tx.sample(frac=1, random_state=1),
                            history_days=90, min_customers=2)
    assert [t.month for t in a] == ["2010-01", "2010-02"]
    for x,y in zip(a,b):
        np.testing.assert_allclose(x.features, y.features)
    manifest = export_batches(a, tmp_path)
    assert len(manifest) == 2
    arr = np.load(tmp_path / "retail_2010-02.npz")
    assert arr["X"].shape[1] == 5
