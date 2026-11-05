import pandas as pd
from src.features import build_features


def test_features():
    df = pd.DataFrame({"step": [25], "type": ["CASH_OUT"], "amount": [100.0],
                       "oldbalanceOrg": [100.0], "oldbalanceDest": [0.0]})
    f = build_features(df)
    assert f.loc[0, "hour"] == 1
    assert f.loc[0, "type_CASH_OUT"] == 1 and f.loc[0, "type_TRANSFER"] == 0
    assert f.loc[0, "dest_old_zero"] == 1
    assert abs(f.loc[0, "amount_over_balance"] - 100 / 101) < 1e-4
    assert not any(c.startswith("newbalance") for c in f.columns)
