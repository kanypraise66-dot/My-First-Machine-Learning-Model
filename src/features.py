import pandas as pd

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-transaction features only (no newbalance* columns)."""
    feat = pd.DataFrame(index=df.index)
    feat["amount"] = df["amount"]
    feat["oldbalanceOrg"] = df["oldbalanceOrg"]
    feat["oldbalanceDest"] = df["oldbalanceDest"]
    feat["hour"] = df["step"] % 24
    feat["amount_over_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    feat["dest_old_zero"] = (df["oldbalanceDest"] == 0).astype("int8")
    for t in TYPES:
        feat[f"type_{t}"] = (df["type"] == t).astype("int8")
    return feat.astype("float32")
