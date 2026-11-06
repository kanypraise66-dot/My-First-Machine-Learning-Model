import json
from pathlib import Path

import joblib
import pandas as pd

from src.scoring import score_events

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = ["step", "type", "amount", "nameOrig", "oldbalanceOrg",
            "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"]
NUMERIC = ["step", "amount", "oldbalanceOrg", "newbalanceOrig",
           "oldbalanceDest", "newbalanceDest"]
VALID_TYPES = {"PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"}


def load_artifacts(model_path=ROOT / "models" / "gbm_fraud.joblib",
                   meta_path=ROOT / "models" / "model_meta.json"):
    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta


def validate(df: pd.DataFrame):
    """Split incoming data into (clean, rejected) so one bad row doesn't kill a batch."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.drop(columns=["isFraud", "isFlaggedFraud"], errors="ignore").copy()
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bad = df[NUMERIC].isna().any(axis=1) | ~df["type"].isin(
        VALID_TYPES) | (df["amount"] < 0)
    return df[~bad].reset_index(drop=True), df[bad].reset_index(drop=True)


def run_batch(df, model, meta):
    clean, rejected = validate(df)
    scored = score_events(clean, model, meta) if len(clean) else clean
    return scored, rejected
