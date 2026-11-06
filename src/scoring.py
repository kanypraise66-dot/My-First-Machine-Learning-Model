import numpy as np
from src.features import build_features


def score_events(df, model, meta):
    X = build_features(df)[meta["features"]]
    p = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["fraud_score"] = p
    out["decision"] = np.where(p >= meta["threshold"], "review", "allow")
    out["reason_codes"] = [
        ",".join(c for c, hit in [
            ("ACCOUNT_DRAINED", b > 0 and a >= 0.999 * b),
            ("RISKY_TYPE", t in ("TRANSFER", "CASH_OUT")),
            ("EMPTY_DESTINATION", d == 0)] if hit)
        for a, b, t, d in zip(df["amount"], df["oldbalanceOrg"], df["type"], df["oldbalanceDest"])
    ]
    out["model_version"] = "gbm_fraud_v1"
    return out
