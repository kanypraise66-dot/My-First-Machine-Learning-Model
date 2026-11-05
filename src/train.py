import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.features import build_features

ROOT = Path(__file__).resolve().parents[1]
USECOLS = ["step", "type", "amount",
           "oldbalanceOrg", "oldbalanceDest", "isFraud"]


def main():
    df = pd.read_csv(ROOT / "Data/raw/PS_log.csv", usecols=USECOLS)
    X, y = build_features(df), df["isFraud"].astype("int8")
    del df
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)

    logit = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000, class_weight="balanced"))
    logit.fit(X_tr, y_tr)
    gb = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, random_state=0)
    gb.fit(X_tr, y_tr)

    metrics = {"logistic_pr_auc": float(average_precision_score(y_te, logit.predict_proba(X_te)[:, 1])),
               "gbm_pr_auc": float(average_precision_score(y_te, gb.predict_proba(X_te)[:, 1]))}
    print(metrics)

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results/metrics").mkdir(parents=True, exist_ok=True)
    (ROOT / "Data/processed").mkdir(parents=True, exist_ok=True)
    joblib.dump(gb, ROOT / "models/gbm_fraud.joblib")
    joblib.dump(logit, ROOT / "models/logit_baseline.joblib")
    json.dump({"features": list(X.columns), "threshold": 0.90, "use_post_balances": False,
               "split": "random stratified 75/25, random_state=0"},
              open(ROOT / "models/model_meta.json", "w"), indent=2)
    json.dump(metrics, open(ROOT / "results/metrics/metrics.json", "w"), indent=2)
    X_te.assign(isFraud=y_te.values).to_csv(
        ROOT / "Data/processed/test_set.csv", index=False)


if __name__ == "__main__":
    main()
