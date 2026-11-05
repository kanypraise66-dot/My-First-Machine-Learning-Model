from src.features import build_features
import sys
import json
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(path_in, path_out):
    meta = json.load(open(ROOT / "models/model_meta.json"))
    model = joblib.load(ROOT / "models/gbm_fraud.joblib")
    df = pd.read_csv(path_in)
    X = build_features(df)[meta["features"]]
    df["fraud_score"] = model.predict_proba(X)[:, 1]
    df["flagged"] = (df["fraud_score"] >= meta["threshold"]).astype(int)
    df.to_csv(path_out, index=False)
    print(f"{int(df['flagged'].sum()):,} of {len(df):,} flagged -> {path_out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
