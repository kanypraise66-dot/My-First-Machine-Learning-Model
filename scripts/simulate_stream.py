from src.pipeline import load_artifacts, run_batch
import pandas as pd
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


model, meta = load_artifacts()
raw = pd.read_csv(ROOT / "Data" / "raw" / "PS_log.csv")

# Fraud is ~0.1% of PaySim, so mix in known fraud or you'll see almost nothing
sample = pd.concat([raw[raw.isFraud == 0].sample(5000, random_state=1),
                    raw[raw.isFraud == 1].sample(100, random_state=1)]
                   ).sample(frac=1, random_state=1).reset_index(drop=True)

BATCH = 500
caught = flagged = 0
for i in range(0, len(sample), BATCH):
    chunk = sample.iloc[i:i + BATCH]
    t0 = time.perf_counter()
    scored, rejected = run_batch(chunk, model, meta)
    ms = (time.perf_counter() - t0) * 1000
    review = scored["decision"] == "review"
    flagged += int(review.sum())
    # labels are only used here, after scoring, to check the result
    caught += int(chunk.loc[chunk.index.isin(chunk.index[:len(scored)]),
                  "isFraud"].values[review.values].sum())
    print(f"batch {i // BATCH}: {len(scored)} scored, {len(rejected)} rejected, "
          f"{int(review.sum())} -> review, {ms:.0f} ms")

print(
    f"total flagged: {flagged}, of which real fraud: {caught} (out of 100 fraud rows)")
