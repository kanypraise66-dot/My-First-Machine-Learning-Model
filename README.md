# Fraud Detection on PaySim

Gradient boosting model that flags fraudulent mobile-money transactions using only information available before the transaction completes. Main result: PR-AUC 0.805 (logistic baseline: 0.204). See Results and Limitations below.

## How to run
1. Place `PS_log.csv` in `Data/raw/` (not committed: file is too large).
2. `pip install -r requirements.txt`
3. `python -m src.train` — trains both models, saves them to `models/`, metrics to `results/metrics/`
4. `python scripts/predict.py input.csv scored.csv` — scores new transactions
5. `pytest` — runs the feature tests

## Results

**Dataset:** PaySim mobile-money simulation (~6.36M transactions, fraud rate ~0.13%). Synthetic data, so real-world performance will likely be lower.

**Setup:** Stratified 75/25 random split (`random_state=0`), no post-transaction features (`USE_POST_BALANCES = False`). Main metric: PR-AUC (a model with no skill scores ~0.0013).

| Model | PR-AUC |
|---|---|
| Logistic regression | 0.204 |
| Gradient boosting | 0.805 |

**Leakage check:** using post-transaction balances raised logistic PR-AUC to 0.569 (vs ~0.2 without), so they are excluded from the main model.

**Operating points (gradient boosting, test set of 2,053 frauds):**

| Threshold | Precision | Recall | Flagged | False alarms |
|---|---|---|---|---|
| 0.90 | 0.822 | 0.821 | 2,050 | 364 |
| 0.50 | 0.700 | 0.848 | 2,485 | 745 |
| 0.10 | 0.493 | 0.933 | 3,888 | 1,973 |

Confusion matrix at 0.90: TN 1,588,238 / FP 364 / FN 367 / TP 1,686. The 0.90 threshold was chosen on the test set, so these figures are slightly optimistic.

**Drivers:** the model relies mostly on `amount_over_balance`, `amount`, and transaction type (TRANSFER, CASH_OUT); `hour` contributes little.

### Limitations
- Recall plateaus near 93%. About 135 test frauds (6.6%) score near zero; most are CASH_OUT transactions that drain the origin into a destination with an existing balance. About 430,000 legitimate transactions share that profile (0.17% fraud), so with current features they are largely indistinguishable.
- A time-based split (train on the first 80% of steps) gave PR-AUC 1.000. I do not treat this as reliable: no post-transaction features are present (checked), but the late test period has a different fraud rate (0.34%), and the likely explanation (the period is easier to separate) is unverified.
- The cost-optimal threshold (0.066) depends on an assumed review cost.
- Data is synthetic.