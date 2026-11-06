import pandas as pd
import pytest

from src.pipeline import load_artifacts, run_batch

ROW = dict(step=1, type="TRANSFER", amount=500.0, nameOrig="C1", oldbalanceOrg=500.0,
           newbalanceOrig=0.0, nameDest="C2", oldbalanceDest=0.0, newbalanceDest=0.0)


@pytest.fixture(scope="module")
def art():
    return load_artifacts()


def test_single_row(art):
    scored, rej = run_batch(pd.DataFrame([ROW]), *art)
    assert len(scored) == 1 and 0 <= scored.fraud_score.iloc[0] <= 1


def test_unknown_type_rejected(art):
    scored, rej = run_batch(pd.DataFrame([{**ROW, "type": "AIRTIME"}]), *art)
    assert len(rej) == 1 and len(scored) == 0


def test_missing_column_raises(art):
    with pytest.raises(ValueError):
        run_batch(pd.DataFrame([ROW]).drop(columns=["amount"]), *art)


def test_label_column_ignored(art):
    scored, _ = run_batch(pd.DataFrame([{**ROW, "isFraud": 1}]), *art)
    assert "fraud_score" in scored


def test_empty_account_not_flagged_as_drained(art):
    scored, _ = run_batch(pd.DataFrame(
        [{**ROW, "oldbalanceOrg": 0.0, "amount": 50.0}]), *art)
    assert "ACCOUNT_DRAINED" not in scored.reason_codes.iloc[0]
