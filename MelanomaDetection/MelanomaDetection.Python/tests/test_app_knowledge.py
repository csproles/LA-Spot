"""app_knowledge.py's generated passages must never drift from policy.py's own
constants -- that's the entire point of generating them instead of typing the
numbers out by hand. See app_knowledge.py's module docstring."""

import app_knowledge
import policy


def _text_by_id(passage_id):
    return next(p["text"] for p in app_knowledge.passages() if p["id"] == passage_id)


def test_risk_band_passage_matches_policy_thresholds():
    text = _text_by_id("app-risk-bands")
    assert f"{policy.BAND_LOW_MAX:g}" in text
    assert f"{policy.BAND_MODERATE_MAX:g}" in text


def test_cadence_passage_matches_policy_cadence():
    text = _text_by_id("app-recheck-cadence")
    assert str(policy.CADENCE_DAYS["low"]) in text
    assert str(policy.CADENCE_DAYS["moderate"]) in text
    assert str(policy.TIGHTENED_CADENCE_DAYS["low"]) in text
    assert str(policy.TIGHTENED_CADENCE_DAYS["moderate"]) in text


def test_passage_ids_are_unique():
    ids = [p["id"] for p in app_knowledge.passages()]
    assert len(ids) == len(set(ids))
