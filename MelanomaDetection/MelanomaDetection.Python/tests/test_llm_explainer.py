import json
from types import SimpleNamespace

import pytest

import llm_explainer

ABCDE = {
    "asymmetry": {"score": 4.0, "details": {"raw_asymmetry_ratio": 0.34, "concern": True}},
    "border": {"score": 5.0, "details": {"raw_border_irregularity": 0.62, "concern": True}},
    "color": {"score": 2.0, "details": {"color_cv": 0.1, "concern": False, "dangerous_colors_pct": {}}},
    "diameter": {"score": 1.0, "details": {"diameter_mm": 8.4, "concern": False}},
}

VALID_REPLY = json.dumps(
    {
        "noticed": [
            {"text": "The analysis flagged asymmetry in this spot.", "basis": "analysis", "source_ids": [], "quote": None},
            {
                "text": "The booklet describes asymmetry as one half of the shape not matching the other.",
                "basis": "booklet",
                "source_ids": ["nci-abcd-asymmetry"],
                "quote": "The shape of one half does not match the other.",
            },
        ],
        "next_steps": [
            {
                "text": "See a dermatologist or other healthcare provider to have any change looked at.",
                "basis": "booklet",
                "source_ids": ["nci-report-changes"],
                "quote": "Changes in the skin or a mole should be reported to the doctor or nurse without delay.",
            },
            {
                "text": "Keep notes on this spot so you can notice changes.",
                "basis": "booklet",
                "source_ids": ["nci-self-exam-record"],
                "quote": "It may be helpful to record the dates of your skin exams",
            },
        ],
    }
)

INVALID_REPLY = json.dumps({"noticed": [], "next_steps": []})


class FakeClient:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def user_message(call):
    return next(m["content"] for m in call["messages"] if m["role"] == "user" and "ANALYSIS DATA" in m["content"])


def test_valid_reply_is_rendered_with_sources():
    client = FakeClient([VALID_REPLY])
    text = llm_explainer.explain_findings(ABCDE, client=client)

    assert len(client.calls) == 1
    assert text.startswith("What the analysis noticed:")
    assert "Suggested next steps:" in text
    assert "Sources:" in text


def test_request_is_deterministic_json_and_carries_the_grounding_rules():
    client = FakeClient([VALID_REPLY])
    llm_explainer.explain_findings(ABCDE, client=client)
    call = client.calls[0]

    assert call["temperature"] == 0
    assert call["reasoning_effort"] == "none"
    assert call["max_completion_tokens"] == 900
    assert "max_tokens" not in call
    assert call["response_format"] == {"type": "json_object"}
    system_text = " ".join(m["content"] for m in call["messages"] if m["role"] == "system")
    assert llm_explainer.SYSTEM_PROMPT in system_text
    assert "GROUNDING RULES" in system_text


def test_only_passages_for_flagged_features_reach_the_model():
    client = FakeClient([VALID_REPLY])
    llm_explainer.explain_findings(ABCDE, client=client)
    content = user_message(client.calls[0])

    assert "nci-abcd-asymmetry" in content
    assert "nci-abcd-border" in content
    assert "nci-abcd-color" not in content
    assert "nci-abcd-diameter" not in content


def test_a_rejected_reply_is_retried_once_with_the_reasons():
    client = FakeClient([INVALID_REPLY, VALID_REPLY])
    text = llm_explainer.explain_findings(ABCDE, client=client)

    assert len(client.calls) == 2
    retry_messages = client.calls[1]["messages"]
    assert retry_messages[-1]["role"] == "user"
    assert "rejected" in retry_messages[-1]["content"]
    assert text.startswith("What the analysis noticed:")


def test_two_rejected_replies_fall_back_to_the_passages():
    client = FakeClient([INVALID_REPLY, "not json at all"])
    text = llm_explainer.explain_findings(ABCDE, client=client)

    assert len(client.calls) == llm_explainer.MAX_ATTEMPTS
    assert "The shape of one half does not match the other." in text
    assert "Sources:" in text


def test_api_failures_are_not_swallowed():
    client = FakeClient([RuntimeError("no api key")])
    with pytest.raises(RuntimeError):
        llm_explainer.explain_findings(ABCDE, client=client)


ONE_STEP_REPLY = json.dumps(
    {
        "noticed": [
            {"text": "The analysis flagged asymmetry in this spot.", "basis": "analysis", "source_ids": [], "quote": None},
        ],
        "next_steps": [
            {
                "text": "Keep notes on this spot so you can notice changes.",
                "basis": "booklet",
                "source_ids": ["nci-self-exam-record"],
                "quote": "It may be helpful to record the dates of your skin exams",
            },
        ],
    }
)

WRONG_BAND_REPLY = json.dumps(
    {
        "noticed": [
            {"text": "This came back high risk.", "basis": "analysis", "source_ids": [], "quote": None},
        ],
        "next_steps": [
            {
                "text": "See a dermatologist or other healthcare provider to have this looked at.",
                "basis": "booklet",
                "source_ids": ["nci-report-changes"],
                "quote": "Changes in the skin or a mole should be reported to the doctor or nurse without delay.",
            },
        ],
    }
)


class TestOverallResult:
    def test_summary_line_and_first_step_come_from_policy_not_the_model(self):
        client = FakeClient([ONE_STEP_REPLY])
        text = llm_explainer.explain_findings(ABCDE, risk_score=20.0, client=client)

        assert text.startswith("In short:\nThis check came back low risk signs (20/100).")
        assert "Take a new photo of this spot in about 90 days" in text
        assert "Keep notes on this spot" in text

    def test_high_risk_gets_no_recheck_reminder_since_the_advice_is_to_see_a_doctor_now(self):
        client = FakeClient([ONE_STEP_REPLY])
        text = llm_explainer.explain_findings(ABCDE, risk_score=80.0, client=client)

        assert "high risk signs" in text
        assert "Take a new photo" not in text

    def test_risk_factors_shorten_the_recheck_window_in_the_rendered_text(self):
        client = FakeClient([ONE_STEP_REPLY])
        text = llm_explainer.explain_findings(ABCDE, risk_score=20.0, profile={"familyHistory": True}, client=client)

        assert "about 60 days" in text

    def test_overall_reaches_the_model_and_shrinks_the_next_steps_range(self):
        client = FakeClient([ONE_STEP_REPLY])
        llm_explainer.explain_findings(ABCDE, risk_score=45.0, client=client)
        content = user_message(client.calls[0])

        assert '"overall"' in content
        assert '"band": "moderate"' in content

    def test_a_reply_naming_the_wrong_band_is_rejected_and_retried(self):
        client = FakeClient([WRONG_BAND_REPLY, ONE_STEP_REPLY])
        text = llm_explainer.explain_findings(ABCDE, risk_score=20.0, client=client)

        assert len(client.calls) == 2
        assert "high risk" not in client.calls[1]["messages"][-1]["content"] or "rejected" in client.calls[1]["messages"][-1]["content"]
        assert text.startswith("In short:\nThis check came back low risk signs")

    def test_without_a_risk_score_behaviour_is_unchanged(self):
        client = FakeClient([VALID_REPLY])
        text = llm_explainer.explain_findings(ABCDE, client=client)

        assert "In short:" not in text
        # The prompt's own instructions mention the word "overall" (see rule 4c), so check
        # for the JSON key specifically, not just the word appearing anywhere in the prompt.
        assert '"overall": {' not in user_message(client.calls[0])


class TestChangeAndSymptoms:
    def test_change_data_reaches_the_model_when_supplied(self):
        client = FakeClient([VALID_REPLY])
        evolving = {"score": 6.0, "details": {"signals": ["growth"], "area_growth_ratio": 0.22}}
        llm_explainer.explain_findings(ABCDE, evolving=evolving, client=client)
        content = user_message(client.calls[0])

        assert '"change_since_last_photo"' in content
        assert '"available": true' in content
        assert '"size"' in content

    def test_no_prior_check_is_marked_unavailable_not_omitted(self):
        client = FakeClient([VALID_REPLY])
        evolving = {"score": None, "details": {"reason": "no prior check to compare against"}}
        llm_explainer.explain_findings(ABCDE, evolving=evolving, client=client)
        content = user_message(client.calls[0])

        assert '"available": false' in content

    def test_only_recognised_symptom_choices_reach_the_model(self):
        client = FakeClient([VALID_REPLY])
        llm_explainer.explain_findings(ABCDE, symptoms=["Itchy", "some free-text the model should never see"], client=client)
        content = user_message(client.calls[0])

        assert '"Itchy"' in content
        assert "free-text" not in content

    def test_no_symptoms_key_when_none_are_reported(self):
        client = FakeClient([VALID_REPLY])
        llm_explainer.explain_findings(ABCDE, symptoms=[], client=client)
        # Likewise: the prompt's instructions name "symptoms_reported" regardless of input.
        assert '"symptoms_reported":' not in user_message(client.calls[0])
