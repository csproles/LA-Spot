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
