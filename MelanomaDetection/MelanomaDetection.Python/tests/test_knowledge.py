import json

import pytest

import knowledge


def make_payload(flagged=(), diameter=True):
    payload = {
        "asymmetry": {"score": 0.34, "flagged": "asymmetry" in flagged},
        "border": {"score": 0.62, "flagged": "border" in flagged},
        "color": {
            "spread_high": False,
            "dangerous_color_detected": None,
            "dangerous_color_coverage_pct": 0,
            "flagged": "color" in flagged,
        },
        "evolution": {"flagged": False, "note": "not yet implemented -- no prior-image comparison available"},
    }
    if diameter:
        payload["diameter_mm"] = {"value": 8.4, "flagged": "diameter_mm" in flagged, "measured": True}
    return payload


ASYM_BORDER = make_payload(flagged=("asymmetry", "border"))


def statement(text, basis="analysis", ids=(), quote=None):
    return {"text": text, "basis": basis, "source_ids": list(ids), "quote": quote}


def valid_reply():
    return {
        "noticed": [
            statement("The analysis flagged asymmetry in this spot."),
            statement(
                "The booklet describes asymmetry as one half of the shape not matching the other.",
                "booklet",
                ["nci-abcd-asymmetry"],
                "The shape of one half does not match the other.",
            ),
        ],
        "next_steps": [
            statement(
                "See a dermatologist or other healthcare provider to have any change looked at.",
                "booklet",
                ["nci-report-changes"],
                "Changes in the skin or a mole should be reported to the doctor or nurse without delay.",
            ),
            statement(
                "Keep notes on this spot so you can notice changes.",
                "booklet",
                ["nci-self-exam-record"],
                "It may be helpful to record the dates of your skin exams",
            ),
        ],
    }


def check(reply, payload=ASYM_BORDER):
    passages = knowledge.select_passages(payload)
    return knowledge.parse_and_validate(json.dumps(reply), payload, passages)


class TestKnowledgeFile:
    def test_loads_with_unique_ids_and_real_pages(self):
        source, passages = knowledge.load()
        assert source.publication_number
        assert passages
        assert all(p.page > 0 and p.text.strip() for p in passages.values())

    def test_every_referenced_passage_exists(self):
        _, passages = knowledge.load()
        referenced = set(knowledge.CRITERION_PASSAGE.values()) | set(knowledge.GENERAL_PASSAGES)
        referenced.add("nci-abcd-variation")
        assert referenced <= set(passages)

    def test_a_passage_only_ever_names_its_own_feature(self):
        # This is what lets a flagged feature's passage be cited without ever
        # tripping the "mentioned an unflagged feature" check.
        _, passages = knowledge.load()
        owner = {passage_id: key for key, passage_id in knowledge.CRITERION_PASSAGE.items()}
        for passage in passages.values():
            for key, pattern in knowledge.UNFLAGGED_PATTERNS.items():
                if pattern.search(passage.text):
                    assert owner.get(passage.id) == key, f"{passage.id} names {key}"


class TestSelection:
    def test_only_flagged_features_get_a_passage(self):
        ids = {p.id for p in knowledge.select_passages(ASYM_BORDER)}
        assert {"nci-abcd-asymmetry", "nci-abcd-border", "nci-abcd-variation"} <= ids
        assert "nci-abcd-color" not in ids
        assert "nci-abcd-diameter" not in ids

    def test_general_passages_are_always_offered(self):
        for payload in (ASYM_BORDER, make_payload()):
            ids = {p.id for p in knowledge.select_passages(payload)}
            assert set(knowledge.GENERAL_PASSAGES) <= ids

    def test_nothing_flagged_offers_no_feature_passage(self):
        ids = {p.id for p in knowledge.select_passages(make_payload())}
        assert not ids & set(knowledge.CRITERION_PASSAGE.values())
        assert "nci-abcd-variation" not in ids


class TestValidation:
    def test_a_grounded_reply_passes(self):
        data, problems = check(valid_reply())
        assert problems == []
        assert data is not None

    def test_quote_matches_despite_typographic_dashes_and_case(self):
        reply = valid_reply()
        reply["noticed"][1]["quote"] = "asymmetry-the shape of one half does not match"
        _, problems = check(reply)
        assert problems == []

    def test_rejects_invalid_json(self):
        data, problems = knowledge.parse_and_validate("not json", ASYM_BORDER, knowledge.select_passages(ASYM_BORDER))
        assert data is None
        assert problems

    def test_rejects_wrong_top_level_keys(self):
        data, problems = check({"noticed": []})
        assert data is None
        assert problems

    def test_rejects_a_passage_that_was_not_provided(self):
        reply = valid_reply()
        reply["noticed"][1]["source_ids"] = ["nci-abcd-color"]
        _, problems = check(reply)
        assert any("only cite these ids" in p for p in problems)

    def test_rejects_a_quote_that_is_not_in_the_passage(self):
        reply = valid_reply()
        reply["noticed"][1]["quote"] = "Asymmetry means the mole is dangerous."
        _, problems = check(reply)
        assert any("verbatim" in p for p in problems)

    def test_rejects_a_booklet_statement_without_a_quote(self):
        reply = valid_reply()
        reply["noticed"][1]["quote"] = None
        _, problems = check(reply)
        assert problems

    def test_rejects_an_analysis_statement_that_cites_a_source(self):
        reply = valid_reply()
        reply["noticed"][0]["source_ids"] = ["nci-abcd-asymmetry"]
        _, problems = check(reply)
        assert problems

    def test_rejects_a_number_from_nowhere(self):
        reply = valid_reply()
        reply["noticed"][0]["text"] = "The analysis flagged asymmetry with a 73 percent score."
        _, problems = check(reply)
        assert any("number" in p for p in problems)

    def test_accepts_a_number_from_the_data(self):
        reply = valid_reply()
        reply["noticed"][0]["text"] = "The analysis flagged asymmetry (score 0.34)."
        _, problems = check(reply)
        assert problems == []

    @pytest.mark.parametrize(
        "text",
        [
            "You have melanoma.",
            "This is cancer.",
            "It is probably benign.",
            "Don't worry about this spot.",
            "This is an urgent situation.",
        ],
    )
    def test_rejects_diagnosis_reassurance_and_alarm(self, text):
        reply = valid_reply()
        reply["noticed"][0]["text"] = text
        _, problems = check(reply)
        assert problems

    def test_rejects_mentioning_a_feature_that_was_not_flagged(self):
        reply = valid_reply()
        reply["noticed"][0]["text"] = "The color of the spot also looks uneven."
        _, problems = check(reply)
        assert any("not flagged" in p for p in problems)

    def test_may_mention_diameter_only_when_flagged(self):
        reply = valid_reply()
        reply["noticed"][0]["text"] = "The spot is about 8.4 mm across."
        _, unflagged = check(reply, ASYM_BORDER)
        assert any("not flagged" in p for p in unflagged)

        flagged_payload = make_payload(flagged=("asymmetry", "border", "diameter_mm"))
        _, flagged = check(reply, flagged_payload)
        assert flagged == []

    def test_requires_a_recommendation_to_see_a_professional(self):
        reply = valid_reply()
        reply["next_steps"] = [
            statement("Keep notes on this spot.", "booklet", ["nci-self-exam-record"], "write notes about the way your skin looks"),
            statement("Protect your skin from the sun.", "booklet", ["nci-uv-midday"], "avoid exposure to the midday sun"),
        ]
        _, problems = check(reply)
        assert any("dermatologist" in p for p in problems)

    def test_enforces_item_counts(self):
        reply = valid_reply()
        reply["next_steps"] = reply["next_steps"][:1]
        _, problems = check(reply)
        assert any("next_steps" in p for p in problems)


class TestFallback:
    @pytest.mark.parametrize(
        "payload",
        [
            ASYM_BORDER,
            make_payload(),
            make_payload(flagged=("asymmetry", "border", "color", "diameter_mm")),
            make_payload(flagged=("color",), diameter=False),
            make_payload(diameter=False),
        ],
    )
    def test_fallback_passes_the_same_validator_as_a_model_reply(self, payload):
        passages = knowledge.select_passages(payload)
        data = knowledge.fallback_data(payload, passages)
        _, problems = knowledge.parse_and_validate(json.dumps(data), payload, passages)
        assert problems == []

    def test_fallback_says_size_could_not_be_measured_when_diameter_is_missing(self):
        payload = make_payload(diameter=False)
        data = knowledge.fallback_data(payload, knowledge.select_passages(payload))
        assert any("could not be measured" in item["text"] for item in data["noticed"])


class TestRender:
    def test_has_both_sections_and_cites_pages(self):
        source, _ = knowledge.load()
        passages = knowledge.select_passages(ASYM_BORDER)
        text = knowledge.render_explanation(valid_reply(), source, passages)
        assert text.startswith("What the analysis noticed:")
        assert "\nSuggested next steps:\n" in text
        assert "Sources:" in text
        assert "pages 6, 8, 40" in text
        assert source.caveat in text

    def test_no_sources_block_when_nothing_is_cited(self):
        source, _ = knowledge.load()
        reply = {
            "noticed": [statement("The analysis flagged asymmetry in this spot.")],
            "next_steps": [statement("See a dermatologist."), statement("Keep notes on this spot.")],
        }
        text = knowledge.render_explanation(reply, source, knowledge.select_passages(ASYM_BORDER))
        assert "Sources:" not in text
