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


def statement(text, basis="analysis", ids=(), quote=None, letter=None):
    item = {"text": text, "basis": basis, "source_ids": list(ids), "quote": quote}
    if letter is not None:
        item["letter"] = letter
    return item


def lettered_a():
    return statement(
        "We noticed the two halves of this spot look different, which means the shape of one half does not match the other.",
        "booklet", ["nci-abcd-asymmetry"], "The shape of one half does not match the other.", "A",
    )


def lettered_b():
    return statement(
        "We noticed the edge of this spot looks uneven, which means the edges are often ragged, notched, blurred, or irregular.",
        "booklet", ["nci-abcd-border"], "The edges are often ragged, notched, blurred, or irregular in outline", "B",
    )


def valid_reply():
    return {
        "noticed": [lettered_a(), lettered_b()],
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
        reply["noticed"][0]["quote"] = "asymmetry-the shape of one half does not match"
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
        reply["noticed"].insert(0, statement("You reported these about the spot: itchy.", "analysis", ["nci-abcd-asymmetry"]))
        _, problems = check(reply)
        assert any("analysis" in p and "source_ids" in p for p in problems)

    def test_rejects_a_number_from_nowhere(self):
        reply = valid_reply()
        reply["noticed"][0]["text"] = "The analysis flagged asymmetry with a 73 percent score."
        _, problems = check(reply)
        assert any("number" in p for p in problems)

    def test_accepts_a_number_from_the_data(self):
        reply = valid_reply()
        reply["noticed"][0]["text"] = "We noticed asymmetry (score 0.34), which means the shape of one half does not match the other."
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
        reply["noticed"][0]["text"] = "We noticed the spot is about 8.4 mm across, which means the shape of one half does not match the other."
        _, unflagged = check(reply, ASYM_BORDER)
        assert any("not flagged" in p for p in unflagged)

        flagged_payload = make_payload(flagged=("asymmetry", "border", "diameter_mm"))
        _, flagged = check(reply, flagged_payload)
        assert not any("not flagged" in p for p in flagged)

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
        size = [item for item in data["noticed"] if item.get("letter") == "D"]
        assert len(size) == 1
        assert "cannot measure the size" in size[0]["text"]


class TestDashes:
    @pytest.mark.parametrize("text", [
        "We noticed asymmetry — the shape is uneven, which means one half does not match.",
        "We noticed asymmetry – the shape is uneven, which means one half does not match.",
        "We noticed asymmetry -- the shape is uneven, which means one half does not match.",
    ])
    def test_a_dash_is_rejected(self, text):
        reply = valid_reply()
        reply["noticed"][0]["text"] = text
        _, problems = knowledge.parse_and_validate(json.dumps(reply), ASYM_BORDER, knowledge.select_passages(ASYM_BORDER))
        assert any("dash" in problem for problem in problems)

    def test_a_hyphen_is_fine(self):
        reply = valid_reply()
        reply["noticed"][0]["text"] = "We noticed asymmetry in this well-defined spot, which means the shape of one half does not match the other."
        data, problems = knowledge.parse_and_validate(json.dumps(reply), ASYM_BORDER, knowledge.select_passages(ASYM_BORDER))
        assert problems == []

    def test_a_dash_inside_a_quotation_is_left_alone(self):
        reply = valid_reply()
        reply["noticed"][0]["text"] = 'We noticed asymmetry, which means the booklet says "one half — the other half" of the shape.'
        _, problems = knowledge.parse_and_validate(json.dumps(reply), ASYM_BORDER, knowledge.select_passages(ASYM_BORDER))
        assert not any("dash" in problem for problem in problems)


class TestRender:
    def test_every_point_is_a_bullet(self):
        source, _ = knowledge.load()
        passages = knowledge.select_passages(ASYM_BORDER)
        text = knowledge.render_explanation(
            valid_reply(), source, passages, lead="Lower visual concern. See a doctor.", opening_steps=("Recheck in a month.",)
        )
        head = text.split("\n\nSources:")[0]
        headings = {"In short:", "What the analysis noticed:", "Suggested next steps:", "A: Asymmetry (shape)", "B: Border (edges)"}
        points = [line for line in head.splitlines() if line and line not in headings]
        # the lead, 2 noticed, the opening step, 2 next steps
        assert len(points) == 6
        assert all(line.startswith("• ") for line in points)
        assert "• Lower visual concern. See a doctor." in text
        assert "• Recheck in a month." in text

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


class TestLetteredSections:
    def test_a_reply_missing_a_flagged_letter_is_rejected(self):
        reply = valid_reply()
        reply["noticed"] = [reply["noticed"][0]]
        _, problems = check(reply)
        assert any("missing a lettered item for B" in p for p in problems)

    def test_a_letter_for_an_unflagged_feature_is_rejected(self):
        reply = valid_reply()
        reply["noticed"].append(
            statement("We noticed the color is uneven, which means it varies.", "booklet", ["nci-abcd-asymmetry"], "The shape of one half does not match the other.", "C")
        )
        _, problems = check(reply)
        assert any("letter C" in p and "does not apply" in p for p in problems)

    def test_letters_must_be_in_order_and_unique(self):
        reply = valid_reply()
        reply["noticed"] = [reply["noticed"][1], reply["noticed"][0]]
        _, out_of_order = check(reply)
        assert any("in order" in p for p in out_of_order)

        reply = valid_reply()
        reply["noticed"].append(lettered_a())
        _, twice = check(reply)
        assert any("more than once" in p for p in twice)

    @pytest.mark.parametrize("text", [
        "The analysis flagged asymmetry in this spot.",
        "We noticed the two halves look different.",
        "The two halves look different, which means the shape does not match.",
    ])
    def test_a_lettered_item_must_read_we_noticed_which_means(self, text):
        reply = valid_reply()
        reply["noticed"][0]["text"] = text
        _, problems = check(reply)
        assert any("We noticed" in p and "which means" in p for p in problems)

    @pytest.mark.parametrize("text", [
        "We noticed uneven color, which means the color is uneven.",
        "We noticed the edges are uneven, which means the edges are uneven.",
        "We noticed unevenly shaped halves, which means the halves are unevenly shaped.",
    ])
    def test_a_meaning_that_only_repeats_the_observation_is_rejected(self, text):
        reply = valid_reply()
        reply["noticed"][0]["text"] = text
        _, problems = check(reply)
        assert any("only repeats" in p for p in problems)

    @pytest.mark.parametrize("text", [
        "We noticed uneven halves, which means one half does not match the other.",
        "We noticed uneven edges, which means edges can be ragged, notched, blurred, or irregular.",
        "We noticed the color varies across the spot, which means shades of black, brown, or tan may be present.",
    ])
    def test_a_meaning_that_adds_something_is_accepted(self, text):
        reply = valid_reply()
        reply["noticed"][0]["text"] = text
        _, problems = check(reply)
        assert not any("only repeats" in p for p in problems)

    def test_a_meaning_needs_only_one_new_word_to_count(self):
        assert knowledge._repeats_the_observation("We noticed uneven color, which means the color is uneven.")
        assert not knowledge._repeats_the_observation("We noticed uneven color, which means the color is patchy.")

    def test_a_lettered_item_must_be_short(self):
        reply = valid_reply()
        reply["noticed"][0]["text"] = "We noticed " + "the shape is uneven and " * 12 + ", which means one half does not match."
        _, problems = check(reply)
        assert any("under" in p and "characters" in p for p in problems)

    def test_the_meaning_of_a_feature_has_to_come_from_the_booklet(self):
        reply = valid_reply()
        reply["noticed"][0] = statement(
            "We noticed the two halves look different, which means the shape is uneven.", letter="A"
        )
        _, problems = check(reply)
        assert any("reference passage" in p for p in problems)

    def test_size_that_cannot_be_measured_needs_its_own_d_section(self):
        payload = make_payload(flagged=("asymmetry",), diameter=False)
        reply = {"noticed": [lettered_a()], "next_steps": valid_reply()["next_steps"]}
        _, without = check(reply, payload)
        assert any("missing a lettered item for D" in p for p in without)

        reply["noticed"].append(
            statement("We noticed this photo has no ruler or scale, which means we cannot measure the spot in millimeters.", letter="D")
        )
        _, problems = check(reply, payload)
        assert problems == []

    def test_a_comparison_with_the_last_photo_needs_an_e_section(self):
        payload = dict(ASYM_BORDER, change_since_last_photo={"available": False})
        _, without = check(valid_reply(), payload)
        assert any("missing a lettered item for E" in p for p in without)

        reply = valid_reply()
        reply["noticed"].append(
            statement("We noticed there is no earlier photo of this spot, which means we cannot check for changes over time.", letter="E")
        )
        _, problems = check(reply, payload)
        assert problems == []

    def test_points_that_belong_to_no_letter_are_still_allowed(self):
        reply = valid_reply()
        reply["noticed"].insert(0, statement("You reported these about the spot: itchy.", letter=None))
        _, problems = check(reply)
        assert problems == []

    def test_the_explanation_is_grouped_under_the_letters(self):
        source, _ = knowledge.load()
        passages = knowledge.select_passages(ASYM_BORDER)
        reply = valid_reply()
        reply["noticed"].insert(0, statement("You reported these about the spot: itchy."))
        text = knowledge.render_explanation(reply, source, passages)
        body = text.split("\n\nSources:")[0]

        assert "What the analysis noticed:\n\u2022 You reported these about the spot: itchy.\n\nA: Asymmetry (shape)\n\u2022 We noticed" in body
        assert body.index("A: Asymmetry (shape)") < body.index("B: Border (edges)") < body.index("Suggested next steps:")
        assert "C: Color" not in body

    @pytest.mark.parametrize("payload", [
        ASYM_BORDER,
        make_payload(),
        make_payload(flagged=("asymmetry", "border", "color", "diameter_mm")),
        make_payload(flagged=("color",), diameter=False),
        dict(make_payload(flagged=("border",), diameter=False), change_since_last_photo={"available": True, "changes_noticed": ["shape", "color"]}),
        dict(make_payload(diameter=False), change_since_last_photo={"available": True, "changes_noticed": []}),
        dict(make_payload(diameter=False), change_since_last_photo={"available": False}),
    ])
    def test_the_fallback_gives_every_expected_letter_and_passes_the_validator(self, payload):
        passages = knowledge.select_passages(payload)
        data = knowledge.fallback_data(payload, passages)
        letters = [item["letter"] for item in data["noticed"] if item.get("letter")]
        assert letters == knowledge._expected_letters(payload)
        _, problems = knowledge.parse_and_validate(json.dumps(data), payload, passages)
        assert problems == []
