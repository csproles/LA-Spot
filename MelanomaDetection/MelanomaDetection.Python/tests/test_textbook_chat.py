import json
from types import SimpleNamespace

import numpy as np
import pytest

import textbook_chat as tc

PASSAGE_A = tc.Passage(id="tb-1", source="Book A", page=3, text="Melanoma is the most serious type of skin cancer.")
PASSAGE_B = tc.Passage(id="tb-2", source="Book A", page=9, text="Wear sunscreen with SPF 30 or higher every day.")

APP_PASSAGE = tc.Passage(id="app-1", source="App Help", page=None, text="This app scores spots on a 0-100 scale.")

REFUSED_REPLY = json.dumps({"refused": True, "refusal_reason": "Not covered by the textbook.", "answer": []})


class FakeClient:
    def __init__(self, chat_replies=(), embedding=None):
        self._chat_replies = list(chat_replies)
        self.calls = []
        self.embedding_calls = []
        self._embedding = embedding if embedding is not None else [1.0, 0.0]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat))
        self.embeddings = SimpleNamespace(create=self._create_embedding)

    def _create_chat(self, **kwargs):
        self.calls.append(kwargs)
        reply = self._chat_replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    def _create_embedding(self, **kwargs):
        self.embedding_calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(embedding=self._embedding)])


def valid_reply(text="Melanoma is the most serious type of skin cancer.", source_id="tb-1", quote=None):
    return json.dumps(
        {
            "refused": False,
            "refusal_reason": None,
            "answer": [
                {
                    "text": text,
                    "basis": "passage",
                    "source_ids": [source_id],
                    "quote": quote or "Melanoma is the most serious type of skin cancer.",
                }
            ],
        }
    )


@pytest.fixture(autouse=True)
def no_app_passages(monkeypatch):
    """Isolate tests from the real (policy.py-derived) app passages unless a test opts in."""
    monkeypatch.setattr(tc, "load_app_passages", lambda: {})


class TestMultiBookLoading:
    """load_textbook_passages/load_embeddings against a real knowledge/textbooks/-shaped
    folder (rather than the monkeypatched fakes the rest of this file uses), since these
    two are the only functions that actually read that folder structure."""

    def test_loads_passages_across_multiple_books(self, tmp_path, monkeypatch):
        books_dir = tmp_path / "textbooks"
        books_dir.mkdir()
        (books_dir / "book-a.json").write_text(
            json.dumps({"source": {"title": "Book A"}, "passages": [{"id": "a-1", "page": 5, "text": "Text from book A."}]})
        )
        (books_dir / "book-b.json").write_text(
            json.dumps({"source": {"title": "Book B"}, "passages": [{"id": "b-1", "page": 9, "text": "Text from book B."}]})
        )
        monkeypatch.setattr(tc, "TEXTBOOKS_DIR", books_dir)
        tc.load_textbook_passages.cache_clear()
        try:
            passages = tc.load_textbook_passages()
            assert passages["a-1"].source == "Book A" and passages["a-1"].page == 5
            assert passages["b-1"].source == "Book B" and passages["b-1"].page == 9
        finally:
            tc.load_textbook_passages.cache_clear()

    def test_duplicate_ids_across_books_raise(self, tmp_path, monkeypatch):
        books_dir = tmp_path / "textbooks"
        books_dir.mkdir()
        (books_dir / "book-a.json").write_text(
            json.dumps({"source": {"title": "Book A"}, "passages": [{"id": "dup", "page": 1, "text": "First."}]})
        )
        (books_dir / "book-b.json").write_text(
            json.dumps({"source": {"title": "Book B"}, "passages": [{"id": "dup", "page": 2, "text": "Second."}]})
        )
        monkeypatch.setattr(tc, "TEXTBOOKS_DIR", books_dir)
        tc.load_textbook_passages.cache_clear()
        try:
            with pytest.raises(ValueError, match="Duplicate passage id"):
                tc.load_textbook_passages()
        finally:
            tc.load_textbook_passages.cache_clear()

    def test_missing_textbooks_dir_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tc, "TEXTBOOKS_DIR", tmp_path / "does-not-exist")
        tc.load_textbook_passages.cache_clear()
        try:
            assert tc.load_textbook_passages() == {}
        finally:
            tc.load_textbook_passages.cache_clear()

    def test_embeddings_concatenate_across_books(self, tmp_path, monkeypatch):
        books_dir = tmp_path / "textbooks"
        books_dir.mkdir()
        (books_dir / "book-a.json").write_text(
            json.dumps({"source": {"title": "Book A"}, "passages": [{"id": "a-1", "page": 5, "text": "Text from book A."}]})
        )
        (books_dir / "book-b.json").write_text(
            json.dumps({"source": {"title": "Book B"}, "passages": [{"id": "b-1", "page": 9, "text": "Text from book B."}]})
        )
        np.savez(books_dir / "book-a_embeddings.npz", ids=np.array(["a-1"]), vectors=np.array([[1.0, 0.0]], dtype=np.float32))
        np.savez(books_dir / "book-b_embeddings.npz", ids=np.array(["b-1"]), vectors=np.array([[0.0, 1.0]], dtype=np.float32))
        monkeypatch.setattr(tc, "TEXTBOOKS_DIR", books_dir)
        tc.load_textbook_passages.cache_clear()
        tc.load_embeddings.cache_clear()
        try:
            ids, vectors = tc.load_embeddings()
            assert set(ids) == {"a-1", "b-1"}
            assert vectors.shape == (2, 2)
        finally:
            tc.load_textbook_passages.cache_clear()
            tc.load_embeddings.cache_clear()


class TestRetrieve:
    def test_returns_best_matches_first(self, monkeypatch):
        monkeypatch.setattr(tc, "load_textbook_passages", lambda: {"tb-1": PASSAGE_A, "tb-2": PASSAGE_B})
        monkeypatch.setattr(
            tc,
            "load_embeddings",
            lambda: (["tb-1", "tb-2"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)),
        )
        client = FakeClient(embedding=[1.0, 0.0])
        results = tc.retrieve("question", client)
        assert results[0][0].id == "tb-1"
        assert results[0][1] > results[1][1]

    def test_no_textbook_ingested_yet_returns_nothing_and_makes_no_call(self, monkeypatch):
        monkeypatch.setattr(tc, "load_textbook_passages", lambda: {})
        monkeypatch.setattr(tc, "load_embeddings", lambda: ([], np.zeros((0, 0), dtype=np.float32)))
        client = FakeClient()
        assert tc.retrieve("question", client) == []
        assert client.embedding_calls == []


class TestAnswerQuestion:
    def test_nothing_available_is_a_deterministic_refusal_with_no_llm_call(self, monkeypatch):
        monkeypatch.setattr(tc, "load_textbook_passages", lambda: {})
        monkeypatch.setattr(tc, "load_embeddings", lambda: ([], np.zeros((0, 0), dtype=np.float32)))
        client = FakeClient()
        result = tc.answer_question("What is melanoma?", client=client)
        assert result["refused"] is True
        assert client.calls == []

    def test_valid_grounded_reply_is_accepted(self, monkeypatch):
        monkeypatch.setattr(tc, "load_textbook_passages", lambda: {"tb-1": PASSAGE_A})
        monkeypatch.setattr(tc, "load_embeddings", lambda: (["tb-1"], np.array([[1.0, 0.0]], dtype=np.float32)))
        client = FakeClient(chat_replies=[valid_reply()], embedding=[1.0, 0.0])
        result = tc.answer_question("What is melanoma?", client=client)
        assert result["refused"] is False
        assert result["sources"] == ["Book A: page 3"]
        assert len(client.calls) == 1

    def test_multiple_citations_from_the_same_book_are_grouped(self, monkeypatch):
        monkeypatch.setattr(tc, "load_textbook_passages", lambda: {"tb-1": PASSAGE_A, "tb-2": PASSAGE_B})
        monkeypatch.setattr(
            tc, "load_embeddings", lambda: (["tb-1", "tb-2"], np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
        )
        reply = json.dumps(
            {
                "refused": False,
                "refusal_reason": None,
                "answer": [
                    {
                        "text": "Melanoma is the most serious type of skin cancer.",
                        "basis": "passage",
                        "source_ids": ["tb-1"],
                        "quote": "Melanoma is the most serious type of skin cancer.",
                    },
                    {
                        "text": "Wear sunscreen with SPF 30 or higher every day.",
                        "basis": "passage",
                        "source_ids": ["tb-2"],
                        "quote": "Wear sunscreen with SPF 30 or higher every day.",
                    },
                ],
            }
        )
        client = FakeClient(chat_replies=[reply], embedding=[1.0, 0.0])
        result = tc.answer_question("Tell me two things.", client=client)
        assert result["sources"] == ["Book A: page 3, page 9"]

    def test_model_self_refusal_is_passed_through(self, monkeypatch):
        monkeypatch.setattr(tc, "load_textbook_passages", lambda: {"tb-1": PASSAGE_A})
        monkeypatch.setattr(tc, "load_embeddings", lambda: (["tb-1"], np.array([[1.0, 0.0]], dtype=np.float32)))
        client = FakeClient(chat_replies=[REFUSED_REPLY], embedding=[1.0, 0.0])
        result = tc.answer_question("What's the weather today?", client=client)
        assert result["refused"] is True
        assert result["refusalReason"] == "Not covered by the textbook."

    def test_uncited_reply_is_rejected_then_falls_back_to_a_refusal_not_a_guess(self, monkeypatch):
        monkeypatch.setattr(tc, "load_textbook_passages", lambda: {"tb-1": PASSAGE_A})
        monkeypatch.setattr(tc, "load_embeddings", lambda: (["tb-1"], np.array([[1.0, 0.0]], dtype=np.float32)))
        bad_reply = valid_reply(text="Melanoma affects about 3% of people.", quote="not a real quote from the passage")
        client = FakeClient(chat_replies=[bad_reply, bad_reply], embedding=[1.0, 0.0])
        result = tc.answer_question("What is melanoma?", client=client)
        assert len(client.calls) == tc.MAX_ATTEMPTS
        assert result["refused"] is True
        assert result["answer"] is None

    def test_low_similarity_textbook_passages_are_excluded_from_the_prompt(self, monkeypatch):
        monkeypatch.setattr(tc, "load_textbook_passages", lambda: {"tb-1": PASSAGE_A})
        monkeypatch.setattr(tc, "load_embeddings", lambda: (["tb-1"], np.array([[0.0, 1.0]], dtype=np.float32)))
        monkeypatch.setattr(tc, "load_app_passages", lambda: {"app-1": APP_PASSAGE})
        client = FakeClient(chat_replies=[REFUSED_REPLY], embedding=[1.0, 0.0])  # orthogonal -> similarity 0
        tc.answer_question("unrelated question", client=client)
        prompt = client.calls[0]["messages"][-1]["content"]
        assert "tb-1" not in prompt
        assert "app-1" in prompt

    def test_screen_data_numbers_are_allowed_without_a_citation(self, monkeypatch):
        monkeypatch.setattr(tc, "load_textbook_passages", lambda: {})
        monkeypatch.setattr(tc, "load_embeddings", lambda: ([], np.zeros((0, 0), dtype=np.float32)))
        monkeypatch.setattr(tc, "load_app_passages", lambda: {"app-1": APP_PASSAGE})
        screen_reply = json.dumps(
            {
                "refused": False,
                "refusal_reason": None,
                "answer": [{"text": "Your current risk score is 42.", "basis": "screen", "source_ids": [], "quote": None}],
            }
        )
        client = FakeClient(chat_replies=[screen_reply], embedding=[1.0, 0.0])
        result = tc.answer_question(
            "What does my score mean?",
            page_context={"page": "results", "data": {"riskScore": 42}},
            client=client,
        )
        assert result["refused"] is False
        assert "42" in result["answer"]

    def test_screen_number_not_present_anywhere_is_rejected(self, monkeypatch):
        monkeypatch.setattr(tc, "load_textbook_passages", lambda: {})
        monkeypatch.setattr(tc, "load_embeddings", lambda: ([], np.zeros((0, 0), dtype=np.float32)))
        monkeypatch.setattr(tc, "load_app_passages", lambda: {"app-1": APP_PASSAGE})
        bad_reply = json.dumps(
            {
                "refused": False,
                "refusal_reason": None,
                "answer": [{"text": "Your risk score is 99.", "basis": "screen", "source_ids": [], "quote": None}],
            }
        )
        client = FakeClient(chat_replies=[bad_reply, bad_reply], embedding=[1.0, 0.0])
        result = tc.answer_question(
            "What does my score mean?",
            page_context={"page": "results", "data": {"riskScore": 42}},
            client=client,
        )
        assert result["refused"] is True

    def test_forbidden_diagnostic_language_is_rejected(self, monkeypatch):
        monkeypatch.setattr(tc, "load_textbook_passages", lambda: {"tb-1": PASSAGE_A})
        monkeypatch.setattr(tc, "load_embeddings", lambda: (["tb-1"], np.array([[1.0, 0.0]], dtype=np.float32)))
        bad_reply = valid_reply(text="This spot is malignant.")
        client = FakeClient(chat_replies=[bad_reply, bad_reply], embedding=[1.0, 0.0])
        result = tc.answer_question("Is my spot cancer?", client=client)
        assert result["refused"] is True

    def test_api_failures_are_not_swallowed(self, monkeypatch):
        monkeypatch.setattr(tc, "load_textbook_passages", lambda: {"tb-1": PASSAGE_A})
        monkeypatch.setattr(tc, "load_embeddings", lambda: (["tb-1"], np.array([[1.0, 0.0]], dtype=np.float32)))
        client = FakeClient(embedding=[1.0, 0.0])
        client.embeddings.create = lambda **_: (_ for _ in ()).throw(RuntimeError("no api key"))
        with pytest.raises(RuntimeError):
            tc.answer_question("What is melanoma?", client=client)
