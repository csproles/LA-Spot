"""POST /api/chat as seen through the HTTP layer: auth, validation, and rate limiting.

answer_question itself (retrieval + grounding + OpenAI calls) is covered by
tests/test_textbook_chat.py; here it's monkeypatched to a stub so these tests
never make a real API call.
"""

import importlib
import os

import pytest

import store

KEY = "test-internal-key"
ALICE = {"X-Internal-Api-Key": KEY, "X-User-Id": "alice"}


@pytest.fixture(scope="module")
def main_module(tmp_path_factory):
    store.DB_PATH = str(tmp_path_factory.mktemp("chat-api") / "api.db")
    os.environ["SKINCHECK_INTERNAL_KEY"] = KEY
    main = importlib.import_module("main")
    main.app.config["TESTING"] = True
    return main


@pytest.fixture
def client(main_module):
    return main_module.app.test_client()


@pytest.fixture(autouse=True)
def clean_state(main_module, monkeypatch):
    store.reset_db()
    main_module.limiter.reset()
    monkeypatch.setattr(
        main_module,
        "answer_question",
        lambda question, page_context=None, history=None: {
            "refused": False,
            "answer": f"stub answer to: {question}",
            "refusalReason": None,
            "sources": [],
        },
    )
    yield
    main_module.limiter.reset()


class TestAuth:
    def test_no_headers_is_unauthorized(self, client):
        assert client.post("/api/chat", json={"question": "What is melanoma?"}).status_code == 401

    def test_valid_caller_is_let_through(self, client):
        response = client.post("/api/chat", json={"question": "What is melanoma?"}, headers=ALICE)
        assert response.status_code == 200
        assert response.get_json()["answer"] == "stub answer to: What is melanoma?"


class TestValidation:
    def test_missing_question_is_rejected(self, client):
        assert client.post("/api/chat", json={}, headers=ALICE).status_code == 400

    def test_oversized_question_is_rejected(self, client):
        response = client.post("/api/chat", json={"question": "a" * 501}, headers=ALICE)
        assert response.status_code == 400

    def test_bad_page_context_is_rejected(self, client):
        response = client.post(
            "/api/chat",
            json={"question": "hi", "pageContext": {"page": "results", "data": {"nested": {"a": 1}}}},
            headers=ALICE,
        )
        assert response.status_code == 400

    def test_valid_page_context_is_accepted(self, client):
        response = client.post(
            "/api/chat",
            json={"question": "hi", "pageContext": {"page": "results", "data": {"riskScore": 42}}},
            headers=ALICE,
        )
        assert response.status_code == 200


class TestRateLimit:
    def test_chat_is_capped_per_user(self, client):
        for _ in range(8):
            assert client.post("/api/chat", json={"question": "hi"}, headers=ALICE).status_code == 200

        blocked = client.post("/api/chat", json={"question": "hi"}, headers=ALICE)
        assert blocked.status_code == 429
