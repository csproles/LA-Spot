"""The HTTP layer's identity rules: who may call, and that they only see their own.

main.py builds the detector and opens the database at import time, so the
module is imported once here, against a throw-away database and with the
shared key set, and every test drives it through Flask's test client.
"""

import importlib
import os

import pytest

import store

KEY = "test-internal-key"
ALICE = {"X-Internal-Api-Key": KEY, "X-User-Id": "alice"}
BOB = {"X-Internal-Api-Key": KEY, "X-User-Id": "bob"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    store.DB_PATH = str(tmp_path_factory.mktemp("api") / "api.db")
    os.environ["SKINCHECK_INTERNAL_KEY"] = KEY
    main = importlib.import_module("main")
    main.app.config["TESTING"] = True
    return main.app.test_client()


@pytest.fixture(autouse=True)
def clean_db():
    store.reset_db()


class TestGate:
    def test_health_needs_nothing(self, client):
        assert client.get("/health").status_code == 200

    def test_no_headers_is_unauthorized(self, client):
        assert client.get("/api/spots").status_code == 401

    def test_wrong_key_is_unauthorized(self, client):
        headers = {**ALICE, "X-Internal-Api-Key": "nope"}
        assert client.get("/api/spots", headers=headers).status_code == 401

    def test_key_without_user_is_unauthorized(self, client):
        assert client.get("/api/spots", headers={"X-Internal-Api-Key": KEY}).status_code == 401

    def test_malformed_user_id_is_unauthorized(self, client):
        headers = {**ALICE, "X-User-Id": "alice; drop table"}
        assert client.get("/api/spots", headers=headers).status_code == 401

    def test_valid_caller_is_let_through(self, client):
        response = client.get("/api/spots", headers=ALICE)
        assert response.status_code == 200
        assert response.get_json() == {"spots": []}


class TestScoping:
    def test_spots_belong_to_the_caller(self, client):
        created = client.post("/api/spots", json={"label": "mole", "bodyRegion": "Back"}, headers=ALICE)
        assert created.status_code == 201
        spot_id = created.get_json()["id"]

        assert client.get(f"/api/spots/{spot_id}", headers=ALICE).status_code == 200
        assert client.get(f"/api/spots/{spot_id}", headers=BOB).status_code == 404
        assert client.patch(f"/api/spots/{spot_id}", json={"label": "x"}, headers=BOB).status_code == 404
        assert client.get("/api/spots", headers=BOB).get_json() == {"spots": []}

    def test_profiles_belong_to_the_caller(self, client):
        assert client.put("/api/profile", json={"fullName": "Alice"}, headers=ALICE).status_code == 200
        assert client.get("/api/profile", headers=ALICE).get_json()["fullName"] == "Alice"
        assert client.get("/api/profile", headers=BOB).get_json() == {"configured": False}

    def test_pending_results_belong_to_the_caller(self, client):
        import main

        main._results_store["proc_test"] = {"user_id": "alice", "abcde_scores": {}}
        try:
            assert client.post("/api/image/save/proc_test", headers=BOB).status_code == 404
            assert client.post("/api/image/explain/proc_test", headers=BOB).status_code == 404
            assert client.get("/api/image/results/proc_test", headers=BOB).status_code == 404
        finally:
            main._results_store.pop("proc_test", None)


class TestAccount:
    def test_export_and_delete(self, client):
        client.post("/api/spots", json={"label": "mole", "bodyRegion": "Back"}, headers=ALICE)
        client.put("/api/profile", json={"fullName": "Alice"}, headers=ALICE)
        client.post("/api/spots", json={"label": "other", "bodyRegion": "Arm"}, headers=BOB)

        export = client.get("/api/account/export", headers=ALICE).get_json()
        assert export["profile"]["fullName"] == "Alice"
        assert [s["label"] for s in export["spots"]] == ["mole"]
        assert export["checks"] == []

        deleted = client.delete("/api/account", headers=ALICE)
        assert deleted.status_code == 200
        assert deleted.get_json()["spots"] == 1
        assert client.get("/api/account/export", headers=ALICE).get_json() == {
            "profile": None, "spots": [], "checks": [],
        }
        assert [s["label"] for s in client.get("/api/spots", headers=BOB).get_json()["spots"]] == ["other"]
