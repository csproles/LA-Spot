"""Rate limiting and input validation as seen through the HTTP layer."""

import importlib
import io
import os

import cv2
import numpy as np
import pytest

import store
from tests.test_validation import png_header

KEY = "test-internal-key"
ALICE = {"X-Internal-Api-Key": KEY, "X-User-Id": "alice"}
BOB = {"X-Internal-Api-Key": KEY, "X-User-Id": "bob"}


@pytest.fixture(scope="module")
def main_module(tmp_path_factory):
    store.DB_PATH = str(tmp_path_factory.mktemp("hardening") / "api.db")
    os.environ["SKINCHECK_INTERNAL_KEY"] = KEY
    main = importlib.import_module("main")
    main.app.config["TESTING"] = True
    return main


@pytest.fixture
def client(main_module):
    return main_module.app.test_client()


@pytest.fixture(autouse=True)
def clean_state(main_module):
    store.reset_db()
    main_module.limiter.reset()
    yield
    main_module.limiter.reset()


def upload(client, data, name="mole.png", headers=ALICE, **form):
    return client.post(
        "/api/image/process",
        data={"file": (io.BytesIO(data), name), **form},
        headers=headers,
        content_type="multipart/form-data",
    )


class TestRateLimits:
    def test_explanations_are_capped_per_user(self, client):
        for _ in range(5):
            assert client.post("/api/image/explain/proc_nope", headers=ALICE).status_code == 404

        blocked = client.post("/api/image/explain/proc_nope", headers=ALICE)
        assert blocked.status_code == 429
        assert int(blocked.headers["Retry-After"]) >= 1
        assert "Too many requests" in blocked.get_json()["error"]

    def test_one_users_limit_does_not_touch_another(self, client):
        for _ in range(6):
            client.post("/api/image/explain/proc_nope", headers=ALICE)

        assert client.post("/api/image/explain/proc_nope", headers=BOB).status_code == 404

    def test_analysis_is_capped_per_user(self, client):
        for _ in range(10):
            assert upload(client, b"not an image").status_code == 400
        assert upload(client, b"not an image").status_code == 429

    def test_ordinary_reads_have_a_higher_cap(self, client, main_module):
        for _ in range(main_module.DEFAULT_RATE_LIMIT):
            assert client.get("/api/spots", headers=ALICE).status_code == 200
        assert client.get("/api/spots", headers=ALICE).status_code == 429

    def test_health_is_never_limited(self, client):
        for _ in range(200):
            assert client.get("/health").status_code == 200

    def test_guessing_the_internal_key_gets_locked_out(self, client, main_module):
        wrong = {**ALICE, "X-Internal-Api-Key": "guess"}
        for _ in range(main_module.AUTH_FAILURE_LIMIT):
            assert client.get("/api/spots", headers=wrong).status_code == 401
        assert client.get("/api/spots", headers=wrong).status_code == 429


class TestUploadValidation:
    def test_a_real_photo_still_goes_through(self, client):
        photo = np.full((300, 300, 3), (200, 190, 220), np.uint8)
        cv2.circle(photo, (150, 150), 60, (40, 50, 90), -1)
        _, encoded = cv2.imencode(".png", photo)

        response = upload(client, encoded.tobytes(), notes="  itchy lately ", symptoms=["itchy"])
        assert response.status_code == 200

        results = client.get(f"/api/image/results/{response.get_json()['processingId']}", headers=ALICE).get_json()
        assert results["notes"] == "itchy lately"
        assert results["symptoms"] == ["itchy"]

    def test_rejects_non_image_bytes_with_image_extension(self, client):
        response = upload(client, b"MZ\x90\x00 this is an executable")
        assert response.status_code == 400
        assert "not a valid" in response.get_json()["error"]

    def test_rejects_empty_file(self, client):
        assert upload(client, b"").status_code == 400

    def test_rejects_decompression_bomb(self, client):
        response = upload(client, png_header(100_000, 100_000))
        assert response.status_code == 400
        assert "too large" in response.get_json()["error"]

    def test_rejects_malformed_spot_id(self, client):
        response = upload(client, png_header(10, 10), spot_id="'; drop table spots;--")
        assert response.status_code == 400

    def test_rejects_oversized_notes(self, client):
        response = upload(client, png_header(10, 10), notes="x" * 2001)
        assert response.status_code == 400
        assert "notes" in response.get_json()["error"]


class TestJsonValidation:
    def test_spot_label_too_long(self, client):
        response = client.post("/api/spots", json={"label": "x" * 61, "bodyRegion": "Back"}, headers=ALICE)
        assert response.status_code == 400

    def test_spot_label_required(self, client):
        response = client.post("/api/spots", json={"label": "  ", "bodyRegion": "Back"}, headers=ALICE)
        assert response.status_code == 400
        assert "label is required" in response.get_json()["error"]

    def test_spot_fields_must_be_text_not_a_server_error(self, client):
        response = client.post("/api/spots", json={"label": 5, "bodyRegion": ["Back"]}, headers=ALICE)
        assert response.status_code == 400

    def test_json_body_must_be_an_object(self, client):
        assert client.post("/api/spots", json=["label"], headers=ALICE).status_code == 400
        assert client.put("/api/profile", json="hello", headers=ALICE).status_code == 400

    def test_spot_update_validates(self, client):
        spot_id = client.post("/api/spots", json={"label": "mole", "bodyRegion": "Back"}, headers=ALICE).get_json()["id"]

        assert client.patch(f"/api/spots/{spot_id}", json={"archived": "yes"}, headers=ALICE).status_code == 400
        assert client.patch(f"/api/spots/{spot_id}", json={"label": "y" * 61}, headers=ALICE).status_code == 400
        assert client.patch(f"/api/spots/{spot_id}", json={"label": "renamed"}, headers=ALICE).status_code == 200

    def test_profile_rejects_bad_types(self, client):
        for body in (
            {"fullName": "x" * 121},
            {"location": "x" * 121},
            {"fitzpatrick": "3"},
            {"fitzpatrick": 7},
            {"sunExposure": "extreme"},
            {"familyHistory": "false"},
            {"birthYear": 1850},
            {"birthYear": "1990"},
            {"sex": "other"},
        ):
            assert client.put("/api/profile", json=body, headers=ALICE).status_code == 400, body

    def test_spot_body_side_must_be_front_or_back(self, client):
        bad = client.post("/api/spots", json={"label": "m", "bodyRegion": "Chest/Upper Back", "bodySide": "left"}, headers=ALICE)
        good = client.post("/api/spots", json={"label": "m", "bodyRegion": "Chest/Upper Back", "bodySide": "back"}, headers=ALICE)
        assert bad.status_code == 400
        assert good.status_code == 201 and good.get_json()["bodySide"] == "back"

    def test_profile_keeps_the_risk_model_fields(self, client):
        response = client.put("/api/profile", json={"birthYear": 1990, "sex": "female"}, headers=ALICE)
        assert response.status_code == 200
        assert response.get_json()["birthYear"] == 1990 and response.get_json()["sex"] == "female"

    def test_profile_accepts_good_input(self, client):
        body = {"fullName": "Alice", "location": "Austin, TX", "fitzpatrick": 3, "sunExposure": "high", "familyHistory": True}
        response = client.put("/api/profile", json=body, headers=ALICE)
        assert response.status_code == 200
        assert response.get_json()["familyHistory"] is True
