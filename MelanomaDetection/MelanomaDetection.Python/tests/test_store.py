"""Spot/check/profile persistence.

The point of this module is that history outlives a restart, so these tests
drive it through real connections against a temporary database file rather than
mocking sqlite3 out.
"""

import pytest

import store


@pytest.fixture
def db(tmp_path, monkeypatch):
    """An isolated database file for one test."""
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "test.db"))
    store.init_db()
    return store


SCORES = {
    "asymmetry": {"score": 4.2, "details": {}},
    "border": {"score": 3.0, "details": {}},
    "color": {"score": 6.0, "details": {}},
    "diameter": {"score": 5.5, "details": {"diameter_mm": 4.2}},
    "evolving": {"score": None, "details": {}},
}


def save(db, processing_id, spot_id, risk_score, processed_at, **kwargs):
    db.save_check(
        processing_id=processing_id,
        spot_id=spot_id,
        risk_score=risk_score,
        abcde_scores=kwargs.get("scores", SCORES),
        location=kwargs.get("location", "Left Arm"),
        symptoms=kwargs.get("symptoms", []),
        notes=kwargs.get("notes", ""),
        processed_at=processed_at,
        thumbnail_png=kwargs.get("thumbnail", b"THUMB"),
        mask_png=kwargs.get("mask", b"MASK"),
        mm_per_px=kwargs.get("mm_per_px"),
        area_px=kwargs.get("area_px"),
        lab=kwargs.get("lab"),
    )


class TestSpots:
    def test_create_and_read_back(self, db):
        spot = db.create_spot("dark mole", "Left Arm")
        assert spot["id"].startswith("spot_")
        assert db.get_spot(spot["id"])["label"] == "dark mole"

    def test_missing_spot_is_none(self, db):
        assert db.get_spot("spot_nope") is None

    def test_rename(self, db):
        spot = db.create_spot("mole", "Back")
        assert db.update_spot(spot["id"], label="mole (watch)")["label"] == "mole (watch)"

    def test_archived_spots_are_hidden_by_default(self, db):
        spot = db.create_spot("mole", "Back")
        db.update_spot(spot["id"], archived=True)
        assert db.list_spots() == []
        assert len(db.list_spots(include_archived=True)) == 1

    def test_update_with_nothing_to_change_is_harmless(self, db):
        spot = db.create_spot("mole", "Back")
        assert db.update_spot(spot["id"])["label"] == "mole"


class TestAggregates:
    def test_a_spot_with_no_checks(self, db):
        db.create_spot("mole", "Back")
        summary = db.list_spots()[0]
        assert summary["checkCount"] == 0
        assert summary["lastRiskScore"] is None
        assert summary["trend"] is None

    def test_trend_needs_two_checks(self, db):
        spot = db.create_spot("mole", "Back")
        save(db, "p1", spot["id"], 40.0, "2026-01-01T00:00:00+00:00")
        assert db.list_spots()[0]["trend"] is None

    def test_rising_and_falling_trend(self, db):
        spot = db.create_spot("mole", "Back")
        save(db, "p1", spot["id"], 40.0, "2026-01-01T00:00:00+00:00")
        save(db, "p2", spot["id"], 55.0, "2026-02-01T00:00:00+00:00")
        summary = db.list_spots()[0]
        assert (summary["firstRiskScore"], summary["lastRiskScore"]) == (40.0, 55.0)
        assert summary["trend"] == "up"

        save(db, "p3", spot["id"], 41.0, "2026-03-01T00:00:00+00:00")
        assert db.list_spots()[0]["trend"] == "down"

    def test_small_movement_reads_as_flat(self, db):
        """Under this pipeline's own run-to-run noise, so not a real move."""
        spot = db.create_spot("mole", "Back")
        save(db, "p1", spot["id"], 50.0, "2026-01-01T00:00:00+00:00")
        save(db, "p2", spot["id"], 51.0, "2026-02-01T00:00:00+00:00")
        assert db.list_spots()[0]["trend"] == "flat"

    def test_ordering_is_by_recency_of_last_check(self, db):
        stale = db.create_spot("stale", "Back")
        fresh = db.create_spot("fresh", "Left Arm")
        save(db, "p1", stale["id"], 40.0, "2026-01-01T00:00:00+00:00")
        save(db, "p2", fresh["id"], 40.0, "2026-06-01T00:00:00+00:00")
        assert [s["label"] for s in db.list_spots()] == ["fresh", "stale"]


class TestChecks:
    def test_timeline_is_oldest_first(self, db):
        spot = db.create_spot("mole", "Back")
        save(db, "p2", spot["id"], 50.0, "2026-02-01T00:00:00+00:00")
        save(db, "p1", spot["id"], 40.0, "2026-01-01T00:00:00+00:00")
        assert [c["processingId"] for c in db.get_checks_for_spot(spot["id"])] == ["p1", "p2"]

    def test_last_check_is_the_most_recent(self, db):
        spot = db.create_spot("mole", "Back")
        save(db, "p1", spot["id"], 40.0, "2026-01-01T00:00:00+00:00")
        save(db, "p2", spot["id"], 50.0, "2026-02-01T00:00:00+00:00")
        assert db.get_last_check(spot["id"])["processingId"] == "p2"

    def test_last_check_of_an_unchecked_spot_is_none(self, db):
        spot = db.create_spot("mole", "Back")
        assert db.get_last_check(spot["id"]) is None

    def test_binary_and_measurement_fields_round_trip(self, db):
        spot = db.create_spot("mole", "Back")
        save(
            db, "p1", spot["id"], 48.0, "2026-01-01T00:00:00+00:00",
            thumbnail=b"THUMBNAIL-BYTES", mask=b"MASK-BYTES",
            mm_per_px=0.05, area_px=2800, lab=(135.0, 153.0, 132.0),
            symptoms=["Itchy", "Growing"], notes="watch this",
        )
        check = db.get_check("p1")
        assert check["thumbnail"] == b"THUMBNAIL-BYTES"
        assert check["mask"] == b"MASK-BYTES"
        assert check["mmPerPx"] == 0.05
        assert check["areaPx"] == 2800
        assert check["lab"] == (135.0, 153.0, 132.0)
        assert check["symptoms"] == ["Itchy", "Growing"]
        assert check["notes"] == "watch this"
        assert check["diameterMm"] == 4.2  # lifted out of the diameter details

    def test_lab_is_none_when_not_recorded(self, db):
        spot = db.create_spot("mole", "Back")
        save(db, "p1", spot["id"], 48.0, "2026-01-01T00:00:00+00:00", lab=None)
        assert db.get_check("p1")["lab"] is None

    def test_resaving_updates_rather_than_duplicating(self, db):
        spot = db.create_spot("mole", "Back")
        other = db.create_spot("other", "Left Arm")
        save(db, "p1", None, 48.0, "2026-01-01T00:00:00+00:00")
        save(db, "p1", other["id"], 48.0, "2026-01-01T00:00:00+00:00", notes="filed later")

        assert len(db.list_checks()) == 1
        assert db.get_check("p1")["spotId"] == other["id"]
        assert db.get_check("p1")["notes"] == "filed later"

    def test_an_unfiled_check_is_still_stored(self, db):
        """The HTTP API can be called without a spot; that must not lose the check."""
        save(db, "p1", None, 48.0, "2026-01-01T00:00:00+00:00", location="Right Leg")
        rows = db.list_checks()
        assert len(rows) == 1
        assert rows[0]["spotId"] is None
        assert rows[0]["spotLabel"] is None
        assert rows[0]["location"] == "Right Leg"

    def test_list_checks_joins_the_spot_label(self, db):
        spot = db.create_spot("dark mole", "Left Arm")
        save(db, "p1", spot["id"], 48.0, "2026-01-01T00:00:00+00:00")
        row = db.list_checks()[0]
        assert (row["spotLabel"], row["bodyRegion"]) == ("dark mole", "Left Arm")

    def test_list_checks_is_newest_first(self, db):
        spot = db.create_spot("mole", "Back")
        save(db, "p1", spot["id"], 40.0, "2026-01-01T00:00:00+00:00")
        save(db, "p2", spot["id"], 50.0, "2026-02-01T00:00:00+00:00")
        assert [c["processingId"] for c in db.list_checks()] == ["p2", "p1"]


class TestProfile:
    def test_unset_profile_is_none(self, db):
        assert db.get_profile() is None

    def test_save_then_read(self, db):
        db.save_profile(full_name="Callie", fitzpatrick=2, family_history=True)
        profile = db.get_profile()
        assert profile["fullName"] == "Callie"
        assert profile["fitzpatrick"] == 2
        assert profile["familyHistory"] is True
        assert profile["manyMoles"] is False

    def test_location_and_sun_exposure_round_trip(self, db):
        db.save_profile(full_name="Callie", location="Austin, TX", sun_exposure="high")
        profile = db.get_profile()
        assert profile["location"] == "Austin, TX"
        assert profile["sunExposure"] == "high"

    def test_saving_again_replaces_the_single_row(self, db):
        db.save_profile(full_name="First", fitzpatrick=1)
        db.save_profile(full_name="Second", fitzpatrick=5)
        profile = db.get_profile()
        assert profile["fullName"] == "Second"
        assert profile["fitzpatrick"] == 5


class TestPersistence:
    def test_data_survives_reconnecting(self, db):
        """Every call opens its own connection, so this is the restart path."""
        spot = db.create_spot("dark mole", "Left Arm")
        save(db, "p1", spot["id"], 48.0, "2026-01-01T00:00:00+00:00")
        db.save_profile(full_name="Callie")

        assert db.list_spots()[0]["checkCount"] == 1
        assert db.get_profile()["fullName"] == "Callie"
        assert db.get_check("p1")["thumbnail"] == b"THUMB"
