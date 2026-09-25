"""Spot/check/profile persistence.

The point of this module is that history outlives a restart, so these tests
drive it through real connections against a temporary database file rather than
mocking sqlite3 out.
"""

import pytest

import store

# Every row is owned by a user; these tests act as one and, where it matters,
# check that a second one can't see through the wall.
USER = "user-a"
OTHER = "user-b"


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
        user_id=kwargs.get("user_id", USER),
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
        spot = db.create_spot(USER, "dark mole", "Left Arm")
        assert spot["id"].startswith("spot_")
        assert db.get_spot(USER, spot["id"])["label"] == "dark mole"

    def test_missing_spot_is_none(self, db):
        assert db.get_spot(USER, "spot_nope") is None

    def test_rename(self, db):
        spot = db.create_spot(USER, "mole", "Back")
        assert db.update_spot(USER, spot["id"], label="mole (watch)")["label"] == "mole (watch)"

    def test_archived_spots_are_hidden_by_default(self, db):
        spot = db.create_spot(USER, "mole", "Back")
        db.update_spot(USER, spot["id"], archived=True)
        assert db.list_spots(USER) == []
        assert len(db.list_spots(USER, include_archived=True)) == 1

    def test_update_with_nothing_to_change_is_harmless(self, db):
        spot = db.create_spot(USER, "mole", "Back")
        assert db.update_spot(USER, spot["id"])["label"] == "mole"


class TestAggregates:
    def test_a_spot_with_no_checks(self, db):
        db.create_spot(USER, "mole", "Back")
        summary = db.list_spots(USER)[0]
        assert summary["checkCount"] == 0
        assert summary["lastRiskScore"] is None
        assert summary["trend"] is None

    def test_trend_needs_two_checks(self, db):
        spot = db.create_spot(USER, "mole", "Back")
        save(db, "p1", spot["id"], 40.0, "2026-01-01T00:00:00+00:00")
        assert db.list_spots(USER)[0]["trend"] is None

    def test_rising_and_falling_trend(self, db):
        spot = db.create_spot(USER, "mole", "Back")
        save(db, "p1", spot["id"], 40.0, "2026-01-01T00:00:00+00:00")
        save(db, "p2", spot["id"], 55.0, "2026-02-01T00:00:00+00:00")
        summary = db.list_spots(USER)[0]
        assert (summary["firstRiskScore"], summary["lastRiskScore"]) == (40.0, 55.0)
        assert summary["trend"] == "up"

        save(db, "p3", spot["id"], 41.0, "2026-03-01T00:00:00+00:00")
        assert db.list_spots(USER)[0]["trend"] == "down"

    def test_small_movement_reads_as_flat(self, db):
        """Under this pipeline's own run-to-run noise, so not a real move."""
        spot = db.create_spot(USER, "mole", "Back")
        save(db, "p1", spot["id"], 50.0, "2026-01-01T00:00:00+00:00")
        save(db, "p2", spot["id"], 51.0, "2026-02-01T00:00:00+00:00")
        assert db.list_spots(USER)[0]["trend"] == "flat"

    def test_ordering_is_by_recency_of_last_check(self, db):
        stale = db.create_spot(USER, "stale", "Back")
        fresh = db.create_spot(USER, "fresh", "Left Arm")
        save(db, "p1", stale["id"], 40.0, "2026-01-01T00:00:00+00:00")
        save(db, "p2", fresh["id"], 40.0, "2026-06-01T00:00:00+00:00")
        assert [s["label"] for s in db.list_spots(USER)] == ["fresh", "stale"]


class TestChecks:
    def test_timeline_is_oldest_first(self, db):
        spot = db.create_spot(USER, "mole", "Back")
        save(db, "p2", spot["id"], 50.0, "2026-02-01T00:00:00+00:00")
        save(db, "p1", spot["id"], 40.0, "2026-01-01T00:00:00+00:00")
        assert [c["processingId"] for c in db.get_checks_for_spot(USER, spot["id"])] == ["p1", "p2"]

    def test_last_check_is_the_most_recent(self, db):
        spot = db.create_spot(USER, "mole", "Back")
        save(db, "p1", spot["id"], 40.0, "2026-01-01T00:00:00+00:00")
        save(db, "p2", spot["id"], 50.0, "2026-02-01T00:00:00+00:00")
        assert db.get_last_check(USER, spot["id"])["processingId"] == "p2"

    def test_last_check_of_an_unchecked_spot_is_none(self, db):
        spot = db.create_spot(USER, "mole", "Back")
        assert db.get_last_check(USER, spot["id"]) is None

    def test_binary_and_measurement_fields_round_trip(self, db):
        spot = db.create_spot(USER, "mole", "Back")
        save(
            db, "p1", spot["id"], 48.0, "2026-01-01T00:00:00+00:00",
            thumbnail=b"THUMBNAIL-BYTES", mask=b"MASK-BYTES",
            mm_per_px=0.05, area_px=2800, lab=(135.0, 153.0, 132.0),
            symptoms=["Itchy", "Growing"], notes="watch this",
        )
        check = db.get_check(USER, "p1")
        assert check["thumbnail"] == b"THUMBNAIL-BYTES"
        assert check["mask"] == b"MASK-BYTES"
        assert check["mmPerPx"] == 0.05
        assert check["areaPx"] == 2800
        assert check["lab"] == (135.0, 153.0, 132.0)
        assert check["symptoms"] == ["Itchy", "Growing"]
        assert check["notes"] == "watch this"
        assert check["diameterMm"] == 4.2  # lifted out of the diameter details

    def test_lab_is_none_when_not_recorded(self, db):
        spot = db.create_spot(USER, "mole", "Back")
        save(db, "p1", spot["id"], 48.0, "2026-01-01T00:00:00+00:00", lab=None)
        assert db.get_check(USER, "p1")["lab"] is None

    def test_resaving_updates_rather_than_duplicating(self, db):
        db.create_spot(USER, "mole", "Back")
        other = db.create_spot(USER, "other", "Left Arm")
        save(db, "p1", None, 48.0, "2026-01-01T00:00:00+00:00")
        save(db, "p1", other["id"], 48.0, "2026-01-01T00:00:00+00:00", notes="filed later")

        assert len(db.list_checks(USER)) == 1
        assert db.get_check(USER, "p1")["spotId"] == other["id"]
        assert db.get_check(USER, "p1")["notes"] == "filed later"

    def test_an_unfiled_check_is_still_stored(self, db):
        """The HTTP API can be called without a spot; that must not lose the check."""
        save(db, "p1", None, 48.0, "2026-01-01T00:00:00+00:00", location="Right Leg")
        rows = db.list_checks(USER)
        assert len(rows) == 1
        assert rows[0]["spotId"] is None
        assert rows[0]["spotLabel"] is None
        assert rows[0]["location"] == "Right Leg"

    def test_list_checks_joins_the_spot_label(self, db):
        spot = db.create_spot(USER, "dark mole", "Left Arm")
        save(db, "p1", spot["id"], 48.0, "2026-01-01T00:00:00+00:00")
        row = db.list_checks(USER)[0]
        assert (row["spotLabel"], row["bodyRegion"]) == ("dark mole", "Left Arm")

    def test_list_checks_is_newest_first(self, db):
        spot = db.create_spot(USER, "mole", "Back")
        save(db, "p1", spot["id"], 40.0, "2026-01-01T00:00:00+00:00")
        save(db, "p2", spot["id"], 50.0, "2026-02-01T00:00:00+00:00")
        assert [c["processingId"] for c in db.list_checks(USER)] == ["p2", "p1"]


class TestProfile:
    def test_unset_profile_is_none(self, db):
        assert db.get_profile(USER) is None

    def test_save_then_read(self, db):
        db.save_profile(USER, full_name="Callie", fitzpatrick=2, family_history=True)
        profile = db.get_profile(USER)
        assert profile["fullName"] == "Callie"
        assert profile["fitzpatrick"] == 2
        assert profile["familyHistory"] is True
        assert profile["manyMoles"] is False

    def test_location_and_sun_exposure_round_trip(self, db):
        db.save_profile(USER, full_name="Callie", location="Austin, TX", sun_exposure="high")
        profile = db.get_profile(USER)
        assert profile["location"] == "Austin, TX"
        assert profile["sunExposure"] == "high"

    def test_saving_again_replaces_the_single_row(self, db):
        db.save_profile(USER, full_name="First", fitzpatrick=1)
        db.save_profile(USER, full_name="Second", fitzpatrick=5)
        profile = db.get_profile(USER)
        assert profile["fullName"] == "Second"
        assert profile["fitzpatrick"] == 5


class TestPersistence:
    def test_init_db_keeps_existing_data(self, db):
        """A restart must not lose anyone's spots (it used to wipe the file on purpose)."""
        spot = db.create_spot(USER, "dark mole", "Left Arm")
        db.init_db()
        assert db.get_spot(USER, spot["id"]) is not None

    def test_init_db_recreates_a_pre_users_schema(self, db, tmp_path):
        """A v1 file (no user_id columns) is replaced, not migrated -- it never held kept data."""
        import sqlite3

        path = str(tmp_path / "old.db")
        with sqlite3.connect(path) as old:
            old.executescript("CREATE TABLE spots (id TEXT PRIMARY KEY, label TEXT NOT NULL);")
        db.DB_PATH = path
        db.init_db()
        assert db.list_spots(USER) == []
        assert db.create_spot(USER, "mole", "Back")["id"].startswith("spot_")

    def test_data_survives_reconnecting(self, db):
        """Every call opens its own connection, so this is the restart path."""
        spot = db.create_spot(USER, "dark mole", "Left Arm")
        save(db, "p1", spot["id"], 48.0, "2026-01-01T00:00:00+00:00")
        db.save_profile(USER, full_name="Callie")

        assert db.list_spots(USER)[0]["checkCount"] == 1
        assert db.get_profile(USER)["fullName"] == "Callie"
        assert db.get_check(USER, "p1")["thumbnail"] == b"THUMB"


class TestUserIsolation:
    """The same ids, asked for by a different user, must come back empty."""

    def test_spots_are_invisible_to_other_users(self, db):
        spot = db.create_spot(USER, "mole", "Back")
        assert db.get_spot(OTHER, spot["id"]) is None
        assert db.list_spots(OTHER) == []
        assert db.update_spot(OTHER, spot["id"], label="hijacked") is None
        assert db.get_spot(USER, spot["id"])["label"] == "mole"

    def test_checks_are_invisible_to_other_users(self, db):
        spot = db.create_spot(USER, "mole", "Back")
        save(db, "p1", spot["id"], 40.0, "2026-01-01T00:00:00+00:00")
        assert db.get_check(OTHER, "p1") is None
        assert db.get_last_check(OTHER, spot["id"]) is None
        assert db.get_checks_for_spot(OTHER, spot["id"]) == []
        assert db.list_checks(OTHER) == []

    def test_resaving_someone_elses_check_changes_nothing(self, db):
        spot = db.create_spot(USER, "mole", "Back")
        save(db, "p1", spot["id"], 40.0, "2026-01-01T00:00:00+00:00", notes="mine")
        save(db, "p1", None, 40.0, "2026-01-01T00:00:00+00:00", notes="theirs", user_id=OTHER)
        assert db.get_check(USER, "p1")["notes"] == "mine"
        assert db.get_check(OTHER, "p1") is None

    def test_profiles_are_per_user(self, db):
        db.save_profile(USER, full_name="A", fitzpatrick=1)
        db.save_profile(OTHER, full_name="B", fitzpatrick=5)
        assert db.get_profile(USER)["fullName"] == "A"
        assert db.get_profile(OTHER)["fullName"] == "B"


class TestAccountOperations:
    def test_export_includes_everything_and_no_masks(self, db):
        spot = db.create_spot(USER, "mole", "Back")
        db.update_spot(USER, spot["id"], archived=True)
        save(db, "p1", spot["id"], 40.0, "2026-01-01T00:00:00+00:00", thumbnail=b"PNG", mask=b"MASK")
        db.save_profile(USER, full_name="Callie")

        export = db.export_user_data(USER)
        assert export["profile"]["fullName"] == "Callie"
        assert [s["id"] for s in export["spots"]] == [spot["id"]]  # archived spots included
        assert export["checks"][0]["thumbnailPng"] == "UE5H"
        assert "mask" not in export["checks"][0]
        assert "thumbnail" not in export["checks"][0]

    def test_export_of_an_empty_account(self, db):
        assert db.export_user_data(USER) == {"profile": None, "spots": [], "checks": []}

    def test_delete_removes_only_that_user(self, db):
        mine = db.create_spot(USER, "mole", "Back")
        theirs = db.create_spot(OTHER, "mole", "Back")
        save(db, "p1", mine["id"], 40.0, "2026-01-01T00:00:00+00:00")
        save(db, "p2", theirs["id"], 40.0, "2026-01-01T00:00:00+00:00", user_id=OTHER)
        db.save_profile(USER, full_name="A")
        db.save_profile(OTHER, full_name="B")

        assert db.delete_user_data(USER) == {"checks": 1, "spots": 1, "profile": 1}
        assert db.export_user_data(USER) == {"profile": None, "spots": [], "checks": []}
        assert db.get_spot(OTHER, theirs["id"]) is not None
        assert db.get_check(OTHER, "p2") is not None
        assert db.get_profile(OTHER)["fullName"] == "B"
