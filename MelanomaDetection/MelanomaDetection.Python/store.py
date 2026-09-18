"""SQLite persistence for tracked spots, saved checks, and each user's risk profile.

Every row belongs to one user_id -- the account id the web app sends with each
request -- and every read here is filtered by it, so one person's data is
never reachable through another person's session even by guessing an id.

Why this exists:
    main.py's _results_store is an in-memory dict, so every saved check and any
    notion of "the same mole over time" vanished when the process restarted.
    Longitudinal tracking (and the Evolving score built on top of it) needs the
    history to outlive a restart, so the durable parts live here instead.

What is deliberately NOT stored here:
    The ten full-size pipeline visuals (filtered images, edge maps, per-criterion
    overlays) stay in main.py's in-memory _results_store. Only the 160px
    thumbnail, the segmentation mask, and the numeric scores are persisted --
    enough to rebuild timelines, trends and change comparisons cheaply. The
    trade-off is that "full breakdown" imagery is only available for checks
    processed during the current server session.
"""

import base64
import contextlib
import datetime
import json
import os
import sqlite3
import uuid

DB_PATH = os.environ.get("SKINCHECK_DB") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "skincheck.db"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spots (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    label        TEXT NOT NULL,
    body_region  TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    archived     INTEGER NOT NULL DEFAULT 0
);

-- spot_id stays nullable: the HTTP API can still be called without a spot (the
-- pre-spots client shape), and an unfiled check is better than a failed save.
CREATE TABLE IF NOT EXISTS checks (
    processing_id TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    spot_id       TEXT REFERENCES spots(id),
    risk_score    REAL NOT NULL,
    diameter_mm   REAL,
    mm_per_px     REAL,
    asymmetry     REAL,
    border        REAL,
    color         REAL,
    -- Lesion area and mean LAB color, so a later check can measure change
    -- without re-decoding and re-segmenting this one's imagery.
    area_px       INTEGER,
    lab_l         REAL,
    lab_a         REAL,
    lab_b         REAL,
    -- Body region as recorded on the check itself. Redundant with spots.body_region
    -- for a filed check, but it is the only location an unfiled one has.
    location      TEXT NOT NULL DEFAULT '',
    symptoms      TEXT NOT NULL DEFAULT '[]',
    notes         TEXT NOT NULL DEFAULT '',
    processed_at  TEXT NOT NULL,
    thumbnail     BLOB,
    mask          BLOB
);

CREATE INDEX IF NOT EXISTS idx_spots_user ON spots(user_id);
CREATE INDEX IF NOT EXISTS idx_checks_user ON checks(user_id, processed_at);
CREATE INDEX IF NOT EXISTS idx_checks_spot ON checks(spot_id, processed_at);

-- One row per user (the web app's account id), not a singleton.
CREATE TABLE IF NOT EXISTS profile (
    user_id             TEXT PRIMARY KEY,
    full_name           TEXT NOT NULL DEFAULT '',
    location            TEXT NOT NULL DEFAULT '',
    sun_exposure        TEXT NOT NULL DEFAULT '',
    fitzpatrick         INTEGER,
    family_history      INTEGER NOT NULL DEFAULT 0,
    blistering_sunburns INTEGER NOT NULL DEFAULT 0,
    many_moles          INTEGER NOT NULL DEFAULT 0,
    updated_at          TEXT NOT NULL
);
"""


@contextlib.contextmanager
def _connect():
    """Open a fresh connection for one unit of work, committing and closing it after.

    A connection per call rather than one shared handle, because Flask's dev
    server serves requests on multiple threads and sqlite3 connections are not
    safe to share across them. sqlite3's own context manager only commits; it
    never closes, and an unclosed handle keeps the file locked on Windows, so
    the close is explicit here.
    """
    connection = sqlite3.connect(DB_PATH)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            yield connection
    finally:
        connection.close()


# Bumped whenever _SCHEMA changes shape. Stored in the database's user_version
# pragma so init_db can tell an old file from a current one.
SCHEMA_VERSION = 2


def init_db():
    """Create the schema if it is missing and bring an older file up to date.

    Called once at process startup (main.py), not per-request. People's spots
    and checks are real records now that every row belongs to a signed-in
    account, so this never discards a current-version database. (Until
    2026-09-18 it wiped the file on every start to keep demos fresh; the demo
    account in the web app now serves that purpose instead.)

    Version 1 files -- the single-user schema with no user_id columns -- are
    recreated rather than migrated: that schema was always wiped at startup,
    so there was never anything in one worth keeping.
    """
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    with _connect() as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if 0 < version < SCHEMA_VERSION:
            _drop_all(connection)
        elif version == 0 and _has_tables(connection):
            # Pre-versioning file (the wiped-on-start era): same treatment.
            _drop_all(connection)
        connection.executescript(_SCHEMA)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def reset_db():
    """Drop every table and recreate the schema. For tests; production never calls this."""
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    with _connect() as connection:
        _drop_all(connection)
        connection.executescript(_SCHEMA)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _has_tables(connection) -> bool:
    return connection.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name IN ('spots', 'checks', 'profile')"
    ).fetchone()[0] > 0


def _drop_all(connection):
    connection.executescript(
        "DROP TABLE IF EXISTS checks; DROP TABLE IF EXISTS spots; DROP TABLE IF EXISTS profile;"
    )


def _utc_now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# --- spots ---------------------------------------------------------------


def create_spot(user_id: str, label: str, body_region: str) -> dict:
    """Register a new tracked spot for a user and return it."""
    spot_id = f"spot_{uuid.uuid4().hex[:12]}"
    created_at = _utc_now_iso()
    with _connect() as connection:
        connection.execute(
            "INSERT INTO spots (id, user_id, label, body_region, created_at) VALUES (?, ?, ?, ?, ?)",
            (spot_id, user_id, label, body_region, created_at),
        )
    return {
        "id": spot_id,
        "label": label,
        "bodyRegion": body_region,
        "createdAt": created_at,
        "archived": False,
    }


def get_spot(user_id: str, spot_id: str):
    """Return one of the user's spots as a dict, or None if it doesn't exist (for them)."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM spots WHERE id = ? AND user_id = ?", (spot_id, user_id)
        ).fetchone()
    return _spot_row_to_dict(row) if row else None


def update_spot(user_id: str, spot_id: str, label=None, archived=None):
    """Rename and/or archive one of the user's spots. Returns the updated spot, or None if missing."""
    assignments = []
    values = []
    if label is not None:
        assignments.append("label = ?")
        values.append(label)
    if archived is not None:
        assignments.append("archived = ?")
        values.append(1 if archived else 0)

    if assignments:
        values.extend([spot_id, user_id])
        with _connect() as connection:
            connection.execute(
                f"UPDATE spots SET {', '.join(assignments)} WHERE id = ? AND user_id = ?", values
            )
    return get_spot(user_id, spot_id)


def list_spots(user_id: str, include_archived: bool = False) -> list:
    """All of a user's spots, each with its aggregates (check count, latest risk, trend).

    Returns:
        A list of spot dicts ordered by most recently checked first, each with
        "checkCount", "lastRiskScore", "lastCheckedAt", "firstRiskScore" and
        "trend" ("up" | "down" | "flat" | None). Callers layer policy-derived
        fields such as the next-due date on top -- see policy.next_due_at.
    """
    query = "SELECT * FROM spots WHERE user_id = ?"
    if not include_archived:
        query += " AND archived = 0"

    with _connect() as connection:
        spot_rows = connection.execute(query, (user_id,)).fetchall()
        check_rows = connection.execute(
            "SELECT spot_id, risk_score, processed_at FROM checks "
            "WHERE user_id = ? AND spot_id IS NOT NULL ORDER BY processed_at ASC",
            (user_id,),
        ).fetchall()

    by_spot = {}
    for row in check_rows:
        by_spot.setdefault(row["spot_id"], []).append(row)

    spots = []
    for row in spot_rows:
        spot = _spot_row_to_dict(row)
        checks = by_spot.get(spot["id"], [])
        spot.update(aggregate_checks(checks))
        spots.append(spot)

    # Unchecked spots sort last; among the rest, most recently checked first.
    spots.sort(key=lambda spot: spot["lastCheckedAt"] or "", reverse=True)
    return spots


def aggregate_checks(checks: list) -> dict:
    """Derive count/latest/trend fields from a spot's checks, oldest first.

    Accepts rows with "risk_score" and "processed_at" keys. Public because
    main.py reuses it to summarize a single spot's timeline and a brand new
    spot with no checks yet.
    """
    if not checks:
        return {
            "checkCount": 0,
            "lastRiskScore": None,
            "lastCheckedAt": None,
            "firstRiskScore": None,
            "trend": None,
        }

    last = checks[-1]
    trend = None
    if len(checks) >= 2:
        delta = last["risk_score"] - checks[-2]["risk_score"]
        # 2 points of risk is below this pipeline's own run-to-run noise, so
        # anything inside that band reads as "flat" rather than a real move.
        trend = "flat" if abs(delta) < 2.0 else ("up" if delta > 0 else "down")

    return {
        "checkCount": len(checks),
        "lastRiskScore": last["risk_score"],
        "lastCheckedAt": last["processed_at"],
        "firstRiskScore": checks[0]["risk_score"],
        "trend": trend,
    }


def _spot_row_to_dict(row) -> dict:
    return {
        "id": row["id"],
        "label": row["label"],
        "bodyRegion": row["body_region"],
        "createdAt": row["created_at"],
        "archived": bool(row["archived"]),
    }


# --- checks --------------------------------------------------------------


def save_check(
    user_id: str,
    processing_id: str,
    spot_id,
    risk_score: float,
    abcde_scores: dict,
    location: str,
    symptoms: list,
    notes: str,
    processed_at: str,
    thumbnail_png: bytes,
    mask_png: bytes,
    mm_per_px=None,
    area_px=None,
    lab=None,
):
    """Persist one analyzed check. Re-saving the same processing_id is a no-op update."""
    diameter = abcde_scores.get("diameter", {}).get("details")
    diameter_mm = diameter.get("diameter_mm") if isinstance(diameter, dict) else None
    lab_l, lab_a, lab_b = lab if lab else (None, None, None)

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO checks (processing_id, user_id, spot_id, risk_score, diameter_mm, mm_per_px,
                                asymmetry, border, color, area_px, lab_l, lab_a, lab_b,
                                location, symptoms, notes, processed_at, thumbnail, mask)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(processing_id) DO UPDATE SET
                spot_id = excluded.spot_id,
                location = excluded.location,
                symptoms = excluded.symptoms,
                notes = excluded.notes
            WHERE checks.user_id = excluded.user_id
            """,
            (
                processing_id,
                user_id,
                spot_id,
                risk_score,
                diameter_mm,
                mm_per_px,
                _score_of(abcde_scores, "asymmetry"),
                _score_of(abcde_scores, "border"),
                _score_of(abcde_scores, "color"),
                area_px,
                lab_l,
                lab_a,
                lab_b,
                location or "",
                json.dumps(symptoms or []),
                notes or "",
                processed_at,
                thumbnail_png,
                mask_png,
            ),
        )


def _score_of(abcde_scores: dict, key: str):
    return abcde_scores.get(key, {}).get("score")


def get_last_check(user_id: str, spot_id: str):
    """The most recent stored check for one of the user's spots, or None.

    Called while processing a new photo -- before that photo is saved -- so this
    returns the prior check the new one should be compared against.
    """
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM checks WHERE user_id = ? AND spot_id = ? ORDER BY processed_at DESC LIMIT 1",
            (user_id, spot_id),
        ).fetchone()
    return _check_row_to_dict(row) if row else None


def get_check(user_id: str, processing_id: str):
    """One of the user's stored checks by its processing id, or None."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM checks WHERE processing_id = ? AND user_id = ?", (processing_id, user_id)
        ).fetchone()
    return _check_row_to_dict(row) if row else None


def get_checks_for_spot(user_id: str, spot_id: str) -> list:
    """Every stored check for one of the user's spots, oldest first (timeline order)."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM checks WHERE user_id = ? AND spot_id = ? ORDER BY processed_at ASC",
            (user_id, spot_id),
        ).fetchall()
    return [_check_row_to_dict(row) for row in rows]


def list_checks(user_id: str) -> list:
    """Every stored check of the user's across all spots, newest first (the "All checks" list)."""
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT checks.*, spots.label AS spot_label, spots.body_region AS spot_region
            FROM checks LEFT JOIN spots ON spots.id = checks.spot_id
            WHERE checks.user_id = ?
            ORDER BY checks.processed_at DESC
            """,
            (user_id,),
        ).fetchall()

    checks = []
    for row in rows:
        check = _check_row_to_dict(row)
        check["spotLabel"] = row["spot_label"]
        check["bodyRegion"] = row["spot_region"]
        checks.append(check)
    return checks


def _lab_of(row):
    """The stored mean lesion LAB as a 3-tuple, or None if it wasn't recorded."""
    if row["lab_l"] is None:
        return None
    return (row["lab_l"], row["lab_a"], row["lab_b"])


def _check_row_to_dict(row) -> dict:
    return {
        "processingId": row["processing_id"],
        "spotId": row["spot_id"],
        "riskScore": row["risk_score"],
        "diameterMm": row["diameter_mm"],
        "mmPerPx": row["mm_per_px"],
        "asymmetry": row["asymmetry"],
        "border": row["border"],
        "color": row["color"],
        "areaPx": row["area_px"],
        "lab": _lab_of(row),
        "location": row["location"],
        "symptoms": json.loads(row["symptoms"] or "[]"),
        "notes": row["notes"],
        "processedAt": row["processed_at"],
        "thumbnail": row["thumbnail"],
        "mask": row["mask"],
    }


# --- profile -------------------------------------------------------------


def get_profile(user_id: str):
    """The user's risk-profile row, or None if they haven't filled it in yet."""
    with _connect() as connection:
        row = connection.execute("SELECT * FROM profile WHERE user_id = ?", (user_id,)).fetchone()
    if row is None:
        return None
    return {
        "fullName": row["full_name"],
        "location": row["location"],
        "sunExposure": row["sun_exposure"],
        "fitzpatrick": row["fitzpatrick"],
        "familyHistory": bool(row["family_history"]),
        "blisteringSunburns": bool(row["blistering_sunburns"]),
        "manyMoles": bool(row["many_moles"]),
        "updatedAt": row["updated_at"],
    }


def save_profile(
    user_id: str,
    full_name: str = "",
    location: str = "",
    sun_exposure: str = "",
    fitzpatrick=None,
    family_history: bool = False,
    blistering_sunburns: bool = False,
    many_moles: bool = False,
) -> dict:
    """Insert or replace the user's risk profile, and return it."""
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO profile (user_id, full_name, location, sun_exposure, fitzpatrick,
                                 family_history, blistering_sunburns, many_moles, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                full_name = excluded.full_name,
                location = excluded.location,
                sun_exposure = excluded.sun_exposure,
                fitzpatrick = excluded.fitzpatrick,
                family_history = excluded.family_history,
                blistering_sunburns = excluded.blistering_sunburns,
                many_moles = excluded.many_moles,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                full_name or "",
                location or "",
                sun_exposure or "",
                fitzpatrick,
                1 if family_history else 0,
                1 if blistering_sunburns else 0,
                1 if many_moles else 0,
                _utc_now_iso(),
            ),
        )
    return get_profile(user_id)


# --- whole-account operations ---------------------------------------------


def export_user_data(user_id: str) -> dict:
    """Everything stored for one user, as plain JSON-ready data.

    Backs the "Export my data" download. Thumbnails are included (base64) since
    they are the person's own photos; segmentation masks are an internal
    artefact and are left out.
    """
    spots = list_spots(user_id, include_archived=True)
    checks = []
    for check in list_checks(user_id):
        thumbnail = check.pop("thumbnail")
        check.pop("mask")
        check["thumbnailPng"] = base64.b64encode(thumbnail).decode("ascii") if thumbnail else None
        checks.append(check)

    return {
        "profile": get_profile(user_id),
        "spots": spots,
        "checks": checks,
    }


def delete_user_data(user_id: str) -> dict:
    """Erase every row the user owns. Returns the number removed per table."""
    with _connect() as connection:
        checks = connection.execute("DELETE FROM checks WHERE user_id = ?", (user_id,)).rowcount
        spots = connection.execute("DELETE FROM spots WHERE user_id = ?", (user_id,)).rowcount
        profile = connection.execute("DELETE FROM profile WHERE user_id = ?", (user_id,)).rowcount
    return {"checks": checks, "spots": spots, "profile": profile}
