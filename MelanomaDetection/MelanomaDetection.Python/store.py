"""SQLite persistence for tracked spots, saved checks, and the user's risk profile.

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
    label        TEXT NOT NULL,
    body_region  TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    archived     INTEGER NOT NULL DEFAULT 0
);

-- spot_id stays nullable: the HTTP API can still be called without a spot (the
-- pre-spots client shape), and an unfiled check is better than a failed save.
CREATE TABLE IF NOT EXISTS checks (
    processing_id TEXT PRIMARY KEY,
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

CREATE INDEX IF NOT EXISTS idx_checks_spot ON checks(spot_id, processed_at);

CREATE TABLE IF NOT EXISTS profile (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
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


def _connect():
    """Open a fresh connection.

    A connection per call rather than one shared handle, because Flask's dev
    server serves requests on multiple threads and sqlite3 connections are not
    safe to share across them.
    """
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db():
    """Wipe and recreate the schema, so every server start is a clean slate.

    Called once at process startup (main.py), not per-request. Tracked spots,
    checks and the risk profile are meant to survive within a running session
    but deliberately NOT across restarts -- that keeps the onboarding flow
    (empty state -> risk profile -> first spot) demoable on demand instead of
    accumulating leftover data from earlier runs.
    """
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    with _connect() as connection:
        connection.executescript(_SCHEMA)
        _migrate_profile_columns(connection)


def _migrate_profile_columns(connection):
    """Add profile columns introduced after a DB already existed.

    CREATE TABLE IF NOT EXISTS in _SCHEMA only helps a brand-new database --
    an existing skincheck.db from before these columns were added needs them
    backfilled by hand.
    """
    existing = {row["name"] for row in connection.execute("PRAGMA table_info(profile)")}
    for column in ("location", "sun_exposure"):
        if column not in existing:
            connection.execute(f"ALTER TABLE profile ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")


def _utc_now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# --- spots ---------------------------------------------------------------


def create_spot(label: str, body_region: str) -> dict:
    """Register a new tracked spot and return it."""
    spot_id = f"spot_{uuid.uuid4().hex[:12]}"
    created_at = _utc_now_iso()
    with _connect() as connection:
        connection.execute(
            "INSERT INTO spots (id, label, body_region, created_at) VALUES (?, ?, ?, ?)",
            (spot_id, label, body_region, created_at),
        )
    return {
        "id": spot_id,
        "label": label,
        "bodyRegion": body_region,
        "createdAt": created_at,
        "archived": False,
    }


def get_spot(spot_id: str):
    """Return one spot as a dict, or None if it doesn't exist."""
    with _connect() as connection:
        row = connection.execute("SELECT * FROM spots WHERE id = ?", (spot_id,)).fetchone()
    return _spot_row_to_dict(row) if row else None


def update_spot(spot_id: str, label=None, archived=None):
    """Rename and/or archive a spot. Returns the updated spot, or None if missing."""
    assignments = []
    values = []
    if label is not None:
        assignments.append("label = ?")
        values.append(label)
    if archived is not None:
        assignments.append("archived = ?")
        values.append(1 if archived else 0)

    if assignments:
        values.append(spot_id)
        with _connect() as connection:
            connection.execute(
                f"UPDATE spots SET {', '.join(assignments)} WHERE id = ?", values
            )
    return get_spot(spot_id)


def list_spots(include_archived: bool = False) -> list:
    """All spots, each with its aggregates (check count, latest risk, trend).

    Returns:
        A list of spot dicts ordered by most recently checked first, each with
        "checkCount", "lastRiskScore", "lastCheckedAt", "firstRiskScore" and
        "trend" ("up" | "down" | "flat" | None). Callers layer policy-derived
        fields such as the next-due date on top -- see policy.next_due_at.
    """
    query = "SELECT * FROM spots"
    if not include_archived:
        query += " WHERE archived = 0"

    with _connect() as connection:
        spot_rows = connection.execute(query).fetchall()
        check_rows = connection.execute(
            "SELECT spot_id, risk_score, processed_at FROM checks "
            "WHERE spot_id IS NOT NULL ORDER BY processed_at ASC"
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
            INSERT INTO checks (processing_id, spot_id, risk_score, diameter_mm, mm_per_px,
                                asymmetry, border, color, area_px, lab_l, lab_a, lab_b,
                                location, symptoms, notes, processed_at, thumbnail, mask)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(processing_id) DO UPDATE SET
                spot_id = excluded.spot_id,
                location = excluded.location,
                symptoms = excluded.symptoms,
                notes = excluded.notes
            """,
            (
                processing_id,
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


def get_last_check(spot_id: str):
    """The most recent stored check for a spot, or None.

    Called while processing a new photo -- before that photo is saved -- so this
    returns the prior check the new one should be compared against.
    """
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM checks WHERE spot_id = ? ORDER BY processed_at DESC LIMIT 1",
            (spot_id,),
        ).fetchone()
    return _check_row_to_dict(row) if row else None


def get_check(processing_id: str):
    """One stored check by its processing id, or None."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM checks WHERE processing_id = ?", (processing_id,)
        ).fetchone()
    return _check_row_to_dict(row) if row else None


def get_checks_for_spot(spot_id: str) -> list:
    """Every stored check for one spot, oldest first (timeline order)."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM checks WHERE spot_id = ? ORDER BY processed_at ASC",
            (spot_id,),
        ).fetchall()
    return [_check_row_to_dict(row) for row in rows]


def list_checks() -> list:
    """Every stored check across all spots, newest first (the "All checks" list)."""
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT checks.*, spots.label AS spot_label, spots.body_region AS spot_region
            FROM checks LEFT JOIN spots ON spots.id = checks.spot_id
            ORDER BY checks.processed_at DESC
            """
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


def get_profile():
    """The single risk-profile row, or None if the user hasn't filled it in yet."""
    with _connect() as connection:
        row = connection.execute("SELECT * FROM profile WHERE id = 1").fetchone()
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
    full_name: str = "",
    location: str = "",
    sun_exposure: str = "",
    fitzpatrick=None,
    family_history: bool = False,
    blistering_sunburns: bool = False,
    many_moles: bool = False,
) -> dict:
    """Insert or replace the risk profile, and return it."""
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO profile (id, full_name, location, sun_exposure, fitzpatrick,
                                 family_history, blistering_sunburns, many_moles, updated_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
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
    return get_profile()
