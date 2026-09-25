import io

import cv2
import numpy as np
import pytest
from pypdf import PdfReader

import policy
import report
import store

USER = "user-a"
OTHER = "user-b"


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "test.db"))
    store.init_db()
    return store


def _png(image):
    return cv2.imencode(".png", image)[1].tobytes()


def _photo_and_mask(width=160):
    photo = np.full((width * 3 // 4, width, 3), (150, 180, 215), np.uint8)
    mask = np.zeros(photo.shape[:2], np.uint8)
    center = (width // 2, width * 3 // 8)
    cv2.ellipse(photo, center, (width // 5, width // 7), 20, 0, 360, (40, 60, 90), -1)
    cv2.ellipse(mask, center, (width // 5, width // 7), 20, 0, 360, 255, -1)
    return photo, mask


def _save(user, processing_id, spot_id, processed_at, concern=policy.CONCERN_LOWER, risk=30.0, notes="", with_visuals=False):
    photo, mask = _photo_and_mask()
    store.save_check(
        user_id=user, processing_id=processing_id, spot_id=spot_id, risk_score=risk,
        abcde_scores={
            "asymmetry": {"score": 4.2}, "border": {"score": 2.1}, "color": {"score": 7.3},
            "diameter": {"score": 5.0, "details": {"diameter_mm": 6.8}},
        },
        location="Left Arm", symptoms=["itching"], notes=notes, processed_at=processed_at,
        thumbnail_png=_png(photo), mask_png=_png(mask), overall_visual_concern=concern,
    )
    if with_visuals:
        big, _ = _photo_and_mask(640)
        store.save_check_visuals(user, processing_id, {
            kind: cv2.imencode(".jpg", big)[1].tobytes() for kind in store.VISUAL_KINDS
        })


def _pages(pdf_bytes):
    return [page.extract_text() for page in PdfReader(io.BytesIO(pdf_bytes)).pages]


def test_cover_then_one_page_per_checked_spot_with_the_elevated_one_first(db):
    store.save_profile(USER, full_name="Pat Patient", fitzpatrick=2, family_history=True)
    arm = store.create_spot(USER, "Forearm mole", "Left Arm")
    back = store.create_spot(USER, "Back freckle", "Upper Back")
    store.create_spot(USER, "Never checked", "Leg")
    _save(USER, "c1", arm["id"], "2026-09-01T10:00:00+00:00", with_visuals=True)
    _save(USER, "c2", arm["id"], "2026-09-20T10:00:00+00:00", with_visuals=True, notes="Darker than last month")
    _save(USER, "c3", back["id"], "2026-08-01T10:00:00+00:00", concern=policy.CONCERN_ELEVATED, risk=70.0)

    pdf = report.build_report(USER)
    pages = _pages(pdf)

    assert pdf.startswith(b"%PDF")
    assert len(pages) == 3
    assert "Pat Patient" in pages[0]
    assert "Fitzpatrick 2" in pages[0]
    assert "Never checked" in pages[0] and "No checks yet" in pages[0]
    # Elevated first even though it is older.
    assert "Back freckle" in pages[1] and "Elevated visual concern" in pages[1]
    assert "Forearm mole" in pages[2]
    assert "A - Asymmetry" in pages[2] and "D - Diameter" in pages[2]
    assert "6.8 mm (over 6 mm)" in pages[2]
    assert "Darker than last month" in pages[2]


def test_a_check_without_stored_images_falls_back_to_its_photo_and_outline(db):
    spot = store.create_spot(USER, "Old spot", "Neck")
    _save(USER, "old", spot["id"], "2026-01-01T10:00:00+00:00")

    pages = _pages(report.build_report(USER, fallback_name="From Account"))

    assert "From Account" in pages[0]
    assert "only its photo and outline" in pages[1].replace("\n", " ")


def test_cover_page_numbers_follow_spots_that_run_onto_a_second_page(db):
    long = store.create_spot(USER, "Long history", "Arm")
    short = store.create_spot(USER, "Short history", "Leg")
    for day in range(1, 29):
        _save(USER, f"l{day}", long["id"], f"2026-08-{day:02d}T10:00:00+00:00", with_visuals=day == 28)
    _save(USER, "s1", short["id"], "2026-07-01T10:00:00+00:00")

    pages = _pages(report.build_report(USER))
    short_page = next(index for index, text in enumerate(pages, start=1) if "Short history" in text and index > 1)

    assert short_page > 3  # "Long history" took more than one page
    cover_row = next(line for line in pages[0].splitlines() if line.startswith("Short history"))
    assert cover_row.rstrip().endswith(str(short_page))


def test_only_the_callers_data_is_included_and_odd_characters_dont_break_it(db):
    store.save_profile(USER, full_name="Zoë “Z” Nguyễn \U0001F600")
    mine = store.create_spot(USER, "Mine", "Arm")
    theirs = store.create_spot(OTHER, "Theirs", "Arm")
    _save(USER, "m", mine["id"], "2026-09-01T10:00:00+00:00", notes="Looks — different")
    _save(OTHER, "t", theirs["id"], "2026-09-01T10:00:00+00:00")

    text = "\n".join(_pages(report.build_report(USER)))

    assert "Zoë" in text and '"Z"' in text
    assert "Mine" in text and "Theirs" not in text


def test_an_empty_account_still_gets_a_cover_page(db):
    pages = _pages(report.build_report(USER, fallback_name="New Person"))

    assert len(pages) == 1
    assert "New Person" in pages[0] and "No spots or checks have been saved yet." in pages[0]


def test_deleting_an_account_removes_its_stored_images(db):
    spot = store.create_spot(USER, "Spot", "Arm")
    _save(USER, "v", spot["id"], "2026-09-01T10:00:00+00:00", with_visuals=True)

    store.delete_user_data(USER)

    assert store.get_check_visuals(USER, "v") is None
