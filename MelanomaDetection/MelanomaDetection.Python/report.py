"""The PDF skin-check report: what "Export my data" downloads and what a patient shares with a dermatologist.

A cover page (the person's name, risk profile and a table of every spot), then
one page per spot that has checks: its latest result, the A/B/C/D scores, the
four ABCD evidence overlays, and the spot's check history. Spots with an
elevated result come first so a reader sees them before anything else.

Checks saved before store.check_visuals existed have no stored overlays; their
page shows the photo with the lesion outline redrawn from the saved mask instead.

Uses fpdf2's built-in Helvetica, which only covers Latin-1, so every string
goes through _t() first rather than risking an encoding error on a stray emoji
or a curly quote in someone's notes.
"""

import datetime
import io

import cv2
import numpy as np
from fpdf import FPDF, FontFace
from fpdf.enums import TableCellFillMode, XPos, YPos
from PIL import Image

import policy
import store

# The web app's light theme (wwwroot/app.css), so the report reads as the same product.
NAV = (42, 59, 49)
ACCENT = (47, 74, 60)
ACCENT_SOFT = (221, 228, 220)
ALERT_INK = (83, 30, 26)
ALERT_SOFT = (248, 227, 223)
WARN_INK = (65, 43, 6)
WARN_SOFT = (246, 234, 214)
WELL = (234, 228, 211)
TEXT = (31, 41, 55)
MUTED = (84, 96, 108)
WHITE = (255, 255, 255)

VISUAL_CAPTIONS = {
    "asymmetry": ("A - Asymmetry", "Lesion mirrored on itself: green overlaps, red and blue don't."),
    "border": ("B - Border", "Detected edge. Red means the border crossed its concern threshold."),
    "color": ("C - Color", "Distance from surrounding skin tone: blue is close, red is far."),
    "diameter": ("D - Diameter", "Smallest circle around the lesion, with its measured size."),
}

_REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
    "…": "...", "•": "-", " ": " ", "→": "->",
}


def _t(text) -> str:
    """Text the built-in fonts can draw: common punctuation mapped, anything else outside Latin-1 replaced."""
    text = "" if text is None else str(text)
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def _parse_time(value):
    try:
        parsed = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=datetime.timezone.utc)


def _date(value) -> str:
    parsed = _parse_time(value)
    return f"{parsed:%b} {parsed.day}, {parsed.year}" if parsed else "Unknown date"


def _result_label(check) -> str:
    concern = check.get("overallVisualConcern")
    if concern == policy.CONCERN_NO_DETECTION:
        return "No spot detected"
    if concern in policy.CONCERN_LABEL:
        return policy.CONCERN_LABEL[concern]
    score = check.get("riskScore")
    if score is None:
        return "Not scored"
    band = "low" if score <= policy.BAND_LOW_MAX else "moderate" if score <= policy.BAND_MODERATE_MAX else "high"
    return policy.BAND_LABEL[band]


def _result_colors(check):
    concern = check.get("overallVisualConcern")
    if concern == policy.CONCERN_ELEVATED:
        return ALERT_SOFT, ALERT_INK
    if concern == policy.CONCERN_LOWER:
        return ACCENT_SOFT, ACCENT
    if concern == policy.CONCERN_NO_DETECTION or check.get("riskScore") is None:
        return WELL, TEXT
    score = check["riskScore"]
    if score > policy.BAND_MODERATE_MAX:
        return ALERT_SOFT, ALERT_INK
    if score > policy.BAND_LOW_MAX:
        return WARN_SOFT, WARN_INK
    return ACCENT_SOFT, ACCENT


def _factor(score) -> str:
    """A 0-10 factor score with the web app's band word (RiskBands.FactorBand)."""
    if score is None:
        return "N/A"
    band = "Low" if score < 3.3 else "Moderate" if score < 6.6 else "High"
    return f"{score:.1f} / 10 ({band})"


def _diameter(mm) -> str:
    if mm is None:
        return "N/A"
    return f"{mm:.1f} mm" + (" (over 6 mm)" if mm > 6 else "")


def _is_elevated(check) -> bool:
    return check.get("overallVisualConcern") == policy.CONCERN_ELEVATED


def _outline_on_thumbnail(check):
    """PNG of the saved thumbnail with the saved mask's outline drawn on it, or None."""
    if not check.get("thumbnail"):
        return None
    photo = cv2.imdecode(np.frombuffer(check["thumbnail"], np.uint8), cv2.IMREAD_COLOR)
    if photo is None:
        return None
    if check.get("mask"):
        mask = cv2.imdecode(np.frombuffer(check["mask"], np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, (photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(photo, contours, -1, (80, 255, 80), 1)
    success, buffer = cv2.imencode(".png", photo)
    return buffer.tobytes() if success else None


class _Report(FPDF):
    def __init__(self, patient_name: str):
        super().__init__(orientation="portrait", unit="mm", format="Letter")
        self.patient_name = patient_name
        self.set_margins(18, 16, 18)
        self.set_auto_page_break(True, margin=20)
        self.set_title(_t(f"Skin check report - {patient_name}"))
        self.set_author("LA Spot")
        self.set_creator("LA Spot")

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 5, _t(f"{self.patient_name} - Skin check report"), align="L")
        self.set_x(self.l_margin)
        self.cell(0, 5, "Screening tool only - not a diagnosis", align="C")
        self.set_x(self.l_margin)
        self.cell(0, 5, f"Page {self.page_no()} of {{nb}}", align="R")

    # --- small drawing helpers ---------------------------------------------

    def heading(self, text, size=14, space_before=4):
        self.ln(space_before)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*ACCENT)
        self.cell(0, size * 0.5, _t(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)
        self.set_text_color(*TEXT)

    def paragraph(self, text, size=10, color=TEXT):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*color)
        self.multi_cell(0, size * 0.5, _t(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*TEXT)

    def pill(self, text, fill, ink):
        self.set_font("Helvetica", "B", 10)
        width = self.get_string_width(_t(text)) + 8
        self.set_fill_color(*fill)
        self.set_text_color(*ink)
        self.cell(width, 7, _t(text), fill=True, align="C")
        self.set_text_color(*TEXT)

    def picture(self, data: bytes, x, y, max_w, max_h):
        """Place an image scaled to fit max_w x max_h; returns the height it used."""
        with Image.open(io.BytesIO(data)) as probe:
            px_w, px_h = probe.size
        scale = min(max_w / px_w, max_h / px_h)
        w, h = px_w * scale, px_h * scale
        self.image(io.BytesIO(data), x=x + (max_w - w) / 2, y=y, w=w, h=h)
        return h

    def table_style(self, **overrides):
        # Unfilled rows are painted with the current fill color, so reset it from
        # whatever was drawn last (the cover band, a pill) to plain white.
        self.set_fill_color(*WHITE)
        style = dict(
            borders_layout="HORIZONTAL_LINES",
            headings_style=FontFace(emphasis="BOLD", color=WHITE, fill_color=ACCENT),
            cell_fill_color=(247, 244, 236),
            cell_fill_mode=TableCellFillMode.ROWS,
            line_height=6,
            padding=1.5,
        )
        style.update(overrides)
        return style


def _cover(pdf: _Report, profile, spot_rows, shared_with: str):
    pdf.add_page()
    pdf.set_fill_color(*NAV)
    pdf.rect(0, 0, pdf.w, 66, style="F")

    pdf.set_xy(pdf.l_margin, 14)
    pdf.set_text_color(*ACCENT_SOFT)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 5, "LA SPOT", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 7, "Skin check report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 28)
    pdf.multi_cell(0, 12, _t(pdf.patient_name), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*ACCENT_SOFT)
    today = datetime.date.today()
    prepared = f"Prepared {today:%B} {today.day}, {today.year}"
    if shared_with:
        prepared += f" - shared with {shared_with}"
    pdf.multi_cell(0, 5, _t(prepared), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_y(max(pdf.get_y() + 12, 74))
    pdf.set_text_color(*TEXT)
    pdf.paragraph(
        "This report collects the skin checks saved in LA Spot. Each spot that has been checked gets its own "
        "page with its latest result, its ABCD scores (Asymmetry, Border, Color, Diameter) and the images "
        "showing what each score measured, followed by its check history."
    )

    pdf.heading("Risk profile")
    if profile:
        items = [
            ("Skin type", f"Fitzpatrick {profile['fitzpatrick']}" if profile.get("fitzpatrick") else "Not given"),
            ("Sun exposure", (profile.get("sunExposure") or "Not given").capitalize()),
            ("Family history of melanoma", "Yes" if profile.get("familyHistory") else "No"),
            ("Blistering sunburns", "Yes" if profile.get("blisteringSunburns") else "No"),
            ("Many moles", "Yes" if profile.get("manyMoles") else "No"),
            ("Location", profile.get("location") or "Not given"),
        ]
        pdf.set_font("Helvetica", "", 10)
        with pdf.table(col_widths=(60, 110), first_row_as_headings=False,
                       **pdf.table_style(headings_style=FontFace())) as table:
            for label, value in items:
                row = table.row()
                row.cell(_t(label), style=FontFace(emphasis="BOLD"))
                row.cell(_t(value))
    else:
        pdf.paragraph("No risk profile has been filled in.", color=MUTED)

    pdf.heading("Spots in this report")
    if spot_rows:
        pdf.set_font("Helvetica", "", 9)
        with pdf.table(col_widths=(46, 34, 14, 42, 26, 12), text_align=("LEFT", "LEFT", "CENTER", "LEFT", "LEFT", "CENTER"),
                       **pdf.table_style()) as table:
            header = table.row()
            for title in ("Spot", "Body region", "Checks", "Latest result", "Last checked", "Page"):
                header.cell(title)
            for spot_row in spot_rows:
                row = table.row()
                for value in spot_row:
                    row.cell(_t(value))
    else:
        pdf.paragraph("No spots or checks have been saved yet.", color=MUTED)

    pdf.ln(5)
    pdf.set_fill_color(*ALERT_SOFT)
    pdf.set_text_color(*ALERT_INK)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        0, 4.6,
        _t("LA Spot is a screening prototype, not a medical device. Its results describe visual features of a photo "
           "and are not a diagnosis. Anything that changes, bleeds, itches or worries you should be seen by a "
           "licensed dermatologist, whatever this report says."),
        fill=True, padding=3, new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.set_text_color(*TEXT)


def _spot_page(pdf: _Report, user_id: str, title: str, subtitle: str, checks: list):
    """One spot: latest result, ABCD scores and overlays, then history. checks are newest first."""
    latest = checks[0]
    pdf.add_page()
    pdf.spot_start_page = pdf.page_no()

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*ACCENT)
    pdf.multi_cell(0, 9, _t(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, _t(subtitle), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    fill, ink = _result_colors(latest)
    pdf.pill(_result_label(latest), fill, ink)
    pdf.set_font("Helvetica", "", 10)
    score = latest.get("riskScore")
    pdf.cell(0, 7, _t(f"   Latest check {_date(latest.get('processedAt'))}"
                      + (f"  -  risk score {score:.0f}/100" if score is not None else "")),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    with pdf.table(col_widths=(1, 1, 1, 1), text_align="CENTER", **pdf.table_style(line_height=5.5)) as table:
        header = table.row()
        for label in ("A - Asymmetry", "B - Border", "C - Color", "D - Diameter"):
            header.cell(label)
        row = table.row()
        row.cell(_factor(latest.get("asymmetry")))
        row.cell(_factor(latest.get("border")))
        row.cell(_factor(latest.get("color")))
        row.cell(_t(_diameter(latest.get("diameterMm"))))

    pdf.ln(4)
    visuals = store.get_check_visuals(user_id, latest["processingId"])
    if visuals and any(visuals.values()):
        _visual_grid(pdf, visuals)
    else:
        outline = _outline_on_thumbnail(latest)
        if outline:
            top = pdf.get_y()
            used = pdf.picture(outline, pdf.l_margin, top, 80, 60)
            pdf.set_xy(pdf.l_margin + 86, top)
            pdf.paragraph(
                "Photo with the detected lesion outline. The detailed A/B/C/D images are kept for checks saved "
                "from now on; this check was saved before that, so only its photo and outline are available.",
                size=9, color=MUTED,
            )
            pdf.set_y(max(pdf.get_y(), top + used) + 3)

    details = []
    if latest.get("symptoms"):
        details.append("Symptoms: " + ", ".join(latest["symptoms"]))
    if latest.get("notes"):
        details.append(f'Notes: "{latest["notes"]}"')
    if (latest.get("numLesionInstances") or 0) > 1:
        details.append("More than one spot was visible in this photo; only the main one was analyzed.")
    for line in details:
        pdf.paragraph(line, size=9)

    pdf.heading("Check history", size=12, space_before=3)
    pdf.set_font("Helvetica", "", 9)
    with pdf.table(col_widths=(26, 46, 18, 18, 18, 18, 22),
                   text_align=("LEFT", "LEFT", "CENTER", "CENTER", "CENTER", "CENTER", "CENTER"),
                   **pdf.table_style(line_height=5)) as table:
        header = table.row()
        for label in ("Date", "Result", "Risk", "A", "B", "C", "D"):
            header.cell(label)
        for check in checks:
            row = table.row()
            row.cell(_date(check.get("processedAt")))
            row.cell(_t(_result_label(check)))
            row.cell("-" if check.get("riskScore") is None else f"{check['riskScore']:.0f}")
            for key in ("asymmetry", "border", "color"):
                row.cell("-" if check.get(key) is None else f"{check[key]:.1f}")
            row.cell("-" if check.get("diameterMm") is None else f"{check['diameterMm']:.1f} mm")


def _visual_grid(pdf: _Report, visuals: dict):
    """The four overlays two-by-two, each captioned with what it shows."""
    gap = 6
    cell_w = (pdf.w - pdf.l_margin - pdf.r_margin - gap) / 2
    image_h = 50
    kinds = [kind for kind in store.VISUAL_KINDS if visuals.get(kind)]
    for index in range(0, len(kinds), 2):
        top = pdf.get_y()
        if top + image_h + 14 > pdf.h - pdf.b_margin:
            pdf.add_page()
            top = pdf.get_y()
        for column, kind in enumerate(kinds[index:index + 2]):
            x = pdf.l_margin + column * (cell_w + gap)
            pdf.set_fill_color(*WELL)
            pdf.rect(x, top, cell_w, image_h, style="F")
            pdf.picture(visuals[kind], x, top, cell_w, image_h)
            title, caption = VISUAL_CAPTIONS[kind]
            pdf.set_xy(x, top + image_h + 1.5)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*ACCENT)
            pdf.cell(cell_w, 4.5, title, new_x=XPos.LEFT, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*MUTED)
            pdf.multi_cell(cell_w, 3.8, caption)
        pdf.set_text_color(*TEXT)
        pdf.set_y(top + image_h + 15)


def build_report(user_id: str, fallback_name: str = "", shared_with: str = "") -> bytes:
    """The user's full report as PDF bytes."""
    profile = store.get_profile(user_id)
    name = (profile or {}).get("fullName") or fallback_name or "Patient"
    spots = store.list_spots(user_id, include_archived=True)
    checks = store.list_checks(user_id)  # newest first

    by_spot = {}
    for check in checks:
        by_spot.setdefault(check.get("spotId"), []).append(check)

    pages = []  # (title, subtitle, checks) in page order
    for spot in spots:
        spot_checks = by_spot.get(spot["id"])
        if spot_checks:
            subtitle = spot.get("bodyRegion") or "Body region not recorded"
            if spot.get("archived"):
                subtitle += " (archived)"
            pages.append((spot["label"], subtitle, spot_checks))
    unfiled = by_spot.get(None)
    if unfiled:
        pages.append(("Checks not filed under a spot", unfiled[0].get("location") or "Body region not recorded", unfiled))
    pages.sort(key=lambda page: (not _is_elevated(page[2][0]), -(_parse_time(page[2][0].get("processedAt")) or
                                                                   datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)).timestamp()))

    empty_rows = [
        (spot["label"], spot.get("bodyRegion") or "", "0", "No checks yet", "-", "-")
        for spot in spots
        if not by_spot.get(spot["id"]) and not spot.get("archived")
    ]

    def render(start_pages):
        spot_rows = [
            (title, subtitle, str(len(page_checks)), _result_label(page_checks[0]),
             _date(page_checks[0].get("processedAt")), str(start))
            for (title, subtitle, page_checks), start in zip(pages, start_pages)
        ] + empty_rows
        pdf = _Report(name)
        _cover(pdf, profile, spot_rows, shared_with)
        starts = []
        for title, subtitle, page_checks in pages:
            _spot_page(pdf, user_id, title, subtitle, page_checks)
            starts.append(pdf.spot_start_page)
        return pdf, starts

    # A spot with a long history can run onto a second page, so the cover's page
    # numbers come from a first pass that records where each spot actually starts.
    starts = [index + 2 for index in range(len(pages))]
    for _ in range(3):
        pdf, actual = render(starts)
        if actual == starts:
            break
        starts = actual
    return bytes(pdf.output())
