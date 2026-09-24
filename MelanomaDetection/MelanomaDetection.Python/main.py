"""Flask API exposing the frozen V5 image-processing pipeline (V5Detector)."""

import base64
import datetime
import hmac
import os
import re
import tempfile
import uuid

import cv2
import numpy as np
from flask import Flask, g, jsonify, request
from flask.json.provider import DefaultJSONProvider
from flask_compress import Compress

import evolution
import policy
import store
import validation
from llm_explainer import explain_findings
from ratelimit import RateLimiter
from resultstore import ResultStore
from risk_model import RiskModel
from textbook_chat import answer_question
from v5_detector import V5Detector, _scaled_0_10
from validation import ValidationError

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB, matches the Blazor client-side limit


class NumpyJSONProvider(DefaultJSONProvider):
    """Lets Flask's jsonify handle numpy scalar/array types from OpenCV calls."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


app = Flask(__name__)
app.json = NumpyJSONProvider(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
# Every response here is JSON, and the analysis endpoints' bodies are mostly
# base64 image data (each ~5-7MB uncompressed) -- exactly the kind of
# repetitive text gzip shrinks hardest. Compress() only engages above its
# default size floor, so small responses (health check, profile) skip the
# per-request gzip cost entirely.
Compress(app)

detector = V5Detector()
risk_model = RiskModel()
store.init_db()

# Full-size pipeline imagery for the current session only. Everything that has
# to outlive a restart (spots, saved checks, thumbnails, masks, the risk
# profile) lives in store.py instead. Each entry records the user_id it was
# produced for and is only ever handed back to that user. It is bounded --
# results expire and are capped per user and overall (see resultstore.py) --
# because each holds several MB and this API is reachable by every account.
_results_store = ResultStore()

# --- caller identity -----------------------------------------------------
#
# This API has exactly one client, the Blazor web app, which signs people in
# and forwards the account id of whoever is asking in X-User-Id. Every row in
# store.py and every entry in _results_store is scoped by that id.
#
# The header is only trustworthy if nothing else can reach this port. Docker
# publishes it on the host, so SKINCHECK_INTERNAL_KEY -- a secret shared with
# the web app -- is required alongside it whenever it is configured.
USER_ID_HEADER = "X-User-Id"
INTERNAL_KEY_HEADER = "X-Internal-Api-Key"
INTERNAL_KEY = os.environ.get("SKINCHECK_INTERNAL_KEY", "")
_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Without the key, X-User-Id is trusted from any caller, which lets whoever can
# reach this port act as any user. So refuse to start that way unless local
# development explicitly opts in.
ALLOW_NO_KEY = os.environ.get("SKINCHECK_ALLOW_NO_KEY") == "1"

if not INTERNAL_KEY:
    if not ALLOW_NO_KEY:
        raise RuntimeError(
            "SKINCHECK_INTERNAL_KEY is not set. Set it to a long random string (the web app "
            "needs the same value), or set SKINCHECK_ALLOW_NO_KEY=1 for local development only."
        )
    app.logger.warning(
        "SKINCHECK_INTERNAL_KEY is not set: the %s header is trusted from any caller. "
        "Local development only.",
        USER_ID_HEADER,
    )


# --- rate limits ---------------------------------------------------------
#
# Per account (X-User-Id), per minute. Analysis costs CPU and the explanation
# costs an OpenAI call each, so those are far tighter than ordinary reads. The
# web app enforces slightly lower numbers first so people see a friendly
# message; these are the backstop for anything that reaches this port directly.
RATE_WINDOW_SECONDS = 60
DEFAULT_RATE_LIMIT = 120
ENDPOINT_RATE_LIMITS = {
    "process_image_endpoint": ("analyze", 10),
    "predict_risk_endpoint": ("analyze", 10),
    "explain_results": ("explain", 5),
    "textbook_chat_endpoint": ("chat", 8),
    "export_account_endpoint": ("account", 5),
    "delete_account_endpoint": ("account", 5),
}
# Wrong-key attempts per client address. Keeps guessing SKINCHECK_INTERNAL_KEY impractical.
AUTH_FAILURE_LIMIT = 20

limiter = RateLimiter()


def _too_many_requests(retry_after: int):
    response = jsonify({"error": "Too many requests. Please wait a moment and try again."})
    response.status_code = 429
    response.headers["Retry-After"] = str(retry_after)
    return response


def _reject_unauthorized(message: str):
    """401, or 429 once this client address has failed too many times."""
    retry_after = limiter.hit("auth-failure", request.remote_addr or "unknown", AUTH_FAILURE_LIMIT, RATE_WINDOW_SECONDS)
    if retry_after is not None:
        return _too_many_requests(retry_after)
    return jsonify({"error": message}), 401


@app.before_request
def require_user():
    """Reject any data request that doesn't identify its user (and prove it may), then rate-limit it."""
    if request.path == "/health" or request.method == "OPTIONS":
        return None

    if INTERNAL_KEY:
        presented = request.headers.get(INTERNAL_KEY_HEADER, "")
        if not hmac.compare_digest(presented, INTERNAL_KEY):
            return _reject_unauthorized("Missing or invalid internal API key.")

    user_id = request.headers.get(USER_ID_HEADER, "")
    if not _USER_ID_PATTERN.match(user_id):
        return _reject_unauthorized(f"Missing or invalid {USER_ID_HEADER} header.")

    g.user_id = user_id

    retry_after = limiter.hit("default", user_id, DEFAULT_RATE_LIMIT, RATE_WINDOW_SECONDS)
    if retry_after is None and request.endpoint in ENDPOINT_RATE_LIMITS:
        rule, limit = ENDPOINT_RATE_LIMITS[request.endpoint]
        retry_after = limiter.hit(rule, user_id, limit, RATE_WINDOW_SECONDS)
    if retry_after is not None:
        return _too_many_requests(retry_after)

    return None


def _owned_results(processing_id):
    """In-memory results for a processing id, but only if they belong to the caller."""
    results = _results_store.get(processing_id)
    if results is None or results.get("user_id") != g.user_id:
        return None
    return results


@app.errorhandler(ValidationError)
def handle_validation_error(error):
    return jsonify({"error": str(error)}), 400


@app.errorhandler(413)
def handle_file_too_large(_error):
    return jsonify({"error": "File too large. Maximum allowed size is 5MB."}), 413


@app.errorhandler(500)
def handle_internal_error(_error):
    return jsonify({"error": "An internal error occurred while processing the request."}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/api/image/process", methods=["POST"])
def process_image_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file provided (expected multipart field 'file')"}), 400

    uploaded = request.files["file"]
    if uploaded.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(uploaded.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    # The extension is only a claim. Check the bytes really are a reasonably
    # sized PNG/JPEG/BMP before OpenCV is asked to decode them.
    data = uploaded.read()
    validation.check_image_upload(data)

    spot_id = validation.clean_spot_id(request.form.get("spot_id"))
    location = validation.clean_text(request.form.get("location"), "location", validation.LOCATION_MAX)
    symptoms = validation.clean_symptoms(request.form.getlist("symptoms"))
    notes = validation.clean_text(request.form.get("notes"), "notes", validation.NOTES_MAX, multiline=True)

    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    with os.fdopen(fd, "wb") as tmp_file:
        tmp_file.write(data)

    try:
        results = detector.process_image(tmp_path)
    except FileNotFoundError:
        return jsonify({
            "error": "Could not read the uploaded file as an image. It may be corrupted or in an unsupported format.",
        }), 400
    except Exception:
        app.logger.exception("Unexpected error while processing image")
        return jsonify({"error": "An internal error occurred while processing the image."}), 500
    finally:
        os.remove(tmp_path)

    processing_id = f"proc_{uuid.uuid4().hex[:12]}"
    spot = store.get_spot(g.user_id, spot_id) if spot_id else None

    results["user_id"] = g.user_id
    results["spot_id"] = spot["id"] if spot else None
    results["location"] = location or (spot["bodyRegion"] if spot else "")
    results["symptoms"] = symptoms
    results["notes"] = notes
    results["saved"] = False
    results["processed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Measured now, while the segmentation mask and hair-removed image are in
    # hand, so a later check of this spot can compare against them cheaply.
    results["area_px"] = evolution.lesion_area_px(results["segmentation"])
    results["lab"] = evolution.lesion_lab_mean(results["hair_removed"], results["segmentation"])

    if spot:
        _score_evolution(results)

    _results_store[processing_id] = results

    return jsonify({"processingId": processing_id})


def _abcd_features_to_abcde_scores(abcd: dict) -> dict:
    """Adapts the risk model's raw 17-value ABCD feature dict into the
    {"asymmetry": {"score", "details"}, ...} shape llm_explainer.explain_findings
    and the existing UI panels expect, reusing V5Detector's own display-scaling
    (_scaled_0_10) and thresholds (0.20/0.50/0.35) since these are the same
    ABCD features V5 computes, just fed to a different (CatBoost) decision
    model instead of V5's frozen logistic regression.

    ponytail: "concern" here is approximated as raw_value > threshold, not
    the fuller per-criterion logic in melanoma_pipeline/abcd/pipeline.py
    (asymmetry/border/color scoring functions each compute their own concern
    flag from more than a single threshold comparison). Good enough for the
    explanation prompt's flagged/unflagged split; call abcd.pipeline.score_instance
    directly instead if per-letter concern accuracy ever matters for this model.
    """
    a, b, c = abcd.get("A_value"), abcd.get("B_circularity"), abcd.get("C_value")
    return {
        "asymmetry": {
            "score": _scaled_0_10(a, 0.20),
            "details": {"raw_asymmetry_ratio": a, "concern": bool(a is not None and a > 0.20)},
        },
        "border": {
            "score": _scaled_0_10(b, 0.50),
            "details": {"raw_border_irregularity": b, "concern": bool(b is not None and b > 0.50)},
        },
        "color": {
            "score": _scaled_0_10(c, 0.35),
            "details": {"color_cv": c, "concern": bool(c is not None and c > 0.35)},
        },
        "diameter": {
            "score": None,
            "details": {
                "reason": "no validated physical (mm) calibration is available for this pipeline",
                "diameter_px": abcd.get("D_px"),
            },
        },
        "evolving": {"score": None, "details": {"reason": "no prior check to compare against"}},
    }


@app.route("/predict", methods=["POST"])
def predict_risk_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "No file provided (expected multipart field 'image')"}), 400

    uploaded = request.files["image"]
    if uploaded.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(uploaded.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    data = uploaded.read()
    validation.check_image_upload(data)

    age = validation.clean_age(request.form.get("age"))
    sex = validation.clean_sex(request.form.get("sex"))
    body_site = validation.clean_body_site(request.form.get("body_site"))

    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    with os.fdopen(fd, "wb") as tmp_file:
        tmp_file.write(data)

    try:
        result = risk_model.predict(tmp_path, age, sex, body_site)
    except FileNotFoundError:
        return jsonify({
            "error": "Could not read the uploaded file as an image. It may be corrupted or in an unsupported format.",
        }), 400
    except Exception:
        app.logger.exception("Unexpected error while running the risk model")
        return jsonify({"error": "An internal error occurred while analyzing the image."}), 500
    finally:
        os.remove(tmp_path)

    # Stash a minimal, compatible result under a new processing id so the
    # existing POST /api/image/explain/{id} endpoint (llm_explainer.py) works
    # unchanged for this pipeline's results too -- no new explanation code.
    processing_id = f"pred_{uuid.uuid4().hex[:12]}"
    _results_store[processing_id] = {
        "user_id": g.user_id,
        "abcde_scores": _abcd_features_to_abcde_scores(result["abcd_features"]),
        "risk_score": round(result["risk_score"] * 100, 1),
        "overall_visual_concern": None,
        "symptoms": [],
    }

    result["processingId"] = processing_id
    return jsonify(result)


def _score_evolution(results: dict):
    """Fill in the Evolving score by comparing against this spot's previous check.

    Leaves the placeholder untouched when the spot has no prior check (its first
    photo) or when that prior check predates mask storage, so the UI can show a
    "needs a second photo" state rather than a fabricated zero.
    """
    prior = store.get_last_check(g.user_id, results["spot_id"])
    if prior is None or not prior.get("mask"):
        return

    prior_mask = cv2.imdecode(np.frombuffer(prior["mask"], np.uint8), cv2.IMREAD_GRAYSCALE)
    if prior_mask is None:
        return

    score, details = evolution.score_change(
        current={
            "mask": results["segmentation"],
            "area_px": results["area_px"],
            "lab": results["lab"],
            "mm_per_px": results.get("mm_per_px"),
            "risk_score": results["risk_score"],
        },
        prior={
            "mask": prior_mask,
            "area_px": prior["areaPx"],
            "lab": prior["lab"],
            "mm_per_px": prior["mmPerPx"],
            "risk_score": prior["riskScore"],
        },
    )

    details["compared_with"] = prior["processingId"]
    details["compared_with_at"] = prior["processedAt"]
    results["abcde_scores"]["evolving"] = {"score": score, "details": details}


@app.route("/api/image/results/<processing_id>", methods=["GET"])
def get_results(processing_id):
    results = _owned_results(processing_id)
    if results is None:
        return jsonify({"error": f"No results found for id '{processing_id}'"}), 404

    return jsonify({
        "processingId": processing_id,
        "original": _encode_stage_base64("original", results["original"]),
        "bilateral_filtered": _encode_stage_base64("bilateral_filtered", results["bilateral_filtered"]),
        "noise_removed": _encode_stage_base64("noise_removed", results["noise_removed"]),
        "hair_removed": _encode_stage_base64("hair_removed", results["hair_removed"]),
        "segmentation": _encode_image_base64(results["segmentation"]),
        "edges": _encode_image_base64(results["edges"]),
        "asymmetry_visual": _encode_image_base64(results["asymmetry_visual"]),
        "border_visual": _encode_image_base64(results["border_visual"]),
        "color_visual": _encode_image_base64(results["color_visual"]),
        "diameter_visual": _encode_image_base64(results["diameter_visual"]),
        "multi_instance_overlay": (
            _encode_image_base64(results["multi_instance_overlay"])
            if results.get("multi_instance_overlay") is not None
            else None
        ),
        "abcde_scores": results["abcde_scores"],
        "risk_score": results["risk_score"],
        "overall_visual_concern": results.get("overall_visual_concern"),
        # Explicit boolean so the client never has to string-compare against
        # the "NO_DETECTION" sentinel itself -- see policy.CONCERN_NO_DETECTION.
        "no_detection": results.get("no_detection", False),
        "num_lesion_instances": results.get("num_lesion_instances"),
        "multi_lesion_detected": results.get("multi_lesion_detected", False),
        "location": results.get("location", ""),
        "symptoms": results.get("symptoms", []),
        "notes": results.get("notes", ""),
        "spotId": results.get("spot_id"),
        "priorThumbnail": _prior_thumbnail_base64(results),
    })


def _prior_thumbnail_base64(results: dict):
    """The previous check's thumbnail for this spot, for the before/after comparison."""
    spot_id = results.get("spot_id")
    if not spot_id:
        return None

    evolving_details = results.get("abcde_scores", {}).get("evolving", {}).get("details")
    if not isinstance(evolving_details, dict):
        return None

    compared_with = evolving_details.get("compared_with")
    if not compared_with:
        return None

    prior = store.get_check(g.user_id, compared_with)
    if prior is None or not prior.get("thumbnail"):
        return None
    return base64.b64encode(prior["thumbnail"]).decode("utf-8")


@app.route("/api/image/save/<processing_id>", methods=["POST"])
def save_to_history(processing_id):
    results = _owned_results(processing_id)
    if results is None:
        return jsonify({"error": f"No results found for id '{processing_id}'"}), 404

    # The spot is normally chosen before the photo is analyzed, but accept a
    # late assignment too so a check can be filed after the fact. Symptoms and
    # notes likewise: the client confirms the lesion outline right after the
    # photo and collects details afterwards, so they arrive here, not at process.
    body = validation.json_object(request.get_json(silent=True))
    spot_id = (
        validation.clean_spot_id(body.get("spotId"))
        or validation.clean_spot_id(request.form.get("spot_id"))
        or results.get("spot_id")
    )
    if spot_id and store.get_spot(g.user_id, spot_id) is None:
        return jsonify({"error": f"No spot found for id '{spot_id}'"}), 404

    if body.get("symptoms") is not None:
        results["symptoms"] = validation.clean_symptoms(body["symptoms"])
    if body.get("notes") is not None:
        results["notes"] = validation.clean_text(body["notes"], "notes", validation.NOTES_MAX, multiline=True)

    store.save_check(
        user_id=g.user_id,
        processing_id=processing_id,
        spot_id=spot_id,
        risk_score=results["risk_score"],
        abcde_scores=results["abcde_scores"],
        location=results.get("location", ""),
        symptoms=results.get("symptoms", []),
        notes=results.get("notes", ""),
        processed_at=results["processed_at"],
        thumbnail_png=_encode_image_png(_make_thumbnail(results["original"])),
        mask_png=_encode_image_png(results["segmentation"]),
        mm_per_px=results.get("mm_per_px"),
        area_px=results.get("area_px"),
        lab=results.get("lab"),
        overall_visual_concern=results.get("overall_visual_concern"),
        num_lesion_instances=results.get("num_lesion_instances"),
    )

    results["saved"] = True
    results["spot_id"] = spot_id
    return jsonify({"saved": True, "spotId": spot_id})


def _query_paging(default_limit=None, max_limit=200):
    """(limit, offset) from ?limit=&offset=, or (None, 0) when limit is absent.

    Omitting limit preserves the historical "give me everything" behavior for
    callers that still need the full list (the dashboard's recent-checks
    widget, chat context, data export). Passing it enables real paging.
    """
    raw_limit = request.args.get("limit")
    if raw_limit is None:
        return default_limit, 0

    try:
        limit = int(raw_limit)
        offset = int(request.args.get("offset", 0))
    except ValueError:
        raise ValidationError("limit and offset must be whole numbers.")
    if not 0 < limit <= max_limit or offset < 0:
        raise ValidationError(f"limit must be 1-{max_limit} and offset must be 0 or more.")
    return limit, offset


@app.route("/api/image/history", methods=["GET"])
def get_history():
    limit, offset = _query_paging()
    entries = [
        {
            "processingId": check["processingId"],
            "spotId": check["spotId"],
            "spotLabel": check.get("spotLabel"),
            "location": check.get("bodyRegion") or check["location"],
            "symptoms": check["symptoms"],
            "notes": check["notes"],
            "riskScore": check["riskScore"],
            "overallVisualConcern": check.get("overallVisualConcern"),
            "numLesionInstances": check.get("numLesionInstances"),
            "processedAt": check["processedAt"],
            "thumbnail": base64.b64encode(check["thumbnail"]).decode("utf-8")
            if check["thumbnail"]
            else "",
        }
        for check in store.list_checks(g.user_id, limit=limit, offset=offset)
    ]
    response = {"entries": entries}
    if limit is not None:
        response["total"] = store.count_checks(g.user_id)
    return jsonify(response)


# --- spots ---------------------------------------------------------------


@app.route("/api/spots", methods=["GET"])
def list_spots_endpoint():
    profile = store.get_profile(g.user_id)
    spots = [_with_due_date(spot, profile) for spot in store.list_spots(g.user_id)]
    return jsonify({"spots": spots})


@app.route("/api/spots", methods=["POST"])
def create_spot_endpoint():
    body = validation.json_object(request.get_json(silent=True))
    label = validation.clean_text(body.get("label"), "label", validation.LABEL_MAX, required=True)
    body_region = validation.clean_text(
        body.get("bodyRegion"), "bodyRegion", validation.BODY_REGION_MAX, required=True
    )

    spot = store.create_spot(g.user_id, label, body_region)
    return jsonify(_with_due_date({**spot, **store.aggregate_checks([])}, store.get_profile(g.user_id))), 201


@app.route("/api/spots/<spot_id>", methods=["GET"])
def get_spot_endpoint(spot_id):
    spot = store.get_spot(g.user_id, spot_id)
    if spot is None:
        return jsonify({"error": f"No spot found for id '{spot_id}'"}), 404

    checks = store.get_checks_for_spot(g.user_id, spot_id)
    profile = store.get_profile(g.user_id)
    summary = _with_due_date({**spot, **store.aggregate_checks(_as_rows(checks))}, profile)

    return jsonify({
        **summary,
        "checks": [
            {
                "processingId": check["processingId"],
                "riskScore": check["riskScore"],
                "overallVisualConcern": check.get("overallVisualConcern"),
                "numLesionInstances": check.get("numLesionInstances"),
                "diameterMm": check["diameterMm"],
                "asymmetry": check["asymmetry"],
                "border": check["border"],
                "color": check["color"],
                "symptoms": check["symptoms"],
                "notes": check["notes"],
                "processedAt": check["processedAt"],
                "thumbnail": base64.b64encode(check["thumbnail"]).decode("utf-8")
                if check["thumbnail"]
                else "",
            }
            for check in checks
        ],
    })


@app.route("/api/spots/<spot_id>", methods=["PATCH"])
def update_spot_endpoint(spot_id):
    if store.get_spot(g.user_id, spot_id) is None:
        return jsonify({"error": f"No spot found for id '{spot_id}'"}), 404

    body = validation.json_object(request.get_json(silent=True))
    label = validation.clean_text(body.get("label"), "label", validation.LABEL_MAX)
    archived = body.get("archived")
    if archived is not None and not isinstance(archived, bool):
        raise ValidationError("archived must be true or false.")
    spot = store.update_spot(g.user_id, spot_id, label=label or None, archived=archived)
    return jsonify(spot)


def _as_rows(checks: list) -> list:
    """Adapt store check dicts to the key names store._aggregate expects."""
    return [
        {
            "risk_score": check["riskScore"],
            "overall_visual_concern": check.get("overallVisualConcern"),
            "processed_at": check["processedAt"],
        }
        for check in checks
    ]


def _with_due_date(spot: dict, profile) -> dict:
    """Layer the policy-derived recheck fields onto a spot summary.

    Cadence and band are keyed on the spot's last overall_visual_concern
    (V5's own LOWER/ELEVATED result) when it's known, not a risk_score cut --
    see policy.py. A concern of None (no detection, or a check saved before
    this field existed) falls back to the risk_score band there.
    """
    last_concern = spot.get("lastOverallVisualConcern")
    next_due = policy.next_due_at(spot.get("lastCheckedAt"), spot.get("lastRiskScore"), profile, last_concern)
    return {
        **spot,
        "riskBand": policy.risk_band(spot["lastRiskScore"]) if spot.get("lastRiskScore") is not None else None,
        "cadenceDays": policy.cadence_days(spot.get("lastRiskScore"), profile, last_concern)
        if spot.get("lastCheckedAt")
        else None,
        "nextDueAt": next_due,
    }


# --- risk profile --------------------------------------------------------


@app.route("/api/profile", methods=["GET"])
def get_profile_endpoint():
    profile = store.get_profile(g.user_id)
    if profile is None:
        return jsonify({"configured": False})
    return jsonify({**profile, "configured": True})


@app.route("/api/profile", methods=["PUT"])
def save_profile_endpoint():
    body = validation.json_object(request.get_json(silent=True))

    try:
        fitzpatrick = validation.optional_int_in_range(body.get("fitzpatrick"), "fitzpatrick", 1, 6)
    except ValidationError:
        return jsonify({"error": "fitzpatrick must be 1-6 (Fitzpatrick I-VI) or null."}), 400

    sun_exposure = validation.clean_text(body.get("sunExposure"), "sunExposure", 20)
    if sun_exposure and sun_exposure not in policy.SUN_EXPOSURE_LEVELS:
        return jsonify({"error": "sunExposure must be one of: " + ", ".join(policy.SUN_EXPOSURE_LEVELS)}), 400

    profile = store.save_profile(
        user_id=g.user_id,
        full_name=validation.clean_text(body.get("fullName"), "fullName", validation.FULL_NAME_MAX),
        location=validation.clean_text(body.get("location"), "location", validation.LOCATION_MAX),
        sun_exposure=sun_exposure,
        fitzpatrick=fitzpatrick,
        family_history=validation.optional_bool(body, "familyHistory", False),
        blistering_sunburns=validation.optional_bool(body, "blisteringSunburns", False),
        many_moles=validation.optional_bool(body, "manyMoles", False),
        recheck_reminders=validation.optional_bool(body, "recheckReminders", True),
        high_risk_alerts=validation.optional_bool(body, "highRiskAlerts", True),
        share_with_dermatologist=validation.optional_bool(body, "shareWithDermatologist", True),
        anonymous_analytics=validation.optional_bool(body, "anonymousAnalytics", False),
    )
    return jsonify({**profile, "configured": True})


@app.route("/api/image/explain/<processing_id>", methods=["POST"])
def explain_results(processing_id):
    results = _owned_results(processing_id)
    if results is None:
        return jsonify({"error": f"No results found for id '{processing_id}'"}), 404

    if "explanation" in results:
        return jsonify({"explanation": results["explanation"]})

    try:
        explanation = explain_findings(
            results["abcde_scores"],
            overall_visual_concern=results.get("overall_visual_concern"),
            risk_score=results["risk_score"],
            profile=store.get_profile(g.user_id),
            evolving=results["abcde_scores"].get("evolving"),
            symptoms=results.get("symptoms", []),
        )
    except Exception:
        app.logger.exception("LLM explanation request failed")
        return jsonify({
            "error": "Could not generate an explanation right now. Check that the "
                     "OpenAI API key is configured correctly and try again.",
        }), 502

    results["explanation"] = explanation
    return jsonify({"explanation": explanation})


@app.route("/api/chat", methods=["POST"])
def textbook_chat_endpoint():
    body = validation.json_object(request.get_json(silent=True))
    question = validation.clean_text(
        body.get("question"), "question", validation.CHAT_QUESTION_MAX, required=True, multiline=True
    )
    history = validation.clean_chat_history(body.get("history"))
    page_context = validation.clean_page_context(body.get("pageContext"))

    try:
        result = answer_question(question, page_context=page_context, history=history)
    except Exception:
        app.logger.exception("Textbook chat request failed")
        return jsonify({
            "error": "Could not answer that right now. Check that the OpenAI API key "
                     "is configured correctly and try again.",
        }), 502

    return jsonify(result)


# --- account -------------------------------------------------------------


@app.route("/api/account/export", methods=["GET"])
def export_account_endpoint():
    """Everything stored for the caller, for the web app's data-export download."""
    return jsonify(store.export_user_data(g.user_id))


@app.route("/api/account", methods=["DELETE"])
def delete_account_endpoint():
    """Erase the caller's stored data and any in-memory results of theirs."""
    removed = store.delete_user_data(g.user_id)

    pending = _results_store.delete_user(g.user_id)

    return jsonify({"deleted": True, **removed, "pendingResults": pending})


def _encode_image_png(image: np.ndarray) -> bytes:
    """PNG-encode an image to raw bytes, for storage as a SQLite BLOB."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    return buffer.tobytes()


def _encode_image_base64(image: np.ndarray) -> str:
    return base64.b64encode(_encode_image_png(image)).decode("utf-8")


# Photographic-only pipeline stages: real camera pixels with no drawn
# measurement lines on top, where JPEG's lossy artifacts don't risk hiding or
# distorting anything diagnostic. Encoding these as JPEG rather than lossless
# PNG is most of this endpoint's payload weight (base64 amplifies it another
# 33%). Everything else here -- segmentation/edges/the *_visual overlays --
# stays PNG: those are thin drawn lines and masks where compression artifacts
# could blur exactly the boundary the UI is trying to show.
_JPEG_STAGES = {"original", "bilateral_filtered", "noise_removed", "hair_removed"}
_JPEG_QUALITY = 85


def _encode_image_base64_lossy(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _encode_stage_base64(stage: str, image: np.ndarray) -> str:
    return _encode_image_base64_lossy(image) if stage in _JPEG_STAGES else _encode_image_base64(image)


def _make_thumbnail(image: np.ndarray, max_width: int = 160) -> np.ndarray:
    """Downscale for the history list, so a page of saved checks stays lightweight."""
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / width
    new_size = (max_width, max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
