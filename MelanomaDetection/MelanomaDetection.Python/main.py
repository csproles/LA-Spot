"""Flask API exposing the MelanomaDetector image-processing pipeline."""

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
from flask_cors import CORS

import evolution
import policy
import store
from image_processor import MelanomaDetector
from llm_explainer import explain_findings

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
CORS(app)

detector = MelanomaDetector()
store.init_db()

# Full-size pipeline imagery for the current session only. Everything that has
# to outlive a restart (spots, saved checks, thumbnails, masks, the risk
# profile) lives in store.py instead. Each entry records the user_id it was
# produced for and is only ever handed back to that user.
_results_store = {}
_explanation_cache = {}

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

if not INTERNAL_KEY:
    app.logger.warning(
        "SKINCHECK_INTERNAL_KEY is not set: the %s header is trusted from any caller. "
        "Fine for local development only.",
        USER_ID_HEADER,
    )


@app.before_request
def require_user():
    """Reject any data request that doesn't identify its user (and prove it may)."""
    if request.path == "/health" or request.method == "OPTIONS":
        return None

    if INTERNAL_KEY:
        presented = request.headers.get(INTERNAL_KEY_HEADER, "")
        if not hmac.compare_digest(presented, INTERNAL_KEY):
            return jsonify({"error": "Missing or invalid internal API key."}), 401

    user_id = request.headers.get(USER_ID_HEADER, "")
    if not _USER_ID_PATTERN.match(user_id):
        return jsonify({"error": f"Missing or invalid {USER_ID_HEADER} header."}), 401

    g.user_id = user_id
    return None


def _owned_results(processing_id):
    """In-memory results for a processing id, but only if they belong to the caller."""
    results = _results_store.get(processing_id)
    if results is None or results.get("user_id") != g.user_id:
        return None
    return results


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

    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    uploaded.save(tmp_path)

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
    spot_id = request.form.get("spot_id") or None
    spot = store.get_spot(g.user_id, spot_id) if spot_id else None

    results["user_id"] = g.user_id
    results["spot_id"] = spot["id"] if spot else None
    results["location"] = request.form.get("location") or (spot["bodyRegion"] if spot else "")
    results["symptoms"] = request.form.getlist("symptoms")
    results["notes"] = request.form.get("notes", "")
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
        "original": _encode_image_base64(results["original"]),
        "bilateral_filtered": _encode_image_base64(results["bilateral_filtered"]),
        "noise_removed": _encode_image_base64(results["noise_removed"]),
        "hair_removed": _encode_image_base64(results["hair_removed"]),
        "segmentation": _encode_image_base64(results["segmentation"]),
        "edges": _encode_image_base64(results["edges"]),
        "asymmetry_visual": _encode_image_base64(results["asymmetry_visual"]),
        "border_visual": _encode_image_base64(results["border_visual"]),
        "color_visual": _encode_image_base64(results["color_visual"]),
        "diameter_visual": _encode_image_base64(results["diameter_visual"]),
        "abcde_scores": results["abcde_scores"],
        "risk_score": results["risk_score"],
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
    body = request.get_json(silent=True) or {}
    spot_id = body.get("spotId") or request.form.get("spot_id") or results.get("spot_id")
    if spot_id and store.get_spot(g.user_id, spot_id) is None:
        return jsonify({"error": f"No spot found for id '{spot_id}'"}), 404

    if isinstance(body.get("symptoms"), list):
        results["symptoms"] = [str(s) for s in body["symptoms"]]
    if isinstance(body.get("notes"), str):
        results["notes"] = body["notes"]

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
    )

    results["saved"] = True
    results["spot_id"] = spot_id
    return jsonify({"saved": True, "spotId": spot_id})


@app.route("/api/image/history", methods=["GET"])
def get_history():
    entries = [
        {
            "processingId": check["processingId"],
            "spotId": check["spotId"],
            "spotLabel": check.get("spotLabel"),
            "location": check.get("bodyRegion") or check["location"],
            "symptoms": check["symptoms"],
            "notes": check["notes"],
            "riskScore": check["riskScore"],
            "processedAt": check["processedAt"],
            "thumbnail": base64.b64encode(check["thumbnail"]).decode("utf-8")
            if check["thumbnail"]
            else "",
        }
        for check in store.list_checks(g.user_id)
    ]
    return jsonify({"entries": entries})


# --- spots ---------------------------------------------------------------


@app.route("/api/spots", methods=["GET"])
def list_spots_endpoint():
    profile = store.get_profile(g.user_id)
    spots = [_with_due_date(spot, profile) for spot in store.list_spots(g.user_id)]
    return jsonify({"spots": spots})


@app.route("/api/spots", methods=["POST"])
def create_spot_endpoint():
    body = request.get_json(silent=True) or {}
    label = (body.get("label") or "").strip()
    body_region = (body.get("bodyRegion") or "").strip()

    if not label:
        return jsonify({"error": "A label is required."}), 400
    if not body_region:
        return jsonify({"error": "A body region is required."}), 400

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

    body = request.get_json(silent=True) or {}
    label = body.get("label")
    spot = store.update_spot(
        g.user_id,
        spot_id,
        label=label.strip() if isinstance(label, str) and label.strip() else None,
        archived=body.get("archived"),
    )
    return jsonify(spot)


def _as_rows(checks: list) -> list:
    """Adapt store check dicts to the key names store._aggregate expects."""
    return [
        {"risk_score": check["riskScore"], "processed_at": check["processedAt"]}
        for check in checks
    ]


def _with_due_date(spot: dict, profile) -> dict:
    """Layer the policy-derived recheck fields onto a spot summary."""
    next_due = policy.next_due_at(spot.get("lastCheckedAt"), spot.get("lastRiskScore"), profile)
    return {
        **spot,
        "riskBand": policy.risk_band(spot["lastRiskScore"]) if spot.get("lastRiskScore") is not None else None,
        "cadenceDays": policy.cadence_days(spot.get("lastRiskScore"), profile)
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
    body = request.get_json(silent=True) or {}

    fitzpatrick = body.get("fitzpatrick")
    if fitzpatrick is not None and fitzpatrick not in range(1, 7):
        return jsonify({"error": "fitzpatrick must be 1-6 (Fitzpatrick I-VI) or null."}), 400

    sun_exposure = body.get("sunExposure") or ""
    if sun_exposure and sun_exposure not in policy.SUN_EXPOSURE_LEVELS:
        return jsonify({"error": "sunExposure must be one of: " + ", ".join(policy.SUN_EXPOSURE_LEVELS)}), 400

    profile = store.save_profile(
        user_id=g.user_id,
        full_name=body.get("fullName") or "",
        location=body.get("location") or "",
        sun_exposure=sun_exposure,
        fitzpatrick=fitzpatrick,
        family_history=bool(body.get("familyHistory")),
        blistering_sunburns=bool(body.get("blisteringSunburns")),
        many_moles=bool(body.get("manyMoles")),
    )
    return jsonify({**profile, "configured": True})


@app.route("/api/image/explain/<processing_id>", methods=["POST"])
def explain_results(processing_id):
    results = _owned_results(processing_id)
    if results is None:
        return jsonify({"error": f"No results found for id '{processing_id}'"}), 404

    if processing_id in _explanation_cache:
        return jsonify({"explanation": _explanation_cache[processing_id]})

    try:
        explanation = explain_findings(results["abcde_scores"])
    except Exception:
        app.logger.exception("LLM explanation request failed")
        return jsonify({
            "error": "Could not generate an explanation right now. Check that the "
                     "OpenAI API key is configured correctly and try again.",
        }), 502

    _explanation_cache[processing_id] = explanation
    return jsonify({"explanation": explanation})


# --- account -------------------------------------------------------------


@app.route("/api/account/export", methods=["GET"])
def export_account_endpoint():
    """Everything stored for the caller, for the web app's data-export download."""
    return jsonify(store.export_user_data(g.user_id))


@app.route("/api/account", methods=["DELETE"])
def delete_account_endpoint():
    """Erase the caller's stored data and any in-memory results of theirs."""
    removed = store.delete_user_data(g.user_id)

    owned = [pid for pid, results in list(_results_store.items()) if results.get("user_id") == g.user_id]
    for processing_id in owned:
        _results_store.pop(processing_id, None)
        _explanation_cache.pop(processing_id, None)

    return jsonify({"deleted": True, **removed, "pendingResults": len(owned)})


def _encode_image_png(image: np.ndarray) -> bytes:
    """PNG-encode an image to raw bytes, for storage as a SQLite BLOB."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    return buffer.tobytes()


def _encode_image_base64(image: np.ndarray) -> str:
    return base64.b64encode(_encode_image_png(image)).decode("utf-8")


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
