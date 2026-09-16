"""Flask API exposing the MelanomaDetector image-processing pipeline."""

import base64
import datetime
import os
import tempfile
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS

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
_results_store = {}
_explanation_cache = {}


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
    results["location"] = request.form.get("location", "")
    results["symptoms"] = request.form.getlist("symptoms")
    results["notes"] = request.form.get("notes", "")
    results["saved"] = False
    results["processed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _results_store[processing_id] = results

    return jsonify({"processingId": processing_id})


@app.route("/api/image/results/<processing_id>", methods=["GET"])
def get_results(processing_id):
    results = _results_store.get(processing_id)
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
    })


@app.route("/api/image/save/<processing_id>", methods=["POST"])
def save_to_history(processing_id):
    results = _results_store.get(processing_id)
    if results is None:
        return jsonify({"error": f"No results found for id '{processing_id}'"}), 404

    results["saved"] = True
    return jsonify({"saved": True})


@app.route("/api/image/history", methods=["GET"])
def get_history():
    entries = [
        {
            "processingId": processing_id,
            "location": results.get("location", ""),
            "symptoms": results.get("symptoms", []),
            "notes": results.get("notes", ""),
            "riskScore": results["risk_score"],
            "processedAt": results.get("processed_at"),
            "thumbnail": _encode_image_base64(_make_thumbnail(results["original"])),
        }
        for processing_id, results in _results_store.items()
        if results.get("saved")
    ]
    entries.sort(key=lambda entry: entry["processedAt"] or "", reverse=True)
    return jsonify({"entries": entries})


@app.route("/api/image/explain/<processing_id>", methods=["POST"])
def explain_results(processing_id):
    results = _results_store.get(processing_id)
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


def _encode_image_base64(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    return base64.b64encode(buffer).decode("utf-8")


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
