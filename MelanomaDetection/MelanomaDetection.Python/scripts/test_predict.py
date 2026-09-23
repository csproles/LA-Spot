"""Smoke-test POST /predict against a running Flask server: sends one sample
image with age/sex/body_site, prints the response and timing.

Uses only the standard library (no `requests`) so it runs the same inside
the Docker container as on a dev machine.

Usage (from MelanomaDetection.Python/):
    python scripts/test_predict.py [image_path]

Environment:
    PREDICT_URL              default http://localhost:5002/predict
    SKINCHECK_INTERNAL_KEY   required unless the server was started with
                             SKINCHECK_ALLOW_NO_KEY=1
"""
from __future__ import annotations

import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

DEFAULT_IMAGE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "Images", "Benign", "ISIC_0000005.jpg"
)


def _encode_multipart(fields: dict, file_field: str, file_path: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )

    filename = os.path.basename(file_path)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n".encode()
        + file_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main() -> None:
    image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(DEFAULT_IMAGE)
    url = os.environ.get("PREDICT_URL", "http://localhost:5002/predict")

    body, content_type = _encode_multipart(
        {"age": "45", "sex": "female", "body_site": "lower extremity"}, "image", image_path
    )

    headers = {
        "Content-Type": content_type,
        "X-User-Id": "test-user",
    }
    internal_key = os.environ.get("SKINCHECK_INTERNAL_KEY")
    if internal_key:
        headers["X-Internal-Api-Key"] = internal_key

    request = urllib.request.Request(url, data=body, headers=headers, method="POST")

    print(f"POST {url}  image={image_path}")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            elapsed = time.monotonic() - t0
            payload = json.loads(response.read())
            print(f"HTTP {response.status} in {elapsed:.2f}s\n")
            print(json.dumps(payload, indent=2))
    except urllib.error.HTTPError as e:
        elapsed = time.monotonic() - t0
        print(f"HTTP {e.code} in {elapsed:.2f}s\n")
        print(e.read().decode(errors="replace"))
        sys.exit(1)


if __name__ == "__main__":
    main()
