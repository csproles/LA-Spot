"""Location of the frozen YOLO segmentation weights used by pipeline_v5
(the active pipeline; pipeline_v4 used the same checkpoint but is no longer
wired into the running app).

Never hard-code a developer's local machine path here. The default resolves
to a file shipped INSIDE this package (models/yolo_melanoma_seg.pt), so it
travels with the Docker build context (MelanomaDetection.Python/Dockerfile
does `COPY . .`, which includes models/) without any host-machine
dependency. Set YOLO_WEIGHTS_PATH to override it (e.g. for a larger model
file kept outside the image and mounted as a volume).
"""

import os
from pathlib import Path

_DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "models" / "yolo_melanoma_seg.pt"


def get_yolo_weights_path() -> str:
    path = os.environ.get("YOLO_WEIGHTS_PATH", str(_DEFAULT_WEIGHTS_PATH))
    if not Path(path).exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {path}. Set YOLO_WEIGHTS_PATH to the correct location, "
            f"or ensure models/yolo_melanoma_seg.pt is present in this package."
        )
    return path
