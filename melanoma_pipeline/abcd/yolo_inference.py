"""Single-image YOLO inference — one image per model.predict() call, to
avoid Ultralytics' batch square-resize distortion. Consolidated, UNCHANGED,
from revised_abcd/yolo_single_image.py.

Every detected instance is kept and returned separately (no np.max union
across instances) with its own confidence score, sorted by confidence
descending so index 0 is always the highest-confidence ("primary")
instance.
"""
import cv2
import numpy as np


def run_single_image_inference(model, image_path, conf=0.25):
    """Run inference on exactly one image (source is a single path, never a
    list or directory, so Ultralytics cannot batch it with anything else).

    Returns (instances, orig_shape) where instances is a list of dicts:
        {"instance_id": int, "confidence": float, "class_id": int,
         "class_name": str, "mask": np.uint8 array (H,W) of {0,255}}
    orig_shape is (H, W) of the source image.

    If no lesion is detected, instances is an empty list.
    """
    results = model.predict(source=str(image_path), retina_masks=True, conf=conf, verbose=False)
    result = results[0]
    orig_shape = result.orig_shape  # (H, W), from the ORIGINAL image, not the resized tensor

    instances = []
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return instances, orig_shape

    masks_np = result.masks.data.detach().cpu().numpy()  # (N, mh, mw), float in [0,1]
    confs = result.boxes.conf.detach().cpu().numpy()
    clses = result.boxes.cls.detach().cpu().numpy().astype(int)
    names = result.names

    h, w = orig_shape
    for i in range(masks_np.shape[0]):
        raw = masks_np[i]
        if raw.shape[0] != h or raw.shape[1] != w:
            # retina_masks=True should already match original resolution;
            # this is a defensive fallback, using nearest-neighbor to keep
            # the mask strictly binary.
            raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = (raw > 0.5).astype(np.uint8) * 255

        class_id = int(clses[i])
        class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)

        instances.append({
            "instance_id": i,
            "confidence": float(confs[i]),
            "class_id": class_id,
            "class_name": class_name,
            "mask": binary,
        })

    # Sort by confidence descending so "instance 0" is consistently the
    # most-confident detection for a given image.
    instances.sort(key=lambda inst: -inst["confidence"])
    for new_id, inst in enumerate(instances):
        inst["instance_id"] = new_id

    return instances, orig_shape
