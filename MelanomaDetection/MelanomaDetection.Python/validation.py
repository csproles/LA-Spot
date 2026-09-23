"""Input validation for the Flask API.

Every value that arrives from outside -- JSON bodies, form fields, uploaded
files -- is checked here before it reaches main.py's logic or the database.
The web app validates the same things first so people get friendly messages,
but this API is the last line of defence and never assumes it was called
through the web app's forms.

Anything invalid raises ValidationError; main.py turns that into a 400 whose
message is safe to show directly to the user.
"""

import re
import struct

# --- limits (mirrored by the web app's InputLimits.cs and the inputs' maxlength) ---

LABEL_MAX = 60
BODY_REGION_MAX = 60
LOCATION_MAX = 120
FULL_NAME_MAX = 120
NOTES_MAX = 2000
SYMPTOM_MAX = 60
SYMPTOMS_MAX_COUNT = 20
CHAT_QUESTION_MAX = 500
CHAT_HISTORY_MAX_TURNS = 6
CHAT_PAGE_CONTEXT_MAX_ENTRIES = 20
CHAT_PAGE_NAME_MAX = 40
_CHAT_ROLES = {"user", "assistant"}

MAX_IMAGE_PIXELS = 25_000_000  # ~5000x5000; a small PNG can otherwise inflate to gigabytes when decoded

SPOT_ID_PATTERN = re.compile(r"^spot_[0-9a-f]{12}$")

# /predict's own metadata fields -- risk_model.py falls back to "missing" for
# sex/anatom_site_general (the CatBoost models were trained with that literal
# string standing in for unknown/absent metadata; see NOTES.md).
AGE_MIN, AGE_MAX = 0, 120
VALID_SEX = {"male", "female"}
VALID_BODY_SITES = {"head/neck", "upper extremity", "lower extremity", "anterior torso", "posterior torso"}

# Control characters (everything below space except tab/newline/carriage return, plus DEL).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ValidationError(ValueError):
    """The request carried something we refuse to process. str(error) is user-safe."""


def json_object(payload) -> dict:
    """A request's JSON body as a dict. Missing/absent is {}; any other JSON shape is rejected."""
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValidationError("Request body must be a JSON object.")
    return payload


def clean_text(value, field: str, max_length: int, *, required: bool = False, multiline: bool = False) -> str:
    """Trimmed text, or "" when absent. Rejects non-strings, control characters and over-long values."""
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be text.")

    text = value.strip()
    if not multiline and ("\n" in text or "\r" in text):
        raise ValidationError(f"{field} must be a single line.")
    if _CONTROL_CHARS.search(text):
        raise ValidationError(f"{field} contains characters that aren't allowed.")
    if len(text) > max_length:
        raise ValidationError(f"{field} must be {max_length} characters or fewer.")
    if required and not text:
        raise ValidationError(f"{field} is required.")
    return text


def clean_symptoms(values) -> list:
    """A list of short, single-line symptom names. None means none."""
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        raise ValidationError("symptoms must be a list.")
    if len(values) > SYMPTOMS_MAX_COUNT:
        raise ValidationError(f"symptoms can have at most {SYMPTOMS_MAX_COUNT} entries.")
    cleaned = [clean_text(item, "Each symptom", SYMPTOM_MAX) for item in values]
    return [item for item in cleaned if item]


def clean_chat_history(values) -> list:
    """A list of prior {role, text} turns for the chat feature. None means none."""
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        raise ValidationError("history must be a list.")
    if len(values) > CHAT_HISTORY_MAX_TURNS:
        raise ValidationError(f"history can have at most {CHAT_HISTORY_MAX_TURNS} turns.")
    cleaned = []
    for turn in values:
        if not isinstance(turn, dict) or set(turn) != {"role", "text"}:
            raise ValidationError('Each history turn must be an object with exactly "role" and "text".')
        role = turn["role"]
        if role not in _CHAT_ROLES:
            raise ValidationError('Each history turn\'s role must be "user" or "assistant".')
        text = clean_text(turn["text"], "history text", CHAT_QUESTION_MAX, multiline=True)
        cleaned.append({"role": role, "text": text})
    return cleaned


_PAGE_CONTEXT_SCALAR = (str, int, float, bool, type(None))


def _clean_page_context_value(value, field: str):
    if isinstance(value, _PAGE_CONTEXT_SCALAR):
        if isinstance(value, str) and len(value) > CHAT_PAGE_NAME_MAX * 4:
            raise ValidationError(f"{field} is too long.")
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > CHAT_PAGE_CONTEXT_MAX_ENTRIES:
            raise ValidationError(f"{field} has too many entries.")
        return [_clean_page_context_value(item, field) for item in value]
    raise ValidationError(f"{field} must be text, a number, a boolean, null, or a list of those.")


def clean_page_context(value):
    """The chat feature's optional {"page": str, "data": {...}} summary of what's on screen.

    Only primitive values (or lists of them) are allowed in "data" -- this is
    forwarded into an LLM prompt, so arbitrary nested client-supplied structure
    is rejected rather than passed through.
    """
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"page", "data"}:
        raise ValidationError('pageContext must be an object with exactly "page" and "data".')

    page = clean_text(value["page"], "pageContext.page", CHAT_PAGE_NAME_MAX, required=True)

    data = value["data"]
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValidationError("pageContext.data must be an object.")
    if len(data) > CHAT_PAGE_CONTEXT_MAX_ENTRIES:
        raise ValidationError(f"pageContext.data can have at most {CHAT_PAGE_CONTEXT_MAX_ENTRIES} entries.")

    cleaned_data = {}
    for key, entry in data.items():
        if not isinstance(key, str) or len(key) > CHAT_PAGE_NAME_MAX:
            raise ValidationError("pageContext.data has an invalid key.")
        cleaned_data[key] = _clean_page_context_value(entry, f"pageContext.data.{key}")

    return {"page": page, "data": cleaned_data}


def clean_spot_id(value):
    """A spot id in the shape store.create_spot issues, or None when absent."""
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not SPOT_ID_PATTERN.match(value):
        raise ValidationError("spot id is not valid.")
    return value


def optional_bool(body: dict, key: str, default: bool) -> bool:
    """A boolean field. Absent -> default; anything but true/false is rejected, not coerced."""
    if key not in body or body[key] is None:
        return default
    if not isinstance(body[key], bool):
        raise ValidationError(f"{key} must be true or false.")
    return body[key]


def optional_int_in_range(value, field: str, low: int, high: int):
    """An integer low..high inclusive, or None when absent. bool is not an int here."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValidationError(f"{field} must be a whole number from {low} to {high}.")
    return value


def clean_age(value) -> float | None:
    """A whole-number age 0-120, or None when blank -- risk_model.py sends
    None on to the model as NaN, which CatBoost handles natively."""
    if value is None or value == "":
        return None
    try:
        age = int(value)
    except (TypeError, ValueError):
        raise ValidationError("age must be a whole number.")
    if not AGE_MIN <= age <= AGE_MAX:
        raise ValidationError(f"age must be from {AGE_MIN} to {AGE_MAX}.")
    return float(age)


def clean_sex(value) -> str:
    """"male"/"female" (case-insensitive), or "missing" when blank/unrecognized."""
    if not value:
        return "missing"
    cleaned = value.strip().lower()
    return cleaned if cleaned in VALID_SEX else "missing"


def clean_body_site(value) -> str:
    """One of VALID_BODY_SITES (case-insensitive), or "missing" when blank/unrecognized."""
    if not value:
        return "missing"
    cleaned = value.strip().lower()
    return cleaned if cleaned in VALID_BODY_SITES else "missing"


# --- uploaded images ---------------------------------------------------------


def image_dimensions(data: bytes):
    """(width, height) read from a PNG/JPEG/BMP header without decoding it, else None.

    Also acts as the file-type check: it only recognises those three formats by
    their leading bytes, so a renamed .exe or .html has no dimensions.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])

    if data.startswith(b"BM") and len(data) >= 26:
        width, height = struct.unpack("<ii", data[18:26])
        return abs(width), abs(height)

    if data.startswith(b"\xff\xd8\xff"):
        return _jpeg_dimensions(data)

    return None


def _jpeg_dimensions(data: bytes):
    """Walk JPEG segments to the start-of-frame marker, which holds the size."""
    position = 2
    end = len(data)
    while position + 4 <= end:
        if data[position] != 0xFF:
            return None
        marker = data[position + 1]
        if marker == 0xFF:  # fill byte
            position += 1
            continue
        if marker in (0x01, *range(0xD0, 0xDA)):  # standalone markers carry no length
            position += 2
            continue
        (length,) = struct.unpack(">H", data[position + 2 : position + 4])
        is_start_of_frame = 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC)
        if is_start_of_frame:
            if position + 9 > end:
                return None
            height, width = struct.unpack(">HH", data[position + 5 : position + 9])
            return width, height
        position += 2 + length
    return None


def check_image_upload(data: bytes) -> None:
    """Raise unless the bytes are a plausibly-sized PNG, JPEG or BMP."""
    if not data:
        raise ValidationError("The uploaded file is empty.")

    dimensions = image_dimensions(data)
    if dimensions is None:
        raise ValidationError("The uploaded file is not a valid PNG, JPEG or BMP image.")

    width, height = dimensions
    if width <= 0 or height <= 0:
        raise ValidationError("The uploaded image has no visible area.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValidationError(
            f"The uploaded image is too large ({width}x{height}). Use a photo under 25 megapixels."
        )
