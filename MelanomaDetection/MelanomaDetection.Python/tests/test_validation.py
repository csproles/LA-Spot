"""validation.py and ratelimit.py: the pure rules, without going through HTTP."""

import struct
import zlib

import pytest

import validation
from ratelimit import RateLimiter
from validation import ValidationError


def png_header(width, height):
    """Just enough of a PNG for the size check: signature, then an IHDR chunk."""
    ihdr = struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))


def jpeg_header(width, height):
    """SOI, one APP0 segment, then a SOF0 frame header."""
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = b"\xff\xc0" + struct.pack(">H", 11) + b"\x08" + struct.pack(">HH", height, width) + b"\x01\x01\x11\x00"
    return b"\xff\xd8" + app0 + sof0


def bmp_header(width, height):
    return b"BM" + b"\x00" * 16 + struct.pack("<ii", width, height) + b"\x00" * 8


class TestCleanText:
    def test_trims_and_returns(self):
        assert validation.clean_text("  hi  ", "x", 10) == "hi"

    def test_absent_is_empty(self):
        assert validation.clean_text(None, "x", 10) == ""

    def test_required_rejects_blank(self):
        with pytest.raises(ValidationError, match="required"):
            validation.clean_text("   ", "label", 10, required=True)

    @pytest.mark.parametrize("value", [5, True, ["a"], {"a": 1}])
    def test_rejects_non_text(self, value):
        with pytest.raises(ValidationError, match="must be text"):
            validation.clean_text(value, "label", 10)

    def test_rejects_too_long(self):
        with pytest.raises(ValidationError, match="60 characters"):
            validation.clean_text("a" * 61, "label", 60)

    def test_accepts_exactly_the_limit(self):
        assert validation.clean_text("a" * 60, "label", 60) == "a" * 60

    def test_rejects_control_characters(self):
        with pytest.raises(ValidationError, match="aren't allowed"):
            validation.clean_text("a\x00b", "label", 10)

    def test_single_line_by_default(self):
        with pytest.raises(ValidationError, match="single line"):
            validation.clean_text("a\nb", "label", 10)

    def test_multiline_allowed_when_asked(self):
        assert validation.clean_text("a\nb", "notes", 10, multiline=True) == "a\nb"


class TestSymptoms:
    def test_none_is_empty(self):
        assert validation.clean_symptoms(None) == []

    def test_cleans_and_drops_blanks(self):
        assert validation.clean_symptoms([" itchy ", "", "bleeding"]) == ["itchy", "bleeding"]

    def test_rejects_non_list(self):
        with pytest.raises(ValidationError):
            validation.clean_symptoms("itchy")

    def test_rejects_too_many(self):
        with pytest.raises(ValidationError, match="at most"):
            validation.clean_symptoms(["x"] * (validation.SYMPTOMS_MAX_COUNT + 1))

    def test_rejects_non_text_entries(self):
        with pytest.raises(ValidationError):
            validation.clean_symptoms([1])


class TestChatHistory:
    def test_none_is_empty(self):
        assert validation.clean_chat_history(None) == []

    def test_cleans_valid_turns(self):
        turns = [{"role": "user", "text": " hi "}, {"role": "assistant", "text": "hello"}]
        assert validation.clean_chat_history(turns) == [
            {"role": "user", "text": "hi"},
            {"role": "assistant", "text": "hello"},
        ]

    def test_rejects_non_list(self):
        with pytest.raises(ValidationError):
            validation.clean_chat_history("hi")

    def test_rejects_too_many_turns(self):
        turns = [{"role": "user", "text": "hi"}] * (validation.CHAT_HISTORY_MAX_TURNS + 1)
        with pytest.raises(ValidationError, match="at most"):
            validation.clean_chat_history(turns)

    def test_rejects_bad_role(self):
        with pytest.raises(ValidationError, match="role"):
            validation.clean_chat_history([{"role": "system", "text": "hi"}])

    def test_rejects_extra_or_missing_keys(self):
        with pytest.raises(ValidationError):
            validation.clean_chat_history([{"role": "user", "text": "hi", "extra": 1}])
        with pytest.raises(ValidationError):
            validation.clean_chat_history([{"role": "user"}])

    def test_rejects_oversized_text(self):
        with pytest.raises(ValidationError):
            validation.clean_chat_history([{"role": "user", "text": "a" * (validation.CHAT_QUESTION_MAX + 1)}])


class TestPageContext:
    def test_none_is_none(self):
        assert validation.clean_page_context(None) is None

    def test_cleans_primitive_values(self):
        raw = {"page": "results", "data": {"riskScore": 42, "riskBand": "moderate", "flagged": ["asymmetry", "border"]}}
        assert validation.clean_page_context(raw) == raw

    def test_null_data_becomes_empty_dict(self):
        assert validation.clean_page_context({"page": "results", "data": None}) == {"page": "results", "data": {}}

    def test_rejects_missing_page(self):
        with pytest.raises(ValidationError):
            validation.clean_page_context({"data": {}})

    def test_rejects_extra_keys(self):
        with pytest.raises(ValidationError):
            validation.clean_page_context({"page": "results", "data": {}, "extra": 1})

    def test_rejects_nested_objects_in_data(self):
        with pytest.raises(ValidationError):
            validation.clean_page_context({"page": "results", "data": {"nested": {"a": 1}}})

    def test_rejects_too_many_data_entries(self):
        data = {f"k{i}": i for i in range(validation.CHAT_PAGE_CONTEXT_MAX_ENTRIES + 1)}
        with pytest.raises(ValidationError, match="at most"):
            validation.clean_page_context({"page": "results", "data": data})


class TestIdsAndTypes:
    def test_spot_id_shape(self):
        assert validation.clean_spot_id("spot_0123456789ab") == "spot_0123456789ab"
        assert validation.clean_spot_id(None) is None
        assert validation.clean_spot_id("") is None

    @pytest.mark.parametrize("value", ["spot_x", "'; drop table spots;--", ["spot_0123456789ab"], 5])
    def test_bad_spot_ids(self, value):
        with pytest.raises(ValidationError):
            validation.clean_spot_id(value)

    def test_json_object(self):
        assert validation.json_object(None) == {}
        assert validation.json_object({"a": 1}) == {"a": 1}
        for bad in ([], "x", 3):
            with pytest.raises(ValidationError):
                validation.json_object(bad)

    def test_optional_bool_does_not_coerce(self):
        assert validation.optional_bool({}, "k", True) is True
        assert validation.optional_bool({"k": False}, "k", True) is False
        with pytest.raises(ValidationError):
            validation.optional_bool({"k": "false"}, "k", True)

    def test_optional_int_in_range(self):
        assert validation.optional_int_in_range(None, "f", 1, 6) is None
        assert validation.optional_int_in_range(3, "f", 1, 6) == 3
        for bad in (0, 7, "3", 2.5, True):
            with pytest.raises(ValidationError):
                validation.optional_int_in_range(bad, "f", 1, 6)


class TestImageUpload:
    def test_reads_dimensions(self):
        assert validation.image_dimensions(png_header(640, 480)) == (640, 480)
        assert validation.image_dimensions(jpeg_header(640, 480)) == (640, 480)
        assert validation.image_dimensions(bmp_header(640, 480)) == (640, 480)

    def test_accepts_ordinary_photos(self):
        for data in (png_header(4000, 3000), jpeg_header(4000, 3000), bmp_header(1000, -1000)):
            validation.check_image_upload(data)

    def test_rejects_empty(self):
        with pytest.raises(ValidationError, match="empty"):
            validation.check_image_upload(b"")

    @pytest.mark.parametrize("data", [b"MZ\x90\x00 not an image", b"<html></html>", b"GIF89a....", b"\xff\xd8\xff"])
    def test_rejects_other_file_types(self, data):
        with pytest.raises(ValidationError, match="not a valid"):
            validation.check_image_upload(data)

    def test_rejects_decompression_bomb(self):
        # A few dozen bytes on disk, but 100000x100000 pixels once decoded.
        with pytest.raises(ValidationError, match="too large"):
            validation.check_image_upload(png_header(100_000, 100_000))

    def test_rejects_zero_area(self):
        with pytest.raises(ValidationError, match="no visible area"):
            validation.check_image_upload(png_header(0, 500))


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class TestRateLimiter:
    def test_allows_up_to_the_limit_then_blocks(self):
        limiter = RateLimiter(clock=FakeClock())
        assert all(limiter.hit("r", "alice", 3, 60) is None for _ in range(3))
        assert limiter.hit("r", "alice", 3, 60) is not None

    def test_blocked_hit_reports_seconds_to_wait(self):
        clock = FakeClock()
        limiter = RateLimiter(clock=clock)
        for _ in range(2):
            limiter.hit("r", "alice", 2, 60)
        clock.now += 20
        wait = limiter.hit("r", "alice", 2, 60)
        assert 39 <= wait <= 41

    def test_window_slides(self):
        clock = FakeClock()
        limiter = RateLimiter(clock=clock)
        for _ in range(2):
            limiter.hit("r", "alice", 2, 60)
        assert limiter.hit("r", "alice", 2, 60) is not None
        clock.now += 61
        assert limiter.hit("r", "alice", 2, 60) is None

    def test_keys_and_rules_are_independent(self):
        limiter = RateLimiter(clock=FakeClock())
        limiter.hit("r", "alice", 1, 60)
        assert limiter.hit("r", "alice", 1, 60) is not None
        assert limiter.hit("r", "bob", 1, 60) is None
        assert limiter.hit("other", "alice", 1, 60) is None

    def test_blocked_hits_do_not_extend_the_block(self):
        clock = FakeClock()
        limiter = RateLimiter(clock=clock)
        limiter.hit("r", "alice", 1, 60)
        for _ in range(50):
            limiter.hit("r", "alice", 1, 60)
        clock.now += 61
        assert limiter.hit("r", "alice", 1, 60) is None

    def test_quiet_keys_are_forgotten(self):
        clock = FakeClock()
        limiter = RateLimiter(clock=clock)
        for i in range(100):
            limiter.hit("r", f"user{i}", 1, 60)
        clock.now += 200
        limiter.hit("r", "someone-new", 1, 60)
        assert len(limiter._hits) == 1


class TestCoinScale:
    PHOTO = (1200, 900)

    def test_no_coin_means_no_scale(self):
        assert validation.clean_coin_scale(None, None, self.PHOTO) is None
        assert validation.clean_coin_scale("", "  ", self.PHOTO) is None

    def test_millimetres_come_from_the_servers_own_coin_table(self):
        mm_per_px = validation.clean_coin_scale("Quarter", "242.6", self.PHOTO)
        assert mm_per_px == pytest.approx(0.1)

    @pytest.mark.parametrize("coin, diameter", [("penny", None), (None, "100"), ("", "100")])
    def test_coin_and_size_must_come_together(self, coin, diameter):
        with pytest.raises(ValidationError, match="together"):
            validation.clean_coin_scale(coin, diameter, self.PHOTO)

    def test_unknown_coin_is_refused(self):
        with pytest.raises(ValidationError, match="coin must be one of"):
            validation.clean_coin_scale("doubloon", "100", self.PHOTO)

    @pytest.mark.parametrize("diameter", ["abc", "nan", "inf", "-inf", "1e999"])
    def test_size_must_be_a_real_number(self, diameter):
        with pytest.raises(ValidationError, match="number"):
            validation.clean_coin_scale("dime", diameter, self.PHOTO)

    @pytest.mark.parametrize("diameter", ["0", "-50", "19.9", "901"])
    def test_circle_must_be_visible_and_fit_inside_the_photo(self, diameter):
        with pytest.raises(ValidationError, match="fit inside the photo"):
            validation.clean_coin_scale("dime", diameter, self.PHOTO)

    def test_circle_can_fill_the_short_side(self):
        assert validation.clean_coin_scale("nickel", "900", self.PHOTO) == pytest.approx(21.21 / 900)
