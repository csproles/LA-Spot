"""Image preprocessing and heuristic ABCDE lesion scoring using OpenCV/NumPy only.

This module implements a classical (non-ML) computer vision pipeline that mirrors
the manual workflow a dermatologist follows when applying the ABCDE rule to a
dermoscopic photo: clean up the image, isolate the lesion from surrounding skin,
then score it against the five clinical warning signs (Asymmetry, Border
irregularity, Color variation, Diameter, Evolving). It exists as a pre-ML
baseline -- see MelanomaDetector's class docstring for why OpenCV heuristics were
chosen over a trained model at this stage of the project.

This version ports the CV algorithms originally prototyped in the repo's `Code/`
directory (vignette removal, LAB-space lesion segmentation, hair-width mm
calibration, skin-tone-calibrated color analysis) into this package, in place of
the simpler placeholders (grayscale Otsu, k-means color clustering, arbitrary
pixel-reference diameter) this module started with. See each method's "Why this
approach" section for what specifically changed and why.
"""

import cv2
import numpy as np

VELLUS_HAIR_UM = 70.0
"""Average width of vellus (fine body) hair in micrometers, used as a physical
reference scale: measuring a hair's width in pixels and comparing it to this
known real-world size gives a pixels-to-millimeters conversion factor, without
needing a calibration object in the photo or camera metadata."""


class MelanomaDetector:
    """Runs a classical image-processing pipeline and a heuristic ABCDE risk score.

    What it does:
        Given a path to a skin lesion photo, runs it through vignette removal,
        denoising, hair removal, LAB-color-space lesion segmentation, and the
        five ABCDE sub-scores, then combines those into a single 0-100 risk
        score. Also produces four annotated visualization images (one per
        scored ABCDE criterion) showing exactly what evidence each score is
        based on.

    Why this approach:
        This is a pre-ML baseline: segmentation and scoring use only OpenCV/NumPy
        rather than a trained classifier. That keeps the pipeline dependency-light
        and fully interpretable -- every score traces back to a specific,
        inspectable image measurement (contour compactness, LAB color distance
        from the person's own skin tone, a hair-calibrated mm measurement, etc.)
        instead of an opaque model output, which matters for an early-stage tool
        making health-adjacent claims. The tradeoff is a real ceiling on
        classification accuracy: hand-tuned thresholds on classical CV features
        cannot capture the same nuance a trained model would. Replacing or
        augmenting this with an ML model is the natural next step, not a
        hypothetical -- it's just out of scope for this stage.

    Example:
        >>> detector = MelanomaDetector()
        >>> result = detector.process_image("lesion_photo.jpg")
        >>> result["risk_score"]
        62.3
        >>> result["abcde_scores"]["asymmetry"]["score"]
        4.6
    """

    def __init__(self):
        """No configuration needed.

        Every processing parameter (filter kernel sizes, score scaling constants,
        etc.) lives on the method that uses it rather than on shared instance
        state, since a single detector instance is expected to process many
        unrelated images over its lifetime -- main.py's Flask app creates one
        MelanomaDetector at startup and reuses it for every request.
        """
        pass

    def process_image(self, image_path: str) -> dict:
        """Run the full pipeline: load, clean, segment, score, and visualize a lesion photo.

        What it does:
            Reads the image from disk and resizes it if needed, removes the
            dermoscope vignette, denoises (median blur then bilateral filter),
            measures hair width for mm calibration, removes hair, segments the
            lesion in LAB color space, detects edges, computes the five ABCDE
            sub-scores, combines them into one risk score, and renders four
            annotated visualization images explaining those scores.

        Why this order:
            Vignette removal runs first because every later step (denoising,
            hair measurement, segmentation) would otherwise be corrupted by
            treating the dark frame border as real image content -- most
            visibly, segmentation used to mistake the vignette itself for the
            lesion. Median blur then bilateral filter matches the order used by
            the original prototype this pipeline is ported from. Hair width is
            measured right after bilateral filtering (before hair removal
            deletes the hair being measured!) so the mm-per-pixel calibration
            reflects real hair in the photo. Hair removal then runs before
            segmentation for the same reason as before: an unremoved hair
            strand can fracture segmentation into multiple disconnected
            regions. Edge detection runs on the segmented (masked) image so
            edges reflect the lesion boundary, not background texture.

        Args:
            image_path: Filesystem path to a lesion photo (JPEG/PNG/BMP -- anything
                cv2.imread can decode).

        Returns:
            A dict with:
                "original": the (possibly resized) input image, BGR ndarray.
                "bilateral_filtered": image after edge-preserving smoothing.
                "noise_removed": image after median-blur denoising.
                "hair_removed": image after hair/artifact inpainting.
                "segmentation": single-channel lesion mask (255 = lesion).
                "edges": single-channel Canny edge map of the segmented lesion.
                "asymmetry_visual", "border_visual", "color_visual",
                    "diameter_visual": BGR images annotated to show the
                    evidence behind each ABCDE sub-score.
                "abcde_scores": dict of {"asymmetry"|"border"|"color"|"diameter"|
                    "evolving": {"score": float 0-10 or None, "details": dict}}.
                    "details" holds the raw measurement(s) behind each score
                    (e.g. diameter_mm, dangerous color percentages, whether that
                    criterion crossed its own clinical concern threshold).
                "risk_score": float 0-100 combining the ABCDE sub-scores.

        Raises:
            FileNotFoundError: if image_path doesn't exist or cv2 can't decode it
                (corrupt file, unsupported format, etc.) -- cv2.imread returns
                None in both cases rather than raising, so this method raises
                explicitly instead of letting a None silently propagate into the
                rest of the pipeline.

        Example:
            >>> detector = MelanomaDetector()
            >>> result = detector.process_image("Images/Malignant/ISIC_0000002.jpg")
            >>> result["risk_score"]
            58.4
        """
        original = cv2.imread(image_path)
        if original is None:
            raise FileNotFoundError(f"Could not read image at path: {image_path}")

        original = self._resize_if_needed(original)

        no_vignette, circle_info = self._remove_vignette(original)
        median_filtered = self._remove_salt_pepper_noise(no_vignette, kernel_size=3)
        bilateral_filtered = self._apply_bilateral_filter(median_filtered)

        hair_width_px = self._measure_hair_width_px(bilateral_filtered)
        mm_per_px = (VELLUS_HAIR_UM / hair_width_px) / 1000.0 if hair_width_px else None

        hair_removed = self._remove_hair_and_artifacts(bilateral_filtered)

        mask, masked = self._segment_lesion(hair_removed)
        edges = self._detect_edges(masked)

        abcde_scores = self._compute_abcde_scores(hair_removed, mask, circle_info, mm_per_px)
        risk_score = self._calculate_risk_score(abcde_scores)
        visuals = self._build_abcd_visuals(original, mask, abcde_scores)

        return {
            "original": original,
            "bilateral_filtered": bilateral_filtered,
            "noise_removed": median_filtered,
            "hair_removed": hair_removed,
            "segmentation": mask,
            "edges": edges,
            "asymmetry_visual": visuals["asymmetry"],
            "border_visual": visuals["border"],
            "color_visual": visuals["color"],
            "diameter_visual": visuals["diameter"],
            "abcde_scores": abcde_scores,
            "risk_score": risk_score,
        }

    def _resize_if_needed(self, image: np.ndarray, max_width: int = 512) -> np.ndarray:
        """Downscale to at most max_width (preserving aspect ratio) to bound pipeline cost.

        What it does:
            If the image is wider than max_width, shrinks it (keeping aspect
            ratio) using area-based interpolation, the recommended cv2
            interpolation mode for shrinking. Images already at or under
            max_width pass through untouched.

        Why this approach:
            Profiling showed color-cluster analysis and hair-width measurement
            both scale with pixel count, so cutting pixel count here up front
            keeps large uploads (12MP+ phone photos) under the app's processing
            time budget. This runs first in the pipeline so every downstream
            step benefits from the smaller image.

        Args:
            image: BGR image as loaded by cv2.imread.
            max_width: Maximum output width in pixels. Defaults to 512.

        Returns:
            The original image, or a resized copy if it exceeded max_width.

        Example:
            >>> detector = MelanomaDetector()
            >>> img = cv2.imread("large_photo.jpg")  # e.g. 4000x3000
            >>> resized = detector._resize_if_needed(img)
            >>> resized.shape[1] <= 512
            True
        """
        height, width = image.shape[:2]
        if width <= max_width:
            return image

        scale = max_width / width
        new_size = (max_width, max(1, int(round(height * scale))))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def _remove_vignette(self, image: np.ndarray, shrink: float = 0.95):
        """Detect and flatten the dark circular vignette common in dermoscope photos.

        What it does:
            Finds the largest bright region in the frame (thresholding out
            near-black pixels) and fits a minimum enclosing circle to it. If
            that circle covers at least 30% of the frame -- a real dermoscope
            aperture, not noise -- everything outside a slightly shrunk version
            of that circle is replaced with the average color sampled from
            inside it (a flat "skin-colored" fill), and the circle's
            center/radius are returned for later use.

        Why this approach:
            Dermoscope captures are taken through a circular optical aperture,
            leaving a dark ring around the actual photo content. That ring is
            reliably darker than the lesion itself, which previously caused
            segmentation to mistake the vignette for "the lesion" (it's the
            single darkest region in the frame). Rather than working around
            that downstream (e.g. excluding contours that touch the image
            border), this removes the vignette once, up front, so every later
            step just sees a normal-looking photo. The detected circle is also
            reused later for skin-tone sampling in color scoring (see
            _sample_skin_color), since it marks where genuine skin pixels are.

        Args:
            image: BGR image, ideally the resized original.
            shrink: Fraction of the detected radius to keep as "inside" the
                vignette when flattening the outside (0.95 leaves a small
                margin so the flatten doesn't leave a visible seam at the
                vignette's own edge).

        Returns:
            A tuple of (result, circle_info):
                result: BGR image with any detected vignette flattened to a
                    flat skin-color fill, or the original image unchanged if no
                    vignette was detected.
                circle_info: (center_x, center_y, radius) in pixels, or None if
                    no sufficiently large circular vignette was found (e.g. a
                    photo that wasn't taken through a dermoscope at all).

        Example:
            >>> detector = MelanomaDetector()
            >>> img = cv2.imread("dermoscope_photo.jpg")
            >>> no_vignette, circle_info = detector._remove_vignette(img)
            >>> circle_info is None or len(circle_info) == 3
            True
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image, None

        largest = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)

        circle_area = cv2.contourArea(largest)
        img_area = image.shape[0] * image.shape[1]
        if circle_area / img_area < 0.3:
            return image, None

        circle_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(circle_mask, (int(cx), int(cy)), int(radius * shrink), 255, -1)

        skin_color = cv2.mean(image, mask=circle_mask)[:3]
        result = image.copy()
        result[circle_mask == 0] = [int(c) for c in skin_color]

        return result, (int(cx), int(cy), int(radius))

    def _apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """Smooth the image while preserving edges (reduces noise ahead of segmentation).

        What it does:
            Runs cv2.bilateralFilter, which blurs pixels together based on both
            spatial proximity and color similarity -- unlike a plain Gaussian
            blur, it won't blur across a strong edge, since pixels on opposite
            sides of a true edge are dissimilar in color and so barely influence
            each other.

        Why this approach:
            Dermoscopic photos pick up sensor noise and JPEG compression
            artifacts that can fool lesion segmentation. A standard Gaussian
            blur would remove that noise but also soften the lesion's actual
            border, which is exactly the feature the border-irregularity score
            depends on. Bilateral filtering is the standard edge-preserving
            denoising choice for this reason.

        Args:
            image: BGR image to smooth.

        Returns:
            A same-size, same-dtype BGR image with noise reduced and edges intact.

        Example:
            >>> detector = MelanomaDetector()
            >>> smoothed = detector._apply_bilateral_filter(median_filtered_image)
        """
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    def _remove_salt_pepper_noise(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Strip impulse (salt-and-pepper) noise via median blur.

        What it does:
            Replaces each pixel with the median of its kernel_size x kernel_size
            neighborhood. Unlike averaging filters, the median is robust to
            outliers, so a single stray black-or-white pixel gets replaced
            outright rather than blended into its neighbors.

        Why this approach:
            Median blur is a standard, cheap first denoising pass that clears
            single-pixel speckle noise before the more expensive edge-preserving
            bilateral filter runs. kernel_size=3 keeps this gentle -- just
            enough to clear impulse noise without softening small real lesion
            texture.

        Args:
            image: BGR image to denoise (typically the vignette-removed image).
            kernel_size: Neighborhood size for the median filter. Must be odd;
                defaults to 3.

        Returns:
            A same-size, same-dtype BGR image with impulse noise removed.

        Example:
            >>> detector = MelanomaDetector()
            >>> cleaned = detector._remove_salt_pepper_noise(no_vignette_image)
        """
        return cv2.medianBlur(image, kernel_size)

    def _measure_hair_width_px(self, image: np.ndarray, kernel_size: int = 17, threshold: int = 10):
        """Estimate the width, in pixels, of hair strands visible in the photo.

        What it does:
            Uses blackhat morphology to isolate thin dark structures (hair),
            then runs a distance transform on the resulting hair mask and finds
            its "skeleton" (ridge points where the distance transform is
            locally maximal -- i.e. the centerline of each strand). The
            distance-transform value at each skeleton point is half that
            strand's local width; doubling the median of those values across
            all detected hair gives a single representative hair width in
            pixels. Returns None if too little hair-like structure was found to
            trust the estimate.

        Why this approach:
            This is the calibration step that makes a real millimeter diameter
            measurement possible (see _calculate_diameter_score): vellus (fine
            body) hair has a well-documented average width of about 70
            micrometers, essentially constant across people. Measuring that
            same hair's width in pixels in this specific photo gives a
            pixels-to-millimeters conversion factor for this photo's exact zoom
            level, without needing a physical ruler in frame or camera
            metadata. The distance-transform/skeleton approach measures width
            perpendicular to each strand's direction (robust to hair
            orientation) rather than, say, bounding-box dimensions (which would
            overestimate width for anything but a perfectly horizontal or
            vertical hair).

        Args:
            image: BGR image to search for hair (should be denoised but not yet
                hair-removed -- this must run before _remove_hair_and_artifacts
                deletes the very hair being measured).
            kernel_size: Blackhat structuring element size. Defaults to 17,
                tuned to hair's typical thickness relative to a ~500px-wide
                lesion photo.
            threshold: Blackhat response threshold for "this is hair". Defaults
                to 10.

        Returns:
            Estimated hair width in pixels (float), or None if fewer than 50
            hair-like pixels or fewer than 10 usable skeleton points were found
            -- e.g. a photo with no visible hair, or one where dermoscope
            polarization has suppressed surface hair entirely.

        Example:
            >>> detector = MelanomaDetector()
            >>> hair_width_px = detector._measure_hair_width_px(bilateral_filtered_image)
            >>> hair_width_px is None or hair_width_px > 0
            True
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

        if np.sum(hair_mask > 0) < 50:
            return None

        dist = cv2.distanceTransform(hair_mask, cv2.DIST_L2, 5)
        kernel_sk = np.ones((3, 3), np.uint8)
        dist_dilated = cv2.dilate(dist, kernel_sk)
        skeleton = (dist == dist_dilated) & (dist > 0)

        half_widths = dist[skeleton]
        half_widths = half_widths[(half_widths >= 0.5) & (half_widths <= 8.0)]

        if len(half_widths) < 10:
            return None

        return float(np.median(half_widths) * 2)

    def _remove_hair_and_artifacts(self, image: np.ndarray, kernel_size: int = 17, threshold: int = 10) -> np.ndarray:
        """Remove hair strands via blackhat morphology + inpainting (Dull Razor-style).

        What it does:
            Blackhat highlights thin dark structures (hair) against the lighter
            skin background; thresholding that gives a hair mask, which
            cv2.inpaint then fills in using the surrounding skin/lesion texture.

        Why this approach:
            Body hair overlapping a lesion in a dermoscopic photo is a
            well-known confound in automated skin lesion analysis: an
            unremoved hair strand can fracture segmentation into multiple
            disconnected regions, or get mistaken for a genuinely irregular
            border. This "Dull Razor" approach (named after the original 1997
            algorithm it's modeled on) is the standard classical technique for
            this: blackhat morphology is well-suited to hair specifically
            because hairs are thin, elongated, and darker than surrounding
            skin. Uses the same blackhat mask logic as
            _measure_hair_width_px (matching kernel/threshold defaults) so hair
            measurement and hair removal agree on what counts as "hair".

        Args:
            image: BGR image to clean (should already be denoised).
            kernel_size: Blackhat structuring element size. Defaults to 17.
            threshold: Blackhat response threshold for "this is hair". Defaults
                to 10.

        Returns:
            A same-size, same-dtype BGR image with hair strands inpainted out.

        Example:
            >>> detector = MelanomaDetector()
            >>> cleaned = detector._remove_hair_and_artifacts(bilateral_filtered_image)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
        return cv2.inpaint(image, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    def _detect_edges(self, image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
        """Canny edge detection on the segmented (masked) lesion image.

        What it does:
            Converts to grayscale and runs cv2.Canny with a fixed low/high
            hysteresis threshold pair to produce a binary edge map.

        Why this approach:
            Canny is the standard general-purpose edge detector and gives a
            clean, thin edge map suitable for visual inspection of the lesion
            boundary (returned to the caller as one of the pipeline images) --
            it is not used for scoring itself; border-irregularity scoring
            instead uses the segmented contour's geometry directly (see
            _calculate_border_irregularity). Running this on the masked image
            (background zeroed out by segmentation) rather than the full frame
            means the edge map shows only the lesion boundary, not unrelated
            skin texture or lighting gradients elsewhere in the photo.

        Args:
            image: BGR image to detect edges in -- expected to be the masked
                (background-zeroed) output of _segment_lesion.
            low_threshold: Canny's lower hysteresis threshold. Defaults to 50.
            high_threshold: Canny's upper hysteresis threshold. Defaults to 150.

        Returns:
            A same-size, single-channel (uint8) binary edge map.

        Example:
            >>> mask, masked = detector._segment_lesion(hair_removed_image)
            >>> edges = detector._detect_edges(masked)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, low_threshold, high_threshold)

    def _segment_lesion(self, image: np.ndarray):
        """Segment the lesion from surrounding skin using LAB color distance.

        What it does:
            Samples the median LAB color from a rectangular band near the image
            border (assumed to be skin, not lesion) and computes every pixel's
            color distance from that skin tone. Otsu-thresholding that distance
            map separates "looks like skin" from "looks different" (the
            lesion), then morphological close/open cleans up the result and the
            largest connected component is kept as the final mask. If that
            mask ends up implausibly small (under 0.5% of the frame -- LAB
            segmentation failed, e.g. on an unusual lighting condition), falls
            back to grayscale Otsu thresholding instead.

        Why this approach:
            The previous version of this method assumed the lesion was simply
            "the darkest region" in a grayscale image, which works reasonably
            for classic brown/black moles but fails on lighter, pink, or
            red-toned lesions that aren't darker than skin at all. Measuring
            color *distance from this specific photo's own skin tone* (rather
            than absolute darkness, or a generic fixed threshold) adapts to the
            lesion's actual color relative to the person's actual skin in that
            photo's lighting -- a lesion is "different from skin" whether it's
            darker, lighter, or a different hue entirely. LAB color space is
            used (rather than RGB/BGR) because Euclidean distance in LAB
            corresponds much more closely to perceived color difference. The
            grayscale-Otsu fallback exists because LAB segmentation can still
            fail (e.g. if the sampled border band happens to catch lesion
            pixels too), and a degenerate near-empty mask is worse than a rough
            brightness-based guess.

        Args:
            image: BGR image to segment -- expected to be the hair-removed
                image so hair strands don't fracture the mask.

        Returns:
            A tuple of (mask, masked):
                mask: uint8 ndarray, same size as image, 255 inside the
                    detected lesion and 0 elsewhere.
                masked: the input image with everything outside the mask
                    zeroed out (used for edge detection).

        Example:
            >>> mask, masked = detector._segment_lesion(hair_removed_image)
            >>> mask.shape[:2] == hair_removed_image.shape[:2]
            True
        """
        h, w = image.shape[:2]

        border_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(
            border_mask,
            (int(w * 0.05), int(h * 0.05)),
            (int(w * 0.95), int(h * 0.95)),
            255,
            int(min(h, w) * 0.12),
        )

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        border_pixels = lab[border_mask > 0]
        skin_color = np.median(border_pixels, axis=0)

        dist = np.sqrt(np.sum((lab - skin_color) ** 2, axis=2))
        dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.uint8(labels == largest) * 255

        lesion_area = np.sum(mask > 0)
        total_area = h * w
        if lesion_area / total_area < 0.005:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            if num_labels > 1:
                largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                mask = np.uint8(labels == largest) * 255

        masked = cv2.bitwise_and(image, image, mask=mask)
        return mask, masked

    def _pixelate(self, image: np.ndarray, block_size: int = 12) -> np.ndarray:
        """Blockify an image by downscaling then upscaling with nearest-neighbor.

        What it does:
            Shrinks the image by block_size, then scales it back up using
            nearest-neighbor interpolation, producing a chunky "mosaic" version
            where each block_size x block_size region becomes one flat color.

        Why this approach:
            Used by color scoring to smooth out per-pixel sensor noise and fine
            texture before measuring color distances, without blurring across
            genuinely different color regions the way a Gaussian blur would --
            each pixelated block is a hard average, not a gradient.

        Args:
            image: BGR image to pixelate.
            block_size: Side length, in original pixels, of each resulting flat
                block. Larger values coarsen the result more.

        Returns:
            A same-size BGR image with block_size x block_size flat blocks.

        Example:
            >>> detector = MelanomaDetector()
            >>> chunky = detector._pixelate(hair_removed_image, block_size=12)
        """
        h, w = image.shape[:2]
        small = cv2.resize(
            image, (max(1, w // block_size), max(1, h // block_size)), interpolation=cv2.INTER_AREA
        )
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def _sample_skin_color(self, image: np.ndarray, circle_info, lesion_mask: np.ndarray,
                            inner_frac: float = 0.60, outer_frac: float = 0.80) -> np.ndarray:
        """Sample this photo's actual skin tone from a ring around the lesion.

        What it does:
            Using the dermoscope vignette's detected center and radius (from
            _remove_vignette), samples pixels in an annular ring between 60%
            and 80% of that radius, excluding any lesion pixels, and returns
            their median LAB color. Falls back to using the full ring
            (including any lesion overlap) if too few clean pixels are found,
            and to sampling the image corners if the ring itself looks
            implausibly dark (likely still shadowed/vignetted).

        Why this approach:
            Color scoring needs a "what does normal skin look like in this
            photo" baseline to measure the lesion against. Every person's skin
            tone -- and every photo's white balance and lighting -- differs, so
            a fixed/generic skin-color reference would misjudge color
            variation on darker or lighter skin tones alike. Sampling a ring
            specifically within the dermoscope's own field of view (not the
            image corners, which might still be vignette remnants) gives a
            same-photo, same-lighting-condition reference. This is what lets
            downstream "dangerous color" detection (see
            _calculate_color_variation) compare the lesion to this person's own
            skin rather than an assumed default.

        Args:
            image: BGR image to sample from.
            circle_info: (center_x, center_y, radius) from _remove_vignette.
                Must not be None.
            lesion_mask: uint8 lesion mask (255 = lesion pixel) to exclude from
                sampling.
            inner_frac: Inner radius of the sampling ring, as a fraction of the
                vignette radius. Defaults to 0.60.
            outer_frac: Outer radius of the sampling ring, as a fraction of the
                vignette radius. Defaults to 0.80.

        Returns:
            A 3-element LAB color (L, A, B) as a NumPy array representing the
            median sampled skin tone.

        Example:
            >>> no_vignette, circle_info = detector._remove_vignette(original)
            >>> skin_lab = detector._sample_skin_color(original, circle_info, mask)
        """
        cx, cy, radius = circle_info
        h, w = image.shape[:2]

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        ring_mask = ((dist >= radius * inner_frac) & (dist <= radius * outer_frac)).astype(np.uint8) * 255
        ring_mask[lesion_mask > 0] = 0

        n_pixels = int(np.sum(ring_mask > 0))
        if n_pixels < 100:
            ring_mask = ((dist >= radius * inner_frac) & (dist <= radius * outer_frac)).astype(np.uint8) * 255

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        skin_pixels = lab[ring_mask > 0]
        if len(skin_pixels) == 0:
            return np.array([128.0, 128.0, 128.0])
        skin_color = np.median(skin_pixels, axis=0)

        if skin_color[0] < 60:
            cs = min(h, w) // 8
            corners = np.vstack([
                lab[:cs, :cs].reshape(-1, 3),
                lab[:cs, -cs:].reshape(-1, 3),
                lab[-cs:, :cs].reshape(-1, 3),
                lab[-cs:, -cs:].reshape(-1, 3),
            ])
            corners = corners[corners[:, 0] > 60]
            if len(corners) > 10:
                skin_color = np.median(corners, axis=0)
            else:
                bright_pixels = skin_pixels[skin_pixels[:, 0] > np.percentile(skin_pixels[:, 0], 80)]
                if len(bright_pixels) > 10:
                    skin_color = np.median(bright_pixels, axis=0)

        return skin_color

    def _compute_abcde_scores(self, image: np.ndarray, mask: np.ndarray, circle_info, mm_per_px) -> dict:
        """Compute all five ABCDE sub-scores for a segmented lesion.

        What it does:
            Delegates to the four scoring methods (asymmetry, border, color,
            diameter) and packages their results into the dict shape the rest
            of the app expects. If no lesion was segmented at all, returns an
            all-zero/None dict instead of calling the individual scorers.

        Why this approach:
            The ABCDE rule (Asymmetry, Border irregularity, Color variation,
            Diameter, Evolving) is the standard mnemonic dermatologists use for
            a quick visual melanoma screen, so this pipeline scores against the
            same five criteria. "Evolving" is always {"score": None, ...} since
            it describes change over time and cannot be assessed from a single
            static photo. "Diameter" can also be None here -- unlike the
            previous version of this pipeline, which always guessed a relative
            pixel-based diameter, this version reports no diameter score at all
            when hair-based mm calibration isn't possible (see
            _calculate_diameter_score), since a fabricated number would be
            actively misleading.

        Args:
            image: BGR image the color scorer should sample pixels from (the
                hair-removed image, so residual hair doesn't skew color
                measurements).
            mask: uint8 lesion mask from _segment_lesion (255 = lesion pixel).
            circle_info: (center_x, center_y, radius) from _remove_vignette, or
                None if no vignette was detected.
            mm_per_px: Millimeters-per-pixel calibration from
                _measure_hair_width_px, or None if no hair was found to
                calibrate against.

        Returns:
            A dict keyed "asymmetry", "border", "color", "diameter", "evolving",
            each mapping to {"score": float 0-10 or None, "details": dict}. The
            "details" dict holds the raw measurement(s) and clinical concern
            flag behind that score, for transparency and for driving the
            visualization overlays in _build_abcd_visuals.

        Example:
            >>> mask, masked = detector._segment_lesion(hair_removed_image)
            >>> scores = detector._compute_abcde_scores(hair_removed_image, mask, circle_info, mm_per_px)
            >>> scores["asymmetry"]["score"]
            4.27
        """
        if mask is None or np.sum(mask > 0) == 0:
            return {
                "asymmetry": {"score": 0.0, "details": {}},
                "border": {"score": 0.0, "details": {}},
                "color": {"score": 0.0, "details": {}},
                "diameter": {"score": None, "details": {"reason": "no lesion detected"}},
                "evolving": {"score": None, "details": "not assessable from a single static image"},
            }

        asymmetry_score, asymmetry_details = self._calculate_asymmetry(mask)
        border_score, border_details = self._calculate_border_irregularity(mask)
        color_score, color_details = self._calculate_color_variation(image, mask, circle_info)
        diameter_score, diameter_details = self._calculate_diameter_score(mask, mm_per_px)

        return {
            "asymmetry": {"score": asymmetry_score, "details": asymmetry_details},
            "border": {"score": border_score, "details": border_details},
            "color": {"score": color_score, "details": color_details},
            "diameter": {"score": diameter_score, "details": diameter_details},
            "evolving": {"score": None, "details": "not assessable from a single static image"},
        }

    def _calculate_asymmetry(self, mask: np.ndarray):
        """Score 0-10: how much the lesion mask changes when mirrored about its centroid.

        What it does:
            Crops a square region around the lesion's mass centroid (not its
            bounding-box corner), mirrors that crop horizontally and
            vertically, and measures the intersection-over-union (IoU) overlap
            between the original and each mirrored version. High overlap means
            symmetric; low overlap means asymmetric. The two axis results are
            averaged into a single raw asymmetry ratio (0 = perfectly
            symmetric, up to 1 = no overlap at all), then scaled to 0-10 such
            that the clinical concern threshold (0.20) lands at score 5.0.

        Why this approach:
            "Asymmetry" in the clinical ABCDE rule means one half of a mole
            doesn't match the other. Cropping around the *mass centroid*
            (center of pixel mass) rather than the bounding-box center means
            the crop is centered on where the lesion actually is, not skewed
            by an irregular protrusion pulling the bounding box off-center.
            IoU (rather than a plain pixel-count-of-differences ratio) is a
            standard shape-similarity metric that's naturally bounded to
            [0, 1] regardless of lesion size.

        Args:
            mask: uint8 lesion mask (255 = lesion pixel), non-empty.

        Returns:
            A tuple of (score, details):
                score: float 0.0 (symmetric) to 10.0 (highly asymmetric).
                details: dict with "raw_asymmetry_ratio", "centroid_px" (x, y),
                    "concern" (bool, True if raw ratio > 0.20), and
                    "concern_threshold".

        Example:
            >>> mask, masked = detector._segment_lesion(hair_removed_image)
            >>> score, details = detector._calculate_asymmetry(mask)
            >>> details["concern"]
            False
        """
        h, w = mask.shape
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = w // 2, h // 2

        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            half = max(r_max - r_min, c_max - c_min) // 2 + 10
            r0, r1 = max(cy - half, 0), min(cy + half, h)
            c0, c1 = max(cx - half, 0), min(cx + half, w)
            crop = (mask[r0:r1, c0:c1] // 255).astype(np.uint8)
        else:
            crop = np.zeros((1, 1), np.uint8)

        if crop.size == 0 or np.sum(crop) == 0:
            raw_asymmetry = 0.0
        else:
            flip_h = np.fliplr(crop)
            overlap_h = np.sum(crop & flip_h) / (np.sum(crop | flip_h) + 1e-6)
            flip_v = np.flipud(crop)
            overlap_v = np.sum(crop & flip_v) / (np.sum(crop | flip_v) + 1e-6)
            raw_asymmetry = 1 - (overlap_h + overlap_v) / 2

        concern_threshold = 0.20
        concern = bool(raw_asymmetry > concern_threshold)
        score = float(np.clip((raw_asymmetry / concern_threshold) * 5.0, 0.0, 10.0))

        return score, {
            "raw_asymmetry_ratio": round(float(raw_asymmetry), 3),
            "centroid_px": [cx, cy],
            "concern": concern,
            "concern_threshold": concern_threshold,
        }

    def _calculate_border_irregularity(self, mask: np.ndarray):
        """Score 0-10: contour irregularity via circularity (4*pi*area / perimeter^2).

        What it does:
            Finds the lesion's largest contour and computes its circularity --
            1.0 for a perfect circle, lower the more a shape deviates from one
            (a jagged, notched, or elongated outline has more perimeter for its
            enclosed area than a smooth circle does). "Border irregularity" is
            1 minus circularity, then scaled to 0-10 such that the clinical
            concern threshold (0.50) lands at score 5.0.

        Why this approach:
            "Border" in the clinical ABCDE rule refers to ragged, notched, or
            blurred edges -- benign moles tend to have smooth, well-defined
            borders, while melanomas often have irregular ones. Circularity is
            a standard, well-understood shape-regularity metric precisely
            because it's scale-invariant (a small smooth mole and a large
            smooth mole both score near 0 irregularity) and directly sensitive
            to jagged boundaries.

        Args:
            mask: uint8 lesion mask (255 = lesion pixel), non-empty.

        Returns:
            A tuple of (score, details):
                score: float 0.0 (perfectly smooth/circular) to 10.0 (highly
                    irregular).
                details: dict with "raw_border_irregularity", "concern" (bool,
                    True if raw irregularity > 0.50), and "concern_threshold".

        Example:
            >>> mask, masked = detector._segment_lesion(hair_removed_image)
            >>> score, details = detector._calculate_border_irregularity(mask)
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
            raw_border_irregularity = 1 - circularity
        else:
            raw_border_irregularity = 0.0

        concern_threshold = 0.50
        concern = bool(raw_border_irregularity > concern_threshold)
        score = float(np.clip((raw_border_irregularity / concern_threshold) * 5.0, 0.0, 10.0))

        return score, {
            "raw_border_irregularity": round(float(raw_border_irregularity), 3),
            "concern": concern,
            "concern_threshold": concern_threshold,
        }

    def _calculate_color_variation(self, image: np.ndarray, mask: np.ndarray, circle_info):
        """Score 0-10: color variation relative to this photo's own skin tone.

        What it does:
            Measures two independent signals and takes the stronger of the two:
            (1) the coefficient of variation of each lesion pixel's LAB color
            distance from the sampled skin tone (high CV = colors vary a lot
            across the lesion), and (2) the largest fraction of lesion pixels
            matching one of four clinically-named "dangerous" color patterns
            (pink/red, blue-gray, white, black) defined relative to that same
            skin-tone baseline. Each signal is scaled to 0-10 against its own
            clinical concern threshold (0.35 for color-variation CV, 0.08 pixel
            fraction for any single dangerous color), and the score is the
            larger of the two scaled values.

        Why this approach:
            "Color" in the clinical ABCDE rule refers to a lesion showing
            multiple distinct colors, or specific worrying colors (like
            blue-gray or true black), rather than one uniform tan/brown. The
            previous version of this pipeline counted k-means color clusters
            with no reference point, which turned out to be dominated by
            lighting-gradient noise rather than real color signal (see prior
            accuracy validation notes). This version instead measures color
            *relative to the person's own skin tone in this photo*
            (_sample_skin_color), which is both more clinically meaningful
            (colors are assessed the way a dermatologist would -- "does this
            look different from surrounding skin, and in what way") and more
            robust to different skin tones and lighting conditions than a
            fixed color reference would be.

        Args:
            image: BGR image to sample lesion pixel colors from.
            mask: uint8 lesion mask (255 = lesion pixel), non-empty.
            circle_info: (center_x, center_y, radius) from _remove_vignette, or
                None -- if None, skin tone is sampled from the image corners
                instead of a vignette-relative ring.

        Returns:
            A tuple of (score, details):
                score: float 0.0 (uniform, skin-like color) to 10.0 (high
                    color variation or a strongly present dangerous color).
                details: dict with "color_cv", "dangerous_colors_pct" (dict of
                    color name -> pixel fraction, only including colors above
                    their 8% threshold), "concern" (bool), and
                    "skin_lab_baseline" (the sampled reference skin color).

        Example:
            >>> mask, masked = detector._segment_lesion(hair_removed_image)
            >>> score, details = detector._calculate_color_variation(hair_removed_image, mask, circle_info)
            >>> details["dangerous_colors_pct"]
            {}
        """
        if circle_info is not None:
            skin_lab = self._sample_skin_color(image, circle_info, mask)
        else:
            h_img, w_img = image.shape[:2]
            lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
            cs = min(h_img, w_img) // 8
            corners = np.vstack([
                lab_img[:cs, :cs].reshape(-1, 3),
                lab_img[:cs, -cs:].reshape(-1, 3),
                lab_img[-cs:, :cs].reshape(-1, 3),
                lab_img[-cs:, -cs:].reshape(-1, 3),
            ])
            skin_lab = np.median(corners, axis=0)

        pixelated = self._pixelate(image, block_size=12)
        pix_lab = cv2.cvtColor(pixelated, cv2.COLOR_BGR2LAB).astype(np.float32)
        lesion_lab = pix_lab[mask > 0]
        if len(lesion_lab) == 0:
            return 0.0, {"reason": "no lesion pixels"}

        distances = np.sqrt(np.sum((lesion_lab - skin_lab) ** 2, axis=1))
        color_cv = float(np.std(distances)) / (float(np.mean(distances)) + 1e-6)
        cv_concern_threshold = 0.35
        cv_concern = color_cv > cv_concern_threshold

        pix_small = self._pixelate(image, block_size=4)
        pix_small_lab = cv2.cvtColor(pix_small, cv2.COLOR_BGR2LAB).astype(np.float32)
        lesion_lab_dc = pix_small_lab[mask > 0]

        skin_L, skin_A, skin_B = skin_lab
        L_ch, A_ch, B_ch = lesion_lab_dc[:, 0], lesion_lab_dc[:, 1], lesion_lab_dc[:, 2]

        lesion_L_median = float(np.median(L_ch))
        lesion_L_p10 = float(np.percentile(L_ch, 10))

        pink_red_pixels = float(np.mean((A_ch > skin_A + 10) & (L_ch > skin_L - 30)))
        blue_gray_abs = float(np.mean((L_ch < skin_L - 60) & (B_ch < skin_B - 15)))
        blue_gray_rel = (
            lesion_L_p10 < lesion_L_median - 20
            and float(np.percentile(B_ch, 10)) < float(np.median(B_ch)) - 8
        )
        blue_gray_pixels = max(blue_gray_abs, 0.15 if blue_gray_rel else 0.0)
        white_pixels = float(np.mean(L_ch > skin_L + 75))
        black_abs = float(np.mean(L_ch < skin_L - 130))
        black_rel = lesion_L_p10 < lesion_L_median - 35
        black_pixels = max(black_abs, 0.12 if black_rel else 0.0)

        danger_threshold = 0.08
        dangerous_colors = {}
        if pink_red_pixels > danger_threshold:
            dangerous_colors["pink_red"] = round(pink_red_pixels, 3)
        if blue_gray_pixels > danger_threshold:
            dangerous_colors["blue_gray"] = round(blue_gray_pixels, 3)
        if white_pixels > danger_threshold:
            dangerous_colors["white"] = round(white_pixels, 3)
        if black_pixels > danger_threshold:
            dangerous_colors["black"] = round(black_pixels, 3)

        danger_concern = len(dangerous_colors) > 0
        concern = bool(cv_concern or danger_concern)
        max_dangerous_pct = max([pink_red_pixels, blue_gray_pixels, white_pixels, black_pixels], default=0.0)

        cv_score = np.clip((color_cv / cv_concern_threshold) * 5.0, 0.0, 10.0)
        danger_score = np.clip((max_dangerous_pct / danger_threshold) * 5.0, 0.0, 10.0)
        score = float(np.clip(max(cv_score, danger_score), 0.0, 10.0))

        return score, {
            "color_cv": round(color_cv, 3),
            "dangerous_colors_pct": dangerous_colors,
            "concern": concern,
            "skin_lab_baseline": [round(float(v), 1) for v in skin_lab],
        }

    def _calculate_diameter_score(self, mask: np.ndarray, mm_per_px):
        """Score 0-10 (or None): hair-calibrated lesion diameter in millimeters.

        What it does:
            Finds the lesion's minimum enclosing circle and converts its
            diameter from pixels to millimeters using the mm_per_px calibration
            factor from _measure_hair_width_px. Scales the result to 0-10 such
            that the clinical concern threshold (10mm) lands at score 5.0. If
            no calibration factor is available (no hair was detected to
            measure), returns None rather than guessing.

        Why this approach:
            "Diameter" in the clinical ABCDE rule flags lesions larger than
            about 6mm. The previous version of this pipeline had no way to
            convert pixels to real-world millimeters and so scored diameter
            relative to a fixed, arbitrary pixel reference -- meaningless
            across photos taken at different zoom levels. Hair-width
            calibration (see _measure_hair_width_px) gives an actual physical
            scale for this specific photo. When that calibration isn't
            available, this returns None instead of silently falling back to
            an uncalibrated guess, for the same reason "Evolving" is always
            None: a fabricated number here would look precise while being
            meaningless, which is worse than honestly reporting "not
            measurable".

        Args:
            mask: uint8 lesion mask (255 = lesion pixel), non-empty.
            mm_per_px: Millimeters-per-pixel from _measure_hair_width_px, or
                None.

        Returns:
            A tuple of (score, details):
                score: float 0.0-10.0, or None if mm_per_px is None or no
                    lesion contour was found.
                details: dict with "diameter_mm", "mm_per_px_um" (the
                    calibration factor in micrometers/pixel, for transparency),
                    "concern" (bool, True if diameter_mm > 10.0), and
                    "concern_threshold_mm" when a measurement was possible;
                    otherwise just a "reason" key explaining why not.

        Example:
            >>> mask, masked = detector._segment_lesion(hair_removed_image)
            >>> score, details = detector._calculate_diameter_score(mask, mm_per_px)
            >>> score is None or 0.0 <= score <= 10.0
            True
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours and mm_per_px is not None:
            contour = max(contours, key=cv2.contourArea)
            (_, _), radius_px = cv2.minEnclosingCircle(contour)
            diameter_mm = radius_px * 2 * mm_per_px
            concern_threshold = 10.0
            concern = bool(diameter_mm > concern_threshold)
            score = float(np.clip((diameter_mm / concern_threshold) * 5.0, 0.0, 10.0))

            return score, {
                "diameter_mm": round(diameter_mm, 2),
                "mm_per_px_um": round(mm_per_px * 1000, 4),
                "concern": concern,
                "concern_threshold_mm": concern_threshold,
            }

        reason = "no hair detected for scale calibration" if mm_per_px is None else "no lesion contour"
        return None, {"reason": reason}

    def _calculate_risk_score(self, abcde_scores: dict) -> float:
        """Weighted 0-100 risk score from ABCDE sub-scores (each already 0-10).

        What it does:
            Combines the asymmetry, border, color, and diameter sub-scores
            (each 0-10, treating a None diameter as 0) into a single weighted
            sum, then normalizes that sum against the maximum it could possibly
            reach so the final result always falls in 0-100. "Evolving" is
            excluded from the formula entirely, since its score is always None.

        Why this approach:
            The weights (asymmetry 1.3, border 0.1, color 0.5, diameter 0.5)
            mirror the relative emphasis of the classic Stolz ABCD dermoscopy
            rule (the "Total Dermoscopy Score" used in clinical dermoscopy
            training), which weights asymmetry most heavily and border
            irregularity least. This keeps the app's existing 0-100
            continuous risk display (and the UI built around it) even though
            the underlying A/B/C/D measurements are now the more accurate
            ported versions -- only the inputs changed, not this combination
            step. Note this is a heuristic risk indicator, not a validated
            diagnostic score.

        Args:
            abcde_scores: The dict produced by _compute_abcde_scores.

        Returns:
            A float from 0.0 (no risk indicators) to 100.0 (maximum weighted
            risk).

        Example:
            >>> scores = detector._compute_abcde_scores(hair_removed_image, mask, circle_info, mm_per_px)
            >>> detector._calculate_risk_score(scores)
            58.4
        """
        weights = {"asymmetry": 1.3, "border": 0.1, "color": 0.5, "diameter": 0.5}
        max_weighted = sum(weights.values()) * 10.0

        weighted_sum = sum(
            weights[key] * (abcde_scores[key]["score"] or 0.0) for key in weights
        )
        return round((weighted_sum / max_weighted) * 100.0, 2)

    def _build_abcd_visuals(self, original: np.ndarray, mask: np.ndarray, abcde_scores: dict) -> dict:
        """Render four annotated images showing the evidence behind each ABCD score.

        What it does:
            Builds, for each of A/B/C/D:
                Asymmetry: the lesion silhouette mirrored about its centroid,
                    with overlap regions tinted green and mismatched regions
                    tinted red/blue, plus a centroid marker.
                Border: the lesion's detected contour drawn directly on the
                    photo, colored red if border irregularity crossed its
                    concern threshold, green otherwise.
                Color: a heatmap of each pixel's color distance from the
                    sampled skin tone, overlaid on the lesion (blue = close to
                    skin tone, red = far from it).
                Diameter: the lesion's minimum enclosing circle drawn on the
                    photo with its millimeter measurement labeled, colored red
                    or green the same way as the border visual.

        Why this approach:
            Numeric scores alone don't show *why* a lesion scored the way it
            did. These overlays make each measurement visually verifiable --
            a viewer can see the actual mismatched region driving an asymmetry
            score, or the actual off-skin-tone pixels driving a color score,
            rather than trusting an opaque number. This mirrors exactly what
            the original prototype's dashboard showed (see Task: "visibility
            of image transformations"), reimplemented to output plain BGR
            image arrays (rather than a matplotlib figure) so each one can be
            served as its own image in the web app, consistent with how the
            other pipeline-stage images are already returned.

        Args:
            original: BGR image to draw all overlays on top of (the original,
                pre-vignette-removal photo, matching what the reference
                dashboard displayed).
            mask: uint8 lesion mask (255 = lesion pixel) from _segment_lesion.
            abcde_scores: The dict produced by _compute_abcde_scores, used to
                read each criterion's "concern" flag (for red/green coloring)
                and the diameter reading (for its text label).

        Returns:
            A dict with keys "asymmetry", "border", "color", "diameter", each
            a BGR ndarray the same size as `original`.

        Example:
            >>> scores = detector._compute_abcde_scores(hair_removed_image, mask, circle_info, mm_per_px)
            >>> visuals = detector._build_abcd_visuals(original, mask, scores)
            >>> sorted(visuals.keys())
            ['asymmetry', 'border', 'color', 'diameter']
        """
        h, w = original.shape[:2]
        visuals = {}

        # Asymmetry: mirror-overlap overlay (green=overlap, red=original-only, blue=flip-only)
        a_vis = original.copy()
        M = cv2.moments(mask)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else h // 2
        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            half = max(r_max - r_min, c_max - c_min) // 2 + 10
            r0, r1 = max(cy - half, 0), min(cy + half, h)
            c0, c1 = max(cx - half, 0), min(cx + half, w)
            crop = (mask[r0:r1, c0:c1] // 255).astype(np.uint8)
            flip_h = np.fliplr(crop)
            overlap = crop & flip_h
            only_orig = crop & ~flip_h
            only_flip = flip_h & ~crop

            overlay = a_vis[r0:r1, c0:c1].copy()
            overlay[overlap > 0] = (overlay[overlap > 0] * 0.5 + np.array([0, 200, 0]) * 0.5).clip(0, 255).astype(np.uint8)
            overlay[only_orig > 0] = (overlay[only_orig > 0] * 0.5 + np.array([50, 50, 220]) * 0.5).clip(0, 255).astype(np.uint8)
            overlay[only_flip > 0] = (overlay[only_flip > 0] * 0.5 + np.array([220, 50, 50]) * 0.5).clip(0, 255).astype(np.uint8)
            a_vis[r0:r1, c0:c1] = overlay
        cv2.circle(a_vis, (cx, cy), max(5, w // 100), (0, 255, 255), -1)
        visuals["asymmetry"] = a_vis

        # Border: detected contour, colored by concern
        b_vis = original.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            border_concern = abcde_scores["border"]["details"].get("concern", False)
            color_b = (80, 80, 255) if border_concern else (80, 255, 80)
            cv2.drawContours(b_vis, contours, -1, color_b, max(2, w // 300))
        visuals["border"] = b_vis

        # Color: heatmap of distance from sampled skin tone
        lab_full = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
        cs = min(h, w) // 8
        corners = np.vstack([
            lab_full[:cs, :cs].reshape(-1, 3),
            lab_full[:cs, -cs:].reshape(-1, 3),
            lab_full[-cs:, :cs].reshape(-1, 3),
            lab_full[-cs:, -cs:].reshape(-1, 3),
        ])
        skin_lab = np.median(corners, axis=0)
        pix_lab = cv2.cvtColor(self._pixelate(original, block_size=12), cv2.COLOR_BGR2LAB).astype(np.float32)
        dist_map = np.sqrt(np.sum((pix_lab - skin_lab) ** 2, axis=2))
        dist_norm = np.zeros_like(dist_map)
        if np.sum(mask > 0) > 0:
            d_min = dist_map[mask > 0].min()
            d_max = dist_map[mask > 0].max()
            dist_norm[mask > 0] = (dist_map[mask > 0] - d_min) / (d_max - d_min + 1e-6) * 255
        heatmap = cv2.applyColorMap(dist_norm.astype(np.uint8), cv2.COLORMAP_JET)
        c_vis = original.copy()
        c_vis[mask > 0] = (c_vis[mask > 0] * 0.3 + heatmap[mask > 0] * 0.7).clip(0, 255).astype(np.uint8)
        visuals["color"] = c_vis

        # Diameter: minimum enclosing circle + mm label, colored by concern
        d_vis = original.copy()
        if contours:
            contour = max(contours, key=cv2.contourArea)
            (ccx, ccy), rad = cv2.minEnclosingCircle(contour)
            diameter_details = abcde_scores["diameter"]["details"]
            diameter_concern = diameter_details.get("concern", False)
            color_d = (80, 80, 255) if diameter_concern else (80, 255, 80)
            cv2.circle(d_vis, (int(ccx), int(ccy)), int(rad), color_d, max(2, w // 300))
            cv2.circle(d_vis, (int(ccx), int(ccy)), max(4, w // 150), (0, 255, 255), -1)
            d_val = diameter_details.get("diameter_mm")
            label = f"{d_val:.1f}mm" if d_val is not None else "N/A"
            cv2.putText(
                d_vis, label, (int(ccx) - 30, int(ccy) - int(rad) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w / 2000), color_d, max(1, w // 500),
            )
        visuals["diameter"] = d_vis

        return visuals
