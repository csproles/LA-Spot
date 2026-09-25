"""Image preprocessing and visualization helpers shared by the V5 detector.

`v5_detector.V5Detector` builds one `MelanomaDetector` and reuses its generic image-processing
steps by composition: vignette removal, salt-and-pepper denoising, bilateral filtering, hair
and artifact removal, edge detection, and the four annotated ABCD evidence images. Nothing
here decides anything about a lesion. The scoring and segmentation this class used to carry
(the pre-V5 classical pipeline) were removed once V5 replaced them.
"""

import cv2
import numpy as np


class MelanomaDetector:
    """The image-processing steps V5Detector reuses (see the module docstring).

    Stateless: every parameter lives on the method that uses it, and one instance is created at
    startup and reused for every request.
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
            skin.

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
            it is not used for scoring itself. Running this on the masked image
            (background zeroed out by segmentation) rather than the full frame
            means the edge map shows only the lesion boundary, not unrelated
            skin texture or lighting gradients elsewhere in the photo.

        Args:
            image: BGR image to detect edges in -- expected to be the masked
                (background-zeroed) lesion image.
            low_threshold: Canny's lower hysteresis threshold. Defaults to 50.
            high_threshold: Canny's upper hysteresis threshold. Defaults to 150.

        Returns:
            A same-size, single-channel (uint8) binary edge map.

        Example:
            >>> edges = detector._detect_edges(masked_lesion_image)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, low_threshold, high_threshold)

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
            mask: uint8 lesion mask (255 = lesion pixel) from the segmentation step.
            abcde_scores: The per-criterion scores dict (as V5Detector builds it), used to
                read each criterion's "concern" flag (for red/green coloring)
                and the diameter reading (for its text label).

        Returns:
            A dict with keys "asymmetry", "border", "color", "diameter", each
            a BGR ndarray the same size as `original`.

        Example:
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
