using System.Text.Json;
using MelanomaDetection.Web.Models;

namespace MelanomaDetection.Web.Services;

/// <summary>
/// Turns a score's raw "details" JSON from the Python pipeline into the one
/// short line the UI shows under it. The shapes differ per criterion and every
/// key is optional, so each case is handled explicitly and an unrecognised
/// payload yields no caption rather than a broken one.
/// </summary>
public static class AbcdeDetails
{
    public static string Caption(string criterion, AbcdeScore score)
    {
        var details = score.Details;
        if (details.ValueKind != JsonValueKind.Object)
        {
            return string.Empty;
        }

        return criterion switch
        {
            "asymmetry" => AsymmetryCaption(details),
            "border" => BorderCaption(details),
            "color" => ColorCaption(details),
            "diameter" => DiameterCaption(details),
            _ => string.Empty,
        };
    }

    /// <summary>Extended, hover-only explanation for a criterion's caption.
    /// Only diameter uses this today -- see DiameterCaption's px/relative
    /// display, which needs more room than the one-line caption has for why
    /// these aren't millimeters.</summary>
    public static string Tooltip(string criterion, AbcdeScore score)
    {
        var details = score.Details;
        if (details.ValueKind != JsonValueKind.Object)
        {
            return string.Empty;
        }

        return criterion switch
        {
            "diameter" => DiameterTooltip(details),
            _ => string.Empty,
        };
    }

    private static string AsymmetryCaption(JsonElement details)
    {
        if (!details.TryGetProperty("raw_asymmetry_ratio", out var ratio))
        {
            return string.Empty;
        }

        var threshold = Threshold(details, 0.20);
        return $"Mirror-overlap mismatch ratio: {ratio.GetDouble():F2} (concern threshold: {threshold:F2})";
    }

    private static string BorderCaption(JsonElement details)
    {
        if (!details.TryGetProperty("raw_border_irregularity", out var raw))
        {
            return string.Empty;
        }

        var threshold = Threshold(details, 0.50);
        return $"Border irregularity: {raw.GetDouble():F2} (concern threshold: {threshold:F2})";
    }

    private static string ColorCaption(JsonElement details)
    {
        var line = details.TryGetProperty("color_cv", out var cv)
            ? $"Color variation (CV): {cv.GetDouble():F2}"
            : string.Empty;

        if (!details.TryGetProperty("dangerous_colors_pct", out var dangerous)
            || dangerous.ValueKind != JsonValueKind.Object)
        {
            return line;
        }

        var parts = dangerous.EnumerateObject()
            .Select(property => $"{property.Name.Replace('_', '-')} ({property.Value.GetDouble():P0})")
            .ToList();

        if (parts.Count == 0)
        {
            return line;
        }

        var found = $"Dangerous colors detected: {string.Join(", ", parts)}";
        return line.Length > 0 ? $"{line} | {found}" : found;
    }

    private static string DiameterCaption(JsonElement details)
    {
        if (details.TryGetProperty("diameter_mm", out var mm))
        {
            // Legacy only -- no active pipeline (V4 or V5) ever populates this
            // key; kept only so a pre-V4 saved check still renders correctly.
            // Do NOT restore hair-width mm calibration for V5.
            return $"≈{mm.GetDouble():F1}mm (hair-calibrated measurement)";
        }

        if (details.TryGetProperty("lesion_size_px", out var px) && px.ValueKind == JsonValueKind.Number)
        {
            var relative = details.TryGetProperty("relative_size_pct", out var pct)
                && pct.ValueKind == JsonValueKind.Number
                ? $" · Relative to photo: {pct.GetDouble():F1}% of image width"
                : string.Empty;
            return $"Lesion diameter: {px.GetDouble():F0} px{relative}. "
                + "Physical diameter requires a scale reference.";
        }

        return details.TryGetProperty("reason", out var reason)
            ? $"Not measurable: {reason.GetString()}"
            : string.Empty;
    }

    private static string DiameterTooltip(JsonElement details)
    {
        if (!details.TryGetProperty("lesion_size_px", out var px) || px.ValueKind != JsonValueKind.Number)
        {
            return string.Empty;
        }

        return "Pixel and relative measurements depend on this photo's framing, crop, zoom, and "
            + "camera distance from the skin, so they are not millimeters and cannot be compared "
            + "across photos taken differently. A physical (mm) measurement would require a scale "
            + "reference visible in the photo, such as a ruler, which this pipeline does not use.";
    }

    private static double Threshold(JsonElement details, double fallback) =>
        details.TryGetProperty("concern_threshold", out var threshold) ? threshold.GetDouble() : fallback;
}
