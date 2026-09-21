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

    private static string AsymmetryCaption(JsonElement details)
    {
        if (!details.TryGetProperty("raw_asymmetry_ratio", out var ratio))
        {
            return string.Empty;
        }

        var threshold = Threshold(details, 0.20);
        return $"Mirror-overlap mismatch ratio: {ratio.GetDouble():F2} "
            + $"(feature reference threshold: {threshold:F2} -- flags this one feature for review; "
            + "it is not the model's overall decision rule)";
    }

    private static string BorderCaption(JsonElement details)
    {
        if (!details.TryGetProperty("raw_border_irregularity", out var raw))
        {
            return string.Empty;
        }

        var threshold = Threshold(details, 0.50);
        return $"Border irregularity: {raw.GetDouble():F2} "
            + $"(feature reference threshold: {threshold:F2} -- flags this one feature for review; "
            + "it is not the model's overall decision rule)";
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
            return $"≈{mm.GetDouble():F1}mm (hair-calibrated measurement)";
        }

        return details.TryGetProperty("reason", out var reason)
            ? $"Not measurable: {reason.GetString()}"
            : string.Empty;
    }

    private static double Threshold(JsonElement details, double fallback) =>
        details.TryGetProperty("concern_threshold", out var threshold) ? threshold.GetDouble() : fallback;
}
