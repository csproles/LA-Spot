using System.Text.Json;

namespace MelanomaDetection.Web.Services;

/// <summary>
/// Typed reads over the evolving score's loosely-shaped "details" JSON (built
/// by evolution.py). Every key is optional: growth is withheld unless both
/// photos carried a hair-based mm calibration, and nothing at all is reported
/// on a spot's first check.
/// </summary>
public readonly struct EvolutionDetails(JsonElement details)
{
    private readonly JsonElement _details = details;

    private bool IsObject => _details.ValueKind == JsonValueKind.Object;

    public string? Text(string key) =>
        IsObject && _details.TryGetProperty(key, out var value)
            ? value.ValueKind == JsonValueKind.String ? value.GetString() : value.ToString()
            : null;

    public double? Number(string key) =>
        IsObject
        && _details.TryGetProperty(key, out var value)
        && value.ValueKind == JsonValueKind.Number
            ? value.GetDouble()
            : null;

    /// <summary>Whether there was a previous photo to compare against at all.</summary>
    public bool HasPriorCheck => Text("compared_with") is not null;

    /// <summary>How far past its concern threshold a measured difference is.</summary>
    public static string Tone(double measured, double threshold) =>
        measured >= threshold ? "concern" : measured >= threshold * 0.5 ? "watch" : "stable";
}
