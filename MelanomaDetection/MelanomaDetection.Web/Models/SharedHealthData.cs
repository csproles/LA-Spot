using System.Text.Json.Serialization;

namespace MelanomaDetection.Web.Models;

/// <summary>A patient's export (GET /api/account/export), as shown to a provider they shared it with.</summary>
public class SharedHealthData
{
    [JsonPropertyName("profile")]
    public RiskProfile? Profile { get; set; }

    [JsonPropertyName("spots")]
    public List<Spot> Spots { get; set; } = new();

    [JsonPropertyName("checks")]
    public List<SharedCheck> Checks { get; set; } = new();
}

public class SharedCheck
{
    [JsonPropertyName("processingId")]
    public string ProcessingId { get; set; } = string.Empty;

    [JsonPropertyName("spotId")]
    public string? SpotId { get; set; }

    [JsonPropertyName("riskScore")]
    public double RiskScore { get; set; }

    [JsonPropertyName("overallVisualConcern")]
    public string? OverallVisualConcern { get; set; }

    [JsonPropertyName("diameterMm")]
    public double? DiameterMm { get; set; }

    [JsonPropertyName("asymmetry")]
    public double? Asymmetry { get; set; }

    [JsonPropertyName("border")]
    public double? Border { get; set; }

    [JsonPropertyName("color")]
    public double? Color { get; set; }

    [JsonPropertyName("location")]
    public string? Location { get; set; }

    [JsonPropertyName("symptoms")]
    public List<string> Symptoms { get; set; } = new();

    [JsonPropertyName("notes")]
    public string? Notes { get; set; }

    [JsonPropertyName("processedAt")]
    public DateTimeOffset? ProcessedAt { get; set; }

    /// <summary>Base64 PNG, or null.</summary>
    [JsonPropertyName("thumbnailPng")]
    public string? ThumbnailPng { get; set; }
}
