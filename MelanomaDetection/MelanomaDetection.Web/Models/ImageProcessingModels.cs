using System.Text.Json;
using System.Text.Json.Serialization;

namespace MelanomaDetection.Web.Models;

public class ProcessImageResponse
{
    [JsonPropertyName("processingId")]
    public string ProcessingId { get; set; } = string.Empty;
}

/// <summary>Shape of Flask's {"error": "..."} responses (400/404/413/500/502).</summary>
public class ErrorResponse
{
    [JsonPropertyName("error")]
    public string? Error { get; set; }
}

public class ExplainResponse
{
    [JsonPropertyName("explanation")]
    public string Explanation { get; set; } = string.Empty;
}

public class SaveResponse
{
    [JsonPropertyName("saved")]
    public bool Saved { get; set; }
}

/// <summary>One row in the History list -- a saved check, with a small thumbnail.</summary>
public class HistoryEntry
{
    [JsonPropertyName("processingId")]
    public string ProcessingId { get; set; } = string.Empty;

    [JsonPropertyName("location")]
    public string Location { get; set; } = string.Empty;

    [JsonPropertyName("symptoms")]
    public List<string> Symptoms { get; set; } = new();

    [JsonPropertyName("notes")]
    public string Notes { get; set; } = string.Empty;

    [JsonPropertyName("riskScore")]
    public double RiskScore { get; set; }

    [JsonPropertyName("processedAt")]
    public DateTimeOffset? ProcessedAt { get; set; }

    [JsonPropertyName("thumbnail")]
    public string Thumbnail { get; set; } = string.Empty;
}

public class HistoryResponse
{
    [JsonPropertyName("entries")]
    public List<HistoryEntry> Entries { get; set; } = new();
}

public class AbcdeScore
{
    [JsonPropertyName("score")]
    public double? Score { get; set; }

    [JsonPropertyName("details")]
    public JsonElement Details { get; set; }
}

public class AbcdeScores
{
    [JsonPropertyName("asymmetry")]
    public AbcdeScore Asymmetry { get; set; } = new();

    [JsonPropertyName("border")]
    public AbcdeScore Border { get; set; } = new();

    [JsonPropertyName("color")]
    public AbcdeScore Color { get; set; } = new();

    [JsonPropertyName("diameter")]
    public AbcdeScore Diameter { get; set; } = new();

    [JsonPropertyName("evolving")]
    public AbcdeScore Evolving { get; set; } = new();
}

public class ImageProcessingResults
{
    [JsonPropertyName("processingId")]
    public string ProcessingId { get; set; } = string.Empty;

    [JsonPropertyName("original")]
    public string Original { get; set; } = string.Empty;

    [JsonPropertyName("bilateral_filtered")]
    public string BilateralFiltered { get; set; } = string.Empty;

    [JsonPropertyName("noise_removed")]
    public string NoiseRemoved { get; set; } = string.Empty;

    [JsonPropertyName("hair_removed")]
    public string HairRemoved { get; set; } = string.Empty;

    [JsonPropertyName("segmentation")]
    public string Segmentation { get; set; } = string.Empty;

    [JsonPropertyName("edges")]
    public string Edges { get; set; } = string.Empty;

    [JsonPropertyName("asymmetry_visual")]
    public string AsymmetryVisual { get; set; } = string.Empty;

    [JsonPropertyName("border_visual")]
    public string BorderVisual { get; set; } = string.Empty;

    [JsonPropertyName("color_visual")]
    public string ColorVisual { get; set; } = string.Empty;

    [JsonPropertyName("diameter_visual")]
    public string DiameterVisual { get; set; } = string.Empty;

    [JsonPropertyName("abcde_scores")]
    public AbcdeScores AbcdeScores { get; set; } = new();

    [JsonPropertyName("risk_score")]
    public double RiskScore { get; set; }

    [JsonPropertyName("location")]
    public string Location { get; set; } = string.Empty;

    [JsonPropertyName("symptoms")]
    public List<string> Symptoms { get; set; } = new();

    [JsonPropertyName("notes")]
    public string Notes { get; set; } = string.Empty;
}
