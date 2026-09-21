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

/// <summary>One row in the History list -- a saved check, with a small thumbnail.</summary>
public class HistoryEntry
{
    [JsonPropertyName("processingId")]
    public string ProcessingId { get; set; } = string.Empty;

    /// <summary>Null for a check that was saved without being filed under a spot.</summary>
    [JsonPropertyName("spotId")]
    public string? SpotId { get; set; }

    [JsonPropertyName("spotLabel")]
    public string? SpotLabel { get; set; }

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

    [JsonPropertyName("spotId")]
    public string? SpotId { get; set; }

    /// <summary>
    /// Base64 PNG of the previous check's thumbnail for the same spot, for the
    /// before/after comparison. Null on a spot's first check.
    /// </summary>
    [JsonPropertyName("priorThumbnail")]
    public string? PriorThumbnail { get; set; }
}

/// <summary>A tracked spot with the aggregate fields the list and dashboard need.</summary>
public class Spot
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;

    [JsonPropertyName("label")]
    public string Label { get; set; } = string.Empty;

    [JsonPropertyName("bodyRegion")]
    public string BodyRegion { get; set; } = string.Empty;

    [JsonPropertyName("createdAt")]
    public DateTimeOffset? CreatedAt { get; set; }

    [JsonPropertyName("checkCount")]
    public int CheckCount { get; set; }

    [JsonPropertyName("lastRiskScore")]
    public double? LastRiskScore { get; set; }

    [JsonPropertyName("lastCheckedAt")]
    public DateTimeOffset? LastCheckedAt { get; set; }

    /// <summary>"up", "down", "flat", or null until the spot has two checks.</summary>
    [JsonPropertyName("trend")]
    public string? Trend { get; set; }

    /// <summary>Days between rechecks for the current risk band; 0 means "now".</summary>
    [JsonPropertyName("cadenceDays")]
    public int? CadenceDays { get; set; }

    [JsonPropertyName("nextDueAt")]
    public DateTimeOffset? NextDueAt { get; set; }

    public bool IsDue => NextDueAt is not null && NextDueAt <= DateTimeOffset.UtcNow;
}

/// <summary>A spot plus its full check timeline, oldest first.</summary>
public class SpotDetail : Spot
{
    [JsonPropertyName("checks")]
    public List<LASpot> Checks { get; set; } = new();
}

/// <summary>One check in a spot's timeline -- the stored numbers plus a thumbnail.</summary>
public class LASpot
{
    [JsonPropertyName("processingId")]
    public string ProcessingId { get; set; } = string.Empty;

    [JsonPropertyName("riskScore")]
    public double RiskScore { get; set; }

    [JsonPropertyName("diameterMm")]
    public double? DiameterMm { get; set; }

    [JsonPropertyName("asymmetry")]
    public double? Asymmetry { get; set; }

    [JsonPropertyName("border")]
    public double? Border { get; set; }

    [JsonPropertyName("color")]
    public double? Color { get; set; }

    [JsonPropertyName("symptoms")]
    public List<string> Symptoms { get; set; } = new();

    [JsonPropertyName("notes")]
    public string Notes { get; set; } = string.Empty;

    [JsonPropertyName("processedAt")]
    public DateTimeOffset? ProcessedAt { get; set; }

    [JsonPropertyName("thumbnail")]
    public string Thumbnail { get; set; } = string.Empty;
}

public class SpotsResponse
{
    [JsonPropertyName("spots")]
    public List<Spot> Spots { get; set; } = new();
}

/// <summary>The user's personal risk factors; drives recheck cadence on the backend.</summary>
public class RiskProfile
{
    /// <summary>False when the user has never filled the profile in.</summary>
    [JsonPropertyName("configured")]
    public bool Configured { get; set; }

    [JsonPropertyName("fullName")]
    public string FullName { get; set; } = string.Empty;

    /// <summary>Free-text city/region, used to give context to UV and seasonal advice.</summary>
    [JsonPropertyName("location")]
    public string Location { get; set; } = string.Empty;

    /// <summary>Self-reported average time outdoors with skin exposed: "low", "moderate", or "high".</summary>
    [JsonPropertyName("sunExposure")]
    public string SunExposure { get; set; } = string.Empty;

    /// <summary>Fitzpatrick skin type, 1 (always burns) to 6 (never burns), or null.</summary>
    [JsonPropertyName("fitzpatrick")]
    public int? Fitzpatrick { get; set; }

    [JsonPropertyName("familyHistory")]
    public bool FamilyHistory { get; set; }

    [JsonPropertyName("blisteringSunburns")]
    public bool BlisteringSunburns { get; set; }

    [JsonPropertyName("manyMoles")]
    public bool ManyMoles { get; set; }

    [JsonPropertyName("recheckReminders")]
    public bool RecheckReminders { get; set; } = true;

    [JsonPropertyName("highRiskAlerts")]
    public bool HighRiskAlerts { get; set; } = true;

    [JsonPropertyName("shareWithDermatologist")]
    public bool ShareWithDermatologist { get; set; } = true;

    [JsonPropertyName("anonymousAnalytics")]
    public bool AnonymousAnalytics { get; set; }

    public bool HasElevatedRiskFactors =>
        Fitzpatrick is <= 2 || FamilyHistory || BlisteringSunburns || ManyMoles || SunExposure == "high";
}
