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

/// <summary>
/// Response from POST /predict -- the Kaggle-trained CNN+CatBoost melanoma
/// risk model (risk_model.py), separate from V5's own visual-concern pipeline
/// above. RiskScore is CatBoost's raw mean-of-5-folds output (0-1), not a
/// calibrated probability of malignancy -- see risk_model.RISK_THRESHOLDS'
/// own TODO. ProcessingId lets the existing POST /api/image/explain/{id} be
/// reused for this pipeline's results too.
/// </summary>
public class PredictResponse
{
    [JsonPropertyName("risk_score")]
    public double RiskScore { get; set; }

    /// <summary>"low", "medium", or "high" -- placeholder bands, not clinically validated cut points.</summary>
    [JsonPropertyName("risk_level")]
    public string RiskLevel { get; set; } = string.Empty;

    [JsonPropertyName("abcd_features")]
    public Dictionary<string, double?> AbcdFeatures { get; set; } = new();

    /// <summary>False means no lesion could be segmented in the photo -- every score above is
    /// still returned (NaN ABCD features became a "missing" input to CatBoost, not a rejection),
    /// but should be shown as less reliable.</summary>
    [JsonPropertyName("yolo_found_lesion")]
    public bool YoloFoundLesion { get; set; }

    [JsonPropertyName("fold_scores")]
    public List<double> FoldScores { get; set; } = new();

    [JsonPropertyName("processingId")]
    public string ProcessingId { get; set; } = string.Empty;
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

    /// <summary>V5's own result for this check, or null (no detection at
    /// save time is "NO_DETECTION", never null; null means this check
    /// predates the field). Authoritative -- never re-derive a verdict from
    /// RiskScore via RiskBands here.</summary>
    [JsonPropertyName("overallVisualConcern")]
    public string? OverallVisualConcern { get; set; }

    /// <summary>&gt;1 means only the primary (highest-confidence) YOLO
    /// instance was analyzed at save time -- see MultiLesionNotice.razor
    /// for the live-analysis equivalent of this caveat.</summary>
    [JsonPropertyName("numLesionInstances")]
    public int? NumLesionInstances { get; set; }

    [JsonPropertyName("processedAt")]
    public DateTimeOffset? ProcessedAt { get; set; }

    [JsonPropertyName("thumbnail")]
    public string Thumbnail { get; set; } = string.Empty;
}

public class HistoryResponse
{
    [JsonPropertyName("entries")]
    public List<HistoryEntry> Entries { get; set; } = new();

    /// <summary>Total checks the account has, across every page. Only set when the
    /// request passed limit/offset -- see ImageProcessingService.GetHistoryAsync.</summary>
    [JsonPropertyName("total")]
    public int? Total { get; set; }
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

    /// <summary>
    /// Every YOLO-detected instance outlined on the original photo (present
    /// only when MultiLesionDetected is true). The primary instance -- the
    /// only one analyzed -- is drawn distinctly from the others, which are
    /// outlined only, never analyzed.
    /// </summary>
    [JsonPropertyName("multi_instance_overlay")]
    public string? MultiInstanceOverlay { get; set; }

    [JsonPropertyName("abcde_scores")]
    public AbcdeScores AbcdeScores { get; set; } = new();

    [JsonPropertyName("risk_score")]
    public double RiskScore { get; set; }

    /// <summary>
    /// V5's own screening result -- "LOWER VISUAL CONCERN", "ELEVATED
    /// VISUAL CONCERN", or "NO_DETECTION" (see NoDetection below). This is
    /// the authoritative verdict; see Services/VisualConcern.cs. It is not a
    /// diagnosis and RiskScore is not a probability of one.
    /// </summary>
    [JsonPropertyName("overall_visual_concern")]
    public string? OverallVisualConcern { get; set; }

    /// <summary>True when no lesion could be located in this photo. Prefer
    /// this over comparing OverallVisualConcern to a string -- see
    /// policy.CONCERN_NO_DETECTION.</summary>
    [JsonPropertyName("no_detection")]
    public bool NoDetection { get; set; }

    [JsonPropertyName("num_lesion_instances")]
    public int NumLesionInstances { get; set; }

    [JsonPropertyName("multi_lesion_detected")]
    public bool MultiLesionDetected { get; set; }

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

    /// <summary>V5's result on the spot's most recent check, or null if unknown/no detection.</summary>
    [JsonPropertyName("lastOverallVisualConcern")]
    public string? LastOverallVisualConcern { get; set; }

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

    /// <summary>V5's own result for this check, or null (see HistoryEntry's
    /// OverallVisualConcern for the same NO_DETECTION-vs-null distinction).</summary>
    [JsonPropertyName("overallVisualConcern")]
    public string? OverallVisualConcern { get; set; }

    /// <summary>How many YOLO instances this check's photo had, or null for
    /// a check saved before this was tracked. &gt;1 means only the primary
    /// (highest-confidence) instance was actually analyzed.</summary>
    [JsonPropertyName("numLesionInstances")]
    public int? NumLesionInstances { get; set; }

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
