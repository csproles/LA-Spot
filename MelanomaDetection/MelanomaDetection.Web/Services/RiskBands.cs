namespace MelanomaDetection.Web.Services;

/// <summary>
/// Maps a 0-100 risk score to a band, for coloring trend visualizations only
/// (the Spots dashboard sparkline, body map, stat tiles) that plot a bare
/// historical score at a glance across many checks.
///
/// This is NOT used for the check-result headline verdict or for recheck
/// cadence -- V4's own overall_visual_concern is authoritative for both (see
/// Services/VisualConcern.cs and MelanomaDetection.Python/policy.py). These
/// 35/65 thresholds were calibrated for the old classical detector's
/// weighted-sum score and do not line up with V4's decision-model score
/// (whose own operating threshold, 0.25, falls inside this "low" band), so
/// they must never be presented as a medical verdict.
/// </summary>
public static class RiskBands
{
    public const double LowMax = 35.0;
    public const double ModerateMax = 65.0;

    public static string Band(double? riskScore) => riskScore switch
    {
        null => "low",
        < LowMax => "low",
        < ModerateMax => "moderate",
        _ => "high",
    };

    public static string ShortLabel(string band) => band switch
    {
        "low" => "Low",
        "moderate" => "Moderate",
        _ => "High",
    };

    /// <summary>Band for a single 0-10 ABCD sub-score, for the factor bars.</summary>
    public static string FactorBand(double score) => score switch
    {
        < 3.3 => "low",
        < 6.6 => "moderate",
        _ => "high",
    };
}
