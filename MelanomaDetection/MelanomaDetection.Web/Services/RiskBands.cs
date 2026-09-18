namespace MelanomaDetection.Web.Services;

/// <summary>
/// Maps a 0-100 risk score to its band and user-facing copy.
///
/// Thresholds are kept in sync by hand with MelanomaDetection.Python/policy.py,
/// which is the source of truth (it also drives recheck cadence). The client
/// needs its own copy so it can colour and label a score it already holds
/// without a round trip. Change both or neither.
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

    public static string Label(string band) => band switch
    {
        "low" => "Low risk indicators",
        "moderate" => "Some risk indicators present",
        _ => "High risk indicators",
    };

    public static string ShortLabel(string band) => band switch
    {
        "low" => "Low",
        "moderate" => "Moderate",
        _ => "High",
    };

    public static string Recommendation(string band) => band switch
    {
        "low" => "No immediate action needed. Continue routine skin self-exams and keep up with annual dermatologist checkups.",
        "moderate" => "Consider scheduling a dermatologist visit in the coming months to have this spot evaluated.",
        _ => "See a dermatologist as soon as possible for a professional evaluation of this spot.",
    };

    /// <summary>Band for a single 0-10 ABCD sub-score, for the factor bars.</summary>
    public static string FactorBand(double score) => score switch
    {
        < 3.3 => "low",
        < 6.6 => "moderate",
        _ => "high",
    };
}
