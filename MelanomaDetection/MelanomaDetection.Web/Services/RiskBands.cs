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
        "low" => "Low risk signs",
        "moderate" => "Some risk signs",
        _ => "High risk signs",
    };

    public static string ShortLabel(string band) => band switch
    {
        "low" => "Low",
        "moderate" => "Moderate",
        _ => "High",
    };

    public static string Recommendation(string band) => band switch
    {
        "low" => "You don't need to do anything right now. Keep checking your skin every so often, and see a skin doctor (a dermatologist) once a year.",
        "moderate" => "Think about seeing a skin doctor (a dermatologist) in the next few months to have this spot looked at.",
        _ => "Please see a skin doctor (a dermatologist) as soon as you can to have this spot looked at.",
    };

    /// <summary>Band for a single 0-10 ABCD sub-score, for the factor bars.</summary>
    public static string FactorBand(double score) => score switch
    {
        < 3.3 => "low",
        < 6.6 => "moderate",
        _ => "high",
    };
}
