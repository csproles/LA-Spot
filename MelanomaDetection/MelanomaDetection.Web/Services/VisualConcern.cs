namespace MelanomaDetection.Web.Services;

/// <summary>
/// Presentation for V5's own screening result -- the authoritative
/// LOWER/ELEVATED VISUAL CONCERN verdict from
/// MelanomaDetection.Python/pipeline_v5 (frozen decision model, operating
/// threshold 0.25 -- see decision_model.py). V4's pipeline_v4 used the same
/// threshold value but is no longer active. This, not a band cut on the
/// legacy 0-100 risk_score (RiskBands.cs), is what the check-result view
/// shows as the headline: RiskBands' 35/65 thresholds were calibrated for a
/// different, older scoring formula and do not describe what V5's score means.
///
/// A null concern means no lesion could be confidently located in the photo
/// (or, for a historical saved check, that it predates this field) -- it is
/// deliberately its own state, not folded into either verdict.
/// </summary>
public static class VisualConcern
{
    public const string Lower = "LOWER VISUAL CONCERN";
    public const string Elevated = "ELEVATED VISUAL CONCERN";

    public static string Label(string? concern) => concern switch
    {
        Lower => "Lower visual concern",
        Elevated => "Elevated visual concern",
        _ => "Visual concern not available",
    };

    /// <summary>Short word for compact badges (history rows, spot cards,
    /// dashboards) -- NEVER derived from the legacy 0-100 RiskBands cut.
    /// Replaces RiskBands.ShortLabel(RiskBands.Band(score)) wherever a
    /// V5-scored check's badge is rendered, so a secondary surface can't
    /// show a word that contradicts the check's own authoritative verdict.</summary>
    public static string ShortLabel(string? concern) => concern switch
    {
        Lower => "Lower",
        Elevated => "Elevated",
        _ => "N/A",
    };

    /// <summary>
    /// The one-line explanation shown under the label. Deliberately never
    /// says "no immediate action needed" or otherwise tells someone what to
    /// do medically beyond "see a dermatologist" -- this is a screening
    /// prototype's threshold result, not a clearance.
    /// </summary>
    public static string Guidance(string? concern) => concern switch
    {
        Lower => "This photo's visual features did not cross this tool's screening threshold. "
            + "This is not a clearance. Keep up with routine skin self-exams, and see a "
            + "dermatologist if this spot changes or concerns you.",
        Elevated => "This photo's visual features crossed this tool's screening threshold. "
            + "Consider having this spot evaluated by a licensed dermatologist.",
        _ => "A lesion could not be confidently located in this photo, so no visual-concern result "
            + "could be produced. Retaking the photo (better lighting, centering, focus) is recommended.",
    };

    /// <summary>CSS tone class suffix (risk-band-low / risk-band-high / risk-band-muted).</summary>
    public static string CssTone(string? concern) => concern switch
    {
        Lower => "low",
        Elevated => "high",
        _ => "muted",
    };
}
