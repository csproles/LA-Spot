using MelanomaDetection.Web.Models;

namespace MelanomaDetection.Web.Services;

/// <summary>
/// The optional metadata the AI risk model (POST /predict) takes, derived from what the
/// app already knows instead of asked for on every check: age and sex from the risk profile,
/// body site from the spot's body-map region.
/// </summary>
public static class RiskModelInputs
{
    /// <summary>Age this year, or null when no plausible year of birth is on file.</summary>
    public static int? Age(RiskProfile? profile, DateOnly today)
    {
        if (profile?.BirthYear is not { } year || year > today.Year)
        {
            return null;
        }

        var age = today.Year - year;
        return age <= 120 ? age : null;
    }

    /// <summary>"female" / "male", or null when not given.</summary>
    public static string? Sex(RiskProfile? profile) =>
        profile?.Sex is "female" or "male" ? profile.Sex : null;

    /// <summary>
    /// The model's anatomical site for a body-map region (BodyMap.Regions). The two torso regions
    /// cover both chest and back, so they need the side the spot was picked on; without it (spots
    /// created before the side was recorded) the site is left unknown rather than guessed.
    /// </summary>
    public static string? BodySite(string? region, string? side) => region switch
    {
        "Head/Neck" => "head/neck",
        "Left Arm" or "Right Arm" => "upper extremity",
        "Left Leg" or "Right Leg" => "lower extremity",
        "Chest/Upper Back" or "Abdomen/Lower Back" => side switch
        {
            "front" => "anterior torso",
            "back" => "posterior torso",
            _ => null,
        },
        _ => null,
    };
}
