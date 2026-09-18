namespace MelanomaDetection.Web.Data;

/// <summary>
/// One signed-in person. Deliberately minimal: Google is the identity provider,
/// so no password, no tokens -- just what we need to recognise the same person
/// next time and to address them by name. The medical data (spots, checks,
/// risk profile) lives in the Flask store keyed by <see cref="Id"/>.
/// </summary>
public class AppUser
{
    /// <summary>
    /// Our own stable key, sent to the Flask API as the owner of every row.
    /// Never the Google subject, so the identity provider can change without
    /// re-keying anyone's history.
    /// </summary>
    public Guid Id { get; set; }

    /// <summary>
    /// Google's stable, unique "sub" claim for this account. Demo accounts
    /// carry a synthetic "demo:" value so the unique index still holds.
    /// </summary>
    public required string GoogleSubject { get; set; }

    /// <summary>Email as Google reported it at the most recent sign-in (verified by Google). Empty for demo accounts.</summary>
    public required string Email { get; set; }

    public string DisplayName { get; set; } = string.Empty;

    /// <summary>URL of the Google profile picture, or null when Google didn't send one.</summary>
    public string? PictureUrl { get; set; }

    public DateTime CreatedAtUtc { get; set; }

    public DateTime LastSignInAtUtc { get; set; }

    /// <summary>
    /// A throw-away account entered from the sign-in page's "Try the demo".
    /// Starts with nothing, is erased when the person signs out, and is swept
    /// if abandoned. Never tied to a Google identity.
    /// </summary>
    public bool IsDemo { get; set; }
}
