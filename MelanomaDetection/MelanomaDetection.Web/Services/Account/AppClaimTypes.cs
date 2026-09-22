namespace MelanomaDetection.Web.Services.Account;

/// <summary>
/// Claim names this app adds to the sign-in cookie on top of the standard
/// <see cref="System.Security.Claims.ClaimTypes"/> ones Google supplies.
/// </summary>
public static class AppClaimTypes
{
    /// <summary>The <see cref="Data.AppUser.Id"/> of the signed-in account.</summary>
    public const string UserId = "skincheck:user_id";

    /// <summary>Profile picture URL, when the identity provider sent one.</summary>
    public const string Picture = "skincheck:picture";

    /// <summary>Google's <c>email_verified</c> flag, present only on the ticket while signing in.</summary>
    public const string EmailVerified = "skincheck:email_verified";

    /// <summary>Present (value "true") on a demo session; see <see cref="Data.AppUser.IsDemo"/>.</summary>
    public const string Demo = "skincheck:demo";

    /// <summary>Present (value "true") for a dermatologist account; see <see cref="Data.AppUser.IsProvider"/>.</summary>
    public const string Provider = "skincheck:provider";
}
