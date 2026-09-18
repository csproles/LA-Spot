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

    /// <summary>Present (value "true") on a demo session; see <see cref="Data.AppUser.IsDemo"/>.</summary>
    public const string Demo = "skincheck:demo";
}
