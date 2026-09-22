using System.Security.Claims;
using Microsoft.AspNetCore.Components.Authorization;

namespace MelanomaDetection.Web.Services.Account;

/// <summary>
/// Who is signed in, as seen from a Razor component. Reads the cascading
/// <see cref="AuthenticationStateProvider"/> rather than HttpContext, which
/// Microsoft warns is unreliable inside a Blazor Server circuit. Minimal API
/// endpoints already receive a <see cref="ClaimsPrincipal"/> and use
/// <see cref="ReadUserId"/> on it directly.
/// </summary>
public sealed class CurrentUser(AuthenticationStateProvider authenticationStateProvider)
{
    public async ValueTask<Guid?> GetUserIdAsync()
    {
        var state = await authenticationStateProvider.GetAuthenticationStateAsync();
        return ReadUserId(state.User);
    }

    public async ValueTask<bool> IsProviderAsync()
    {
        var state = await authenticationStateProvider.GetAuthenticationStateAsync();
        return IsProvider(state.User);
    }

    /// <summary>The signed-in person's display name, straight from the claim -- no DB round trip needed.</summary>
    public async ValueTask<string?> GetDisplayNameAsync()
    {
        var state = await authenticationStateProvider.GetAuthenticationStateAsync();
        return state.User.Identity?.Name;
    }

    /// <summary>True when the principal is a throw-away demo session.</summary>
    public static bool IsDemo(ClaimsPrincipal principal) => principal.HasClaim(AppClaimTypes.Demo, "true");

    /// <summary>True for a dermatologist account; see <see cref="Data.AppUser.IsProvider"/>.</summary>
    public static bool IsProvider(ClaimsPrincipal principal) => principal.HasClaim(AppClaimTypes.Provider, "true");

    /// <summary>The app's own account id from a principal, or null when it isn't a signed-in user of ours.</summary>
    public static Guid? ReadUserId(ClaimsPrincipal principal)
    {
        var value = principal.FindFirstValue(AppClaimTypes.UserId);
        return Guid.TryParse(value, out var id) ? id : null;
    }
}
