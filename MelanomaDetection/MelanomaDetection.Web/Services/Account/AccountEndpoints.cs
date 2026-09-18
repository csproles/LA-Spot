using System.Security.Claims;
using System.Text.Json;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Authentication.Google;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;

namespace MelanomaDetection.Web.Services.Account;

/// <summary>
/// The HTTP endpoints behind sign-in, sign-out and the account's data rights.
///
/// These are plain endpoints rather than Blazor event handlers on purpose: a
/// cookie can only be issued or cleared on a real HTTP response, and an
/// interactive Blazor circuit never produces one. Every state-changing
/// endpoint is a form POST, which also gives it antiforgery validation.
/// </summary>
public static class AccountEndpoints
{
    public const string ExportFileName = "skin-check-export.json";

    public static IEndpointRouteBuilder MapAccountEndpoints(this IEndpointRouteBuilder endpoints)
    {
        var auth = endpoints.MapGroup("/auth");

        auth.MapPost("/login/google", ([FromForm] string? returnUrl) =>
            {
                var properties = new AuthenticationProperties
                {
                    RedirectUri = LocalRedirect.Sanitize(returnUrl),
                };
                return TypedResults.Challenge(properties, [GoogleDefaults.AuthenticationScheme]);
            })
            .AllowAnonymous();

        auth.MapPost("/login/demo", StartDemoAsync).AllowAnonymous();

        auth.MapPost("/logout", LogoutAsync).RequireAuthorization();

        var account = endpoints.MapGroup("/account").RequireAuthorization();

        account.MapGet("/export", ExportAsync);
        account.MapPost("/delete", DeleteAsync);

        return endpoints;
    }

    /// <summary>
    /// "Try the demo": a brand-new, empty account with no Google identity
    /// behind it. The session is browser-scoped and capped at the configured
    /// lifetime; signing out erases the account, the sweeper catches the rest.
    /// </summary>
    private static async Task<Results<SignInHttpResult, NotFound>> StartDemoAsync(
        [FromForm] string? intent,
        IOptions<DemoSettings> demo,
        UserAccountService accounts,
        CancellationToken cancellationToken)
    {
        if (!demo.Value.Enabled || intent != "demo")
        {
            return TypedResults.NotFound();
        }

        var user = await accounts.CreateDemoAsync(cancellationToken);
        var properties = new AuthenticationProperties
        {
            IsPersistent = false,
            ExpiresUtc = DateTimeOffset.UtcNow.Add(demo.Value.SessionLifetime),
            AllowRefresh = false,
            RedirectUri = "/",
        };
        return TypedResults.SignIn(
            AuthenticationSetup.CreatePrincipal(user),
            properties,
            CookieAuthenticationDefaults.AuthenticationScheme);
    }

    /// <summary>
    /// End the session. A demo account is erased on the way out -- it exists
    /// only for the walkthrough -- so its data never outlives the session.
    /// </summary>
    private static async Task<Results<SignOutHttpResult, ProblemHttpResult>> LogoutAsync(
        [FromForm] string? returnUrl,
        ClaimsPrincipal principal,
        UserAccountService accounts,
        ImageProcessingService imageProcessing,
        CancellationToken cancellationToken)
    {
        if (CurrentUser.IsDemo(principal))
        {
            var userId = CurrentUser.ReadUserId(principal)!.Value;
            try
            {
                await imageProcessing.DeleteUserDataAsync(userId);
            }
            catch (ImageProcessingApiException ex)
            {
                return TypedResults.Problem(ex.Message, statusCode: StatusCodes.Status502BadGateway);
            }

            await accounts.DeleteAsync(userId, cancellationToken);
            returnUrl = "/login?status=demo-ended";
        }

        var properties = new AuthenticationProperties
        {
            RedirectUri = LocalRedirect.Sanitize(returnUrl, "/login"),
        };
        return TypedResults.SignOut(properties, [CookieAuthenticationDefaults.AuthenticationScheme]);
    }

    /// <summary>
    /// Everything we hold about the signed-in person, as one JSON download:
    /// the account row here plus the spots, checks and risk profile from the
    /// analysis service. This is the "right of access / portability" path.
    /// </summary>
    private static async Task<Results<FileContentHttpResult, ProblemHttpResult>> ExportAsync(
        ClaimsPrincipal principal,
        UserAccountService accounts,
        ImageProcessingService imageProcessing,
        CancellationToken cancellationToken)
    {
        var userId = CurrentUser.ReadUserId(principal)!.Value;
        var user = await accounts.FindAsync(userId, cancellationToken);
        if (user is null)
        {
            return TypedResults.Problem("Account not found.", statusCode: StatusCodes.Status404NotFound);
        }

        JsonElement healthData;
        try
        {
            healthData = await imageProcessing.ExportUserDataAsync(userId);
        }
        catch (ImageProcessingApiException ex)
        {
            return TypedResults.Problem(ex.Message, statusCode: StatusCodes.Status502BadGateway);
        }

        var export = new
        {
            exportedAtUtc = DateTime.UtcNow,
            account = new
            {
                id = user.Id,
                email = user.Email,
                displayName = user.DisplayName,
                pictureUrl = user.PictureUrl,
                createdAtUtc = user.CreatedAtUtc,
                lastSignInAtUtc = user.LastSignInAtUtc,
                signInProvider = user.IsDemo ? "Demo" : "Google",
            },
            skinCheckData = healthData,
        };

        var bytes = JsonSerializer.SerializeToUtf8Bytes(export, new JsonSerializerOptions { WriteIndented = true });
        return TypedResults.File(bytes, "application/json", ExportFileName);
    }

    /// <summary>
    /// Erase the account and all of its data, then end the session. The
    /// analysis service is cleared first so a failure there leaves the
    /// account in place to retry, rather than orphaning health data.
    /// </summary>
    private static async Task<Results<SignOutHttpResult, BadRequest<string>, ProblemHttpResult>> DeleteAsync(
        [FromForm] string? confirmation,
        ClaimsPrincipal principal,
        UserAccountService accounts,
        ImageProcessingService imageProcessing,
        CancellationToken cancellationToken)
    {
        if (!string.Equals(confirmation, "delete", StringComparison.OrdinalIgnoreCase))
        {
            return TypedResults.BadRequest("Deletion was not confirmed.");
        }

        var userId = CurrentUser.ReadUserId(principal)!.Value;

        try
        {
            await imageProcessing.DeleteUserDataAsync(userId);
        }
        catch (ImageProcessingApiException ex)
        {
            return TypedResults.Problem(ex.Message, statusCode: StatusCodes.Status502BadGateway);
        }

        await accounts.DeleteAsync(userId, cancellationToken);

        var properties = new AuthenticationProperties { RedirectUri = "/login?status=account-deleted" };
        return TypedResults.SignOut(properties, [CookieAuthenticationDefaults.AuthenticationScheme]);
    }
}
