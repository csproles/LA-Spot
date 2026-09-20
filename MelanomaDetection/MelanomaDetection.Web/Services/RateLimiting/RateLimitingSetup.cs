using System.Globalization;
using System.Threading.RateLimiting;
using MelanomaDetection.Web.Services.Account;

namespace MelanomaDetection.Web.Services.RateLimiting;

/// <summary>Names of the per-endpoint policies, for <c>RequireRateLimiting</c>.</summary>
public static class RateLimitPolicies
{
    /// <summary>Sign-in entry points, per client address. Signing in as a demo creates a database row each time.</summary>
    public const string SignIn = "sign-in";

    /// <summary>Data export and account deletion, per account.</summary>
    public const string AccountData = "account-data";
}

/// <summary>
/// Rate limiting for the plain HTTP surface: sign-in, the account endpoints, and
/// a generous ceiling on everything else. Work done over the Blazor circuit is
/// capped separately by <see cref="OperationRateLimiter"/>.
/// </summary>
public static class RateLimitingSetup
{
    // High enough that loading pages (each pulls a few dozen static files) never
    // trips it, low enough to stop one client hammering the server.
    private const int GlobalRequestsPerMinute = 600;
    private const int SignInAttemptsPerMinute = 10;
    private const int AccountDataRequestsPerMinute = 5;

    public static IServiceCollection AddSkinCheckRateLimiting(this IServiceCollection services)
    {
        services.AddSingleton<OperationRateLimiter>();

        services.AddRateLimiter(options =>
        {
            options.RejectionStatusCode = StatusCodes.Status429TooManyRequests;

            options.OnRejected = async (context, cancellationToken) =>
            {
                var response = context.HttpContext.Response;

                if (context.Lease.TryGetMetadata(MetadataName.RetryAfter, out var retryAfter))
                {
                    response.Headers.RetryAfter =
                        ((int)Math.Ceiling(retryAfter.TotalSeconds)).ToString(CultureInfo.InvariantCulture);
                }

                // A body matters: the status-code-pages middleware re-executes any
                // empty error response as the not-found page, which for a form POST
                // then fails antiforgery and shows a misleading 400 instead of this 429.
                response.ContentType = "text/plain; charset=utf-8";
                await response.WriteAsync("Too many requests. Please wait a moment and try again.", cancellationToken);
            };

            options.GlobalLimiter = PartitionedRateLimiter.Create<HttpContext, string>(context =>
                RateLimitPartition.GetFixedWindowLimiter(
                    PartitionKey(context),
                    _ => Window(GlobalRequestsPerMinute)));

            options.AddPolicy(RateLimitPolicies.SignIn, context =>
                RateLimitPartition.GetFixedWindowLimiter(
                    ClientAddress(context),
                    _ => Window(SignInAttemptsPerMinute)));

            options.AddPolicy(RateLimitPolicies.AccountData, context =>
                RateLimitPartition.GetFixedWindowLimiter(
                    PartitionKey(context),
                    _ => Window(AccountDataRequestsPerMinute)));
        });

        return services;
    }

    // Fixed windows, not sliding: only they report how long until the next permit, which
    // becomes the Retry-After header and the "please wait N seconds" message.
    private static FixedWindowRateLimiterOptions Window(int permitsPerMinute) => new()
    {
        PermitLimit = permitsPerMinute,
        Window = TimeSpan.FromMinutes(1),
        QueueLimit = 0,
        AutoReplenishment = true,
    };

    /// <summary>The signed-in account when there is one (so a shared network doesn't share a cap), else the client address.</summary>
    private static string PartitionKey(HttpContext context)
    {
        var userId = CurrentUser.ReadUserId(context.User);
        return userId is null ? ClientAddress(context) : $"user:{userId}";
    }

    private static string ClientAddress(HttpContext context) =>
        $"ip:{context.Connection.RemoteIpAddress?.ToString() ?? "unknown"}";
}
