using System.Threading.RateLimiting;

namespace MelanomaDetection.Web.Services.RateLimiting;

/// <summary>The actions that cost something real, and so get their own per-person cap.</summary>
public enum LimitedOperation
{
    /// <summary>Photo analysis: CPU on the analysis service.</summary>
    AnalyzePhoto,

    /// <summary>The plain-language explanation: one OpenAI call per result.</summary>
    ExplainResults,

    /// <summary>The Find Care lookup: up to nine requests to the government registries.</summary>
    FindProviders,
}

/// <summary>
/// A per-person cap on the expensive actions.
///
/// ASP.NET's rate-limiting middleware only sees HTTP requests, but nearly
/// everything in this app happens over the Blazor circuit's single long-lived
/// connection -- a person could click "Analyze" as fast as they liked without
/// ever producing a request it could count. So the caps live where the work
/// is requested instead, and are enforced by the services and pages that do it.
///
/// The analysis service enforces slightly higher caps of its own for anything
/// that reaches it directly; these are lower so people see a friendly message
/// from here first.
/// </summary>
public sealed class OperationRateLimiter : IDisposable
{
    private static readonly IReadOnlyDictionary<LimitedOperation, (int Permits, TimeSpan Window)> Rules =
        new Dictionary<LimitedOperation, (int, TimeSpan)>
        {
            [LimitedOperation.AnalyzePhoto] = (8, TimeSpan.FromMinutes(1)),
            [LimitedOperation.ExplainResults] = (4, TimeSpan.FromMinutes(1)),
            [LimitedOperation.FindProviders] = (10, TimeSpan.FromMinutes(1)),
        };

    private readonly PartitionedRateLimiter<(LimitedOperation Operation, Guid UserId)> _limiter =
        PartitionedRateLimiter.Create<(LimitedOperation Operation, Guid UserId), (LimitedOperation, Guid)>(key =>
        {
            var (permits, window) = Rules[key.Operation];
            return RateLimitPartition.GetFixedWindowLimiter(key, _ => new FixedWindowRateLimiterOptions
            {
                PermitLimit = permits,
                Window = window,
                QueueLimit = 0,
                AutoReplenishment = true,
            });
        });

    /// <summary>Counts one use. Returns null when allowed, otherwise how long to wait before trying again.</summary>
    public TimeSpan? TryAcquire(LimitedOperation operation, Guid userId)
    {
        using var lease = _limiter.AttemptAcquire((operation, userId));
        if (lease.IsAcquired)
        {
            return null;
        }

        return lease.TryGetMetadata(MetadataName.RetryAfter, out var retryAfter)
            ? retryAfter
            : TimeSpan.FromSeconds(30);
    }

    /// <summary>A message that is safe to show the person who was turned away.</summary>
    public static string WaitMessage(TimeSpan retryAfter)
    {
        var seconds = Math.Max(1, (int)Math.Ceiling(retryAfter.TotalSeconds));
        return $"You're going a little fast. Please wait {seconds} second{(seconds == 1 ? "" : "s")} and try again.";
    }

    public void Dispose() => _limiter.Dispose();
}
