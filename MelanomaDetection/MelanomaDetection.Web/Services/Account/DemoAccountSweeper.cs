using Microsoft.Extensions.Options;

namespace MelanomaDetection.Web.Services.Account;

/// <summary>
/// Erases demo accounts whose session was abandoned rather than signed out
/// of, so that no throw-away data accumulates. Runs at startup and then
/// hourly; failures (typically the analysis service not being up yet) are
/// logged and retried on the next pass rather than blocking startup.
/// </summary>
public sealed class DemoAccountSweeper(
    IServiceScopeFactory scopeFactory,
    IOptions<DemoSettings> settings,
    ILogger<DemoAccountSweeper> logger) : BackgroundService
{
    private static readonly TimeSpan Interval = TimeSpan.FromHours(1);

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        // Let the host finish starting (and Flask come up) before the first pass.
        await Task.Delay(TimeSpan.FromSeconds(15), stoppingToken);

        while (!stoppingToken.IsCancellationRequested)
        {
            await SweepAsync(stoppingToken);
            await Task.Delay(Interval, stoppingToken);
        }
    }

    private async Task SweepAsync(CancellationToken cancellationToken)
    {
        using var scope = scopeFactory.CreateScope();
        var accounts = scope.ServiceProvider.GetRequiredService<UserAccountService>();
        var imageProcessing = scope.ServiceProvider.GetRequiredService<ImageProcessingService>();

        List<Guid> stale;
        try
        {
            stale = await accounts.FindStaleDemoAccountsAsync(settings.Value.SessionLifetime, cancellationToken);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            logger.LogWarning(ex, "Could not look up stale demo accounts");
            return;
        }

        foreach (var userId in stale)
        {
            try
            {
                await imageProcessing.DeleteUserDataAsync(userId);
                await accounts.DeleteAsync(userId, cancellationToken);
            }
            catch (ImageProcessingApiException ex)
            {
                logger.LogWarning("Could not erase demo account {UserId} yet: {Reason}", userId, ex.Message);
            }
        }

        if (stale.Count > 0)
        {
            logger.LogInformation("Swept {Count} abandoned demo account(s)", stale.Count);
        }
    }
}
