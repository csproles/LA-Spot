using MelanomaDetection.Web.Data;

namespace MelanomaDetection.Web.Services.Scheduling;

/// <summary>
/// Fires the 24h/1h visit reminders. Runs every 5 minutes, same shape as
/// <see cref="Account.DemoAccountSweeper"/> -- this app's existing pattern
/// for periodic work, reused rather than adding a job-queue dependency.
/// Each pass's window is wide enough that no appointment falls between two
/// passes, and NotificationService's (AppointmentId, RecipientUserId, Kind)
/// unique index means a reminder already sent is silently skipped rather
/// than duplicated if two passes' windows overlap.
/// </summary>
public sealed class ReminderSweeper(IServiceScopeFactory scopeFactory, ILogger<ReminderSweeper> logger) : BackgroundService
{
    private static readonly TimeSpan Interval = TimeSpan.FromMinutes(5);
    private static readonly TimeSpan WindowHalfWidth = TimeSpan.FromMinutes(10);

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
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
        var notifications = scope.ServiceProvider.GetRequiredService<NotificationService>();
        var now = DateTime.UtcNow;

        try
        {
            await notifications.SendRemindersAsync(
                NotificationKind.Reminder24h, now.AddHours(24) - WindowHalfWidth, now.AddHours(24) + WindowHalfWidth, cancellationToken);
            await notifications.SendRemindersAsync(
                NotificationKind.Reminder1h, now.AddHours(1) - WindowHalfWidth, now.AddHours(1) + WindowHalfWidth, cancellationToken);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            logger.LogWarning(ex, "Reminder sweep failed");
        }
    }
}
