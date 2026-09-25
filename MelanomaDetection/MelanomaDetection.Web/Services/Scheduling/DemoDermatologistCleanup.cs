using MelanomaDetection.Web.Data;
using Microsoft.EntityFrameworkCore;

namespace MelanomaDetection.Web.Services.Scheduling;

/// <summary>
/// Removes the fake dermatologists an earlier version seeded on every start (Google subject
/// "seed-derm:…", no email), so only real provider accounts -- people whose email is on
/// Provider:AllowedEmails and who have signed in -- can be browsed and booked. Everything
/// hanging off them goes too: their hours, blocked dates, visits booked with them and the
/// notifications about those visits. Idempotent; a no-op once they are gone.
/// </summary>
public static class DemoDermatologistCleanup
{
    public const string SeedSubjectPrefix = "seed-derm:";

    public static async Task RemoveDemoDermatologistsAsync(this WebApplication app)
    {
        var removed = await RemoveAsync(app.Services.GetRequiredService<IDbContextFactory<AppDbContext>>());
        if (removed > 0)
        {
            app.Logger.LogInformation("Removed {Count} seeded demo dermatologists", removed);
        }
    }

    /// <summary>Deletes the seeded doctors and their data; returns how many doctors were removed.</summary>
    public static async Task<int> RemoveAsync(IDbContextFactory<AppDbContext> factory, CancellationToken cancellationToken = default)
    {
        await using var db = await factory.CreateDbContextAsync(cancellationToken);
        var ids = await db.Users
            .Where(u => u.GoogleSubject.StartsWith(SeedSubjectPrefix))
            .Select(u => u.Id)
            .ToListAsync(cancellationToken);
        if (ids.Count == 0)
        {
            return 0;
        }

        await using var transaction = await db.Database.BeginTransactionAsync(cancellationToken);
        var appointmentIds = await db.Appointments
            .Where(a => ids.Contains(a.ProviderId))
            .Select(a => a.Id)
            .ToListAsync(cancellationToken);
        await db.Notifications.Where(n => appointmentIds.Contains(n.AppointmentId)).ExecuteDeleteAsync(cancellationToken);
        await db.Appointments.Where(a => ids.Contains(a.ProviderId)).ExecuteDeleteAsync(cancellationToken);
        await db.AvailabilityRules.Where(r => ids.Contains(r.ProviderId)).ExecuteDeleteAsync(cancellationToken);
        await db.AvailabilityExceptions.Where(e => ids.Contains(e.ProviderId)).ExecuteDeleteAsync(cancellationToken);
        await db.Providers.Where(p => ids.Contains(p.Id)).ExecuteDeleteAsync(cancellationToken);
        await db.Users.Where(u => ids.Contains(u.Id)).ExecuteDeleteAsync(cancellationToken);
        await transaction.CommitAsync(cancellationToken);
        return ids.Count;
    }
}
