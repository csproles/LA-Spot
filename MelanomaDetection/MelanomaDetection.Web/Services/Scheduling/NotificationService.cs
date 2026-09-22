using MelanomaDetection.Web.Data;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;

namespace MelanomaDetection.Web.Services.Scheduling;

/// <summary>
/// Records booking/cancellation/reminder notices for a visit's two parties.
/// This demo never actually emails or texts anyone -- see the phase-7 design
/// choice -- notices are written to the Notifications table and shown
/// in-app (Pages/Notifications.razor) instead, each carrying the same .ics
/// text a real email's calendar attachment would have.
/// </summary>
public sealed class NotificationService(IDbContextFactory<AppDbContext> dbFactory)
{
    public async Task NotifyBookedAsync(Guid appointmentId, CancellationToken cancellationToken = default)
    {
        var appt = await LoadSnapshotAsync(appointmentId, cancellationToken);
        if (appt is null)
        {
            return;
        }

        var ics = IcsBuilder.BuildVisitEvent(
            appt.Id, appt.StartUtc, appt.EndUtc, "LA Spot video visit", VisitDescription(appt), IcsStatus.Confirmed, DateTime.UtcNow);

        await CreateAsync(appt.PatientId, appointmentId, NotificationKind.BookingConfirmed,
            "Your video visit is booked",
            $"Your visit with {appt.ProviderName} is booked for {FormatWhen(appt.StartUtc)}.",
            ics, cancellationToken);

        await CreateAsync(appt.ProviderId, appointmentId, NotificationKind.BookingConfirmed,
            "New visit booked",
            $"{appt.PatientName} booked a visit with you for {FormatWhen(appt.StartUtc)}.",
            ics, cancellationToken);
    }

    public async Task NotifyCancelledAsync(Guid appointmentId, string cancelledBy, CancellationToken cancellationToken = default)
    {
        var appt = await LoadSnapshotAsync(appointmentId, cancellationToken);
        if (appt is null)
        {
            return;
        }

        var ics = IcsBuilder.BuildVisitEvent(
            appt.Id, appt.StartUtc, appt.EndUtc, "LA Spot video visit (cancelled)", VisitDescription(appt), IcsStatus.Cancelled, DateTime.UtcNow);

        await CreateAsync(appt.PatientId, appointmentId, NotificationKind.Cancelled,
            "Your video visit was cancelled",
            $"Your visit with {appt.ProviderName} for {FormatWhen(appt.StartUtc)} was cancelled by the {cancelledBy}.",
            ics, cancellationToken);

        await CreateAsync(appt.ProviderId, appointmentId, NotificationKind.Cancelled,
            "A visit was cancelled",
            $"{appt.PatientName}'s visit for {FormatWhen(appt.StartUtc)} was cancelled by the {cancelledBy}.",
            ics, cancellationToken);
    }

    /// <summary>Sends the given reminder kind to both parties of every booked appointment starting within the window.</summary>
    public async Task SendRemindersAsync(
        NotificationKind kind, DateTime windowStartUtc, DateTime windowEndUtc, CancellationToken cancellationToken = default)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        var dueAppointmentIds = await db.Appointments.AsNoTracking()
            .Where(a => a.Status == AppointmentStatus.Booked && a.StartUtc >= windowStartUtc && a.StartUtc <= windowEndUtc)
            .Select(a => a.Id)
            .ToListAsync(cancellationToken);

        var whenLabel = kind == NotificationKind.Reminder24h ? "tomorrow" : "in about an hour";

        foreach (var appointmentId in dueAppointmentIds)
        {
            var appt = await LoadSnapshotAsync(appointmentId, cancellationToken);
            if (appt is null)
            {
                continue;
            }

            await CreateAsync(appt.PatientId, appointmentId, kind,
                "Upcoming video visit reminder",
                $"Reminder: your visit with {appt.ProviderName} is {whenLabel}, {FormatWhen(appt.StartUtc)}.",
                null, cancellationToken);

            await CreateAsync(appt.ProviderId, appointmentId, kind,
                "Upcoming video visit reminder",
                $"Reminder: your visit with {appt.PatientName} is {whenLabel}, {FormatWhen(appt.StartUtc)}.",
                null, cancellationToken);
        }
    }

    public async Task<IReadOnlyList<Notification>> ListForUserAsync(Guid userId, CancellationToken cancellationToken = default)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        return await db.Notifications.AsNoTracking()
            .Where(n => n.RecipientUserId == userId)
            .OrderByDescending(n => n.CreatedAtUtc)
            .ToListAsync(cancellationToken);
    }

    /// <summary>Marks every one of a user's notifications read -- called when they open the notifications page.</summary>
    public async Task MarkAllReadAsync(Guid userId, CancellationToken cancellationToken = default)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        await db.Notifications
            .Where(n => n.RecipientUserId == userId && !n.IsRead)
            .ExecuteUpdateAsync(setters => setters.SetProperty(n => n.IsRead, true), cancellationToken);
    }

    /// <summary>The raw .ics text for one notification, or null if it has none or doesn't belong to <paramref name="userId"/>.</summary>
    public async Task<string?> FindIcsForUserAsync(Guid notificationId, Guid userId, CancellationToken cancellationToken = default)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        return await db.Notifications.AsNoTracking()
            .Where(n => n.Id == notificationId && n.RecipientUserId == userId)
            .Select(n => n.IcsContent)
            .SingleOrDefaultAsync(cancellationToken);
    }

    private async Task CreateAsync(
        Guid recipientUserId, Guid appointmentId, NotificationKind kind, string subject, string body, string? ics,
        CancellationToken cancellationToken)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        db.Notifications.Add(new Notification
        {
            Id = Guid.NewGuid(),
            AppointmentId = appointmentId,
            RecipientUserId = recipientUserId,
            Kind = kind,
            Subject = subject,
            Body = body,
            IcsContent = ics,
            CreatedAtUtc = DateTime.UtcNow,
        });

        try
        {
            await db.SaveChangesAsync(cancellationToken);
        }
        catch (DbUpdateException ex) when (IsUniqueConstraintViolation(ex))
        {
            // Already sent -- a duplicate call, or a reminder sweep pass overlapping the previous one.
        }
    }

    private async Task<AppointmentSnapshot?> LoadSnapshotAsync(Guid appointmentId, CancellationToken cancellationToken)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);

        // Projecting straight into a record from inside the join doesn't translate to SQL
        // (see BookingService.QueryAppointmentsAsync's note) -- anonymous type, then map after.
        var row = await db.Appointments.AsNoTracking()
            .Where(a => a.Id == appointmentId)
            .Join(db.Users.AsNoTracking(), a => a.ProviderId, u => u.Id, (a, providerUser) => new { a, providerUser })
            .Join(db.Users.AsNoTracking(), x => x.a.PatientId, u => u.Id, (x, patientUser) => new
            {
                x.a.Id,
                x.a.ProviderId,
                ProviderName = x.providerUser.DisplayName,
                x.a.PatientId,
                PatientName = patientUser.DisplayName,
                x.a.StartUtc,
                x.a.EndUtc,
                x.a.Reason,
            })
            .SingleOrDefaultAsync(cancellationToken);

        return row is null
            ? null
            : new AppointmentSnapshot(row.Id, row.ProviderId, row.ProviderName, row.PatientId, row.PatientName, row.StartUtc, row.EndUtc, row.Reason);
    }

    private static string VisitDescription(AppointmentSnapshot appt) =>
        string.IsNullOrWhiteSpace(appt.Reason) ? $"Video visit between {appt.PatientName} and {appt.ProviderName}." : appt.Reason;

    private static string FormatWhen(DateTime startUtc) => $"{startUtc:dddd, MMM d} at {startUtc:h:mm tt} UTC";

    private static bool IsUniqueConstraintViolation(DbUpdateException ex) =>
        ex.InnerException is SqliteException { SqliteErrorCode: 19 }; // SQLITE_CONSTRAINT

    private sealed record AppointmentSnapshot(
        Guid Id, Guid ProviderId, string ProviderName, Guid PatientId, string PatientName, DateTime StartUtc, DateTime EndUtc, string? Reason);
}
