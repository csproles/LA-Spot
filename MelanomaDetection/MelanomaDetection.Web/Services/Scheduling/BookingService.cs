using System.Linq.Expressions;
using MelanomaDetection.Web.Data;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;

namespace MelanomaDetection.Web.Services.Scheduling;

public enum BookingOutcome { Booked, SlotNotAvailable, Conflict }

public sealed record BookingResult(BookingOutcome Outcome, Appointment? Appointment = null)
{
    public static BookingResult Booked(Appointment appointment) => new(BookingOutcome.Booked, appointment);

    public static readonly BookingResult SlotNotAvailable = new(BookingOutcome.SlotNotAvailable);

    public static readonly BookingResult Conflict = new(BookingOutcome.Conflict);
}

/// <summary>An appointment plus the two display names neither side has to look up separately.</summary>
public sealed record AppointmentView(
    Guid Id, Guid ProviderId, string ProviderName, Guid PatientId, string PatientName,
    DateTime StartUtc, DateTime EndUtc, AppointmentStatus Status, string? Reason);

/// <summary>
/// Books and cancels appointments. The real double-booking guard is the
/// filtered unique index on (ProviderId, StartUtc) in <see cref="AppDbContext"/>
/// -- this class re-checks availability first only to give a friendly result
/// for the common case, not as the source of truth for concurrent requests.
/// </summary>
public sealed class BookingService(
    IDbContextFactory<AppDbContext> dbFactory, AvailabilityService availability, NotificationService notifications, ILogger<BookingService> logger)
{
    public async Task<BookingResult> BookAsync(
        Guid providerId, Guid patientId, DateTime startUtc, string? reason, CancellationToken cancellationToken = default)
    {
        var provider = await GetProviderAsync(providerId, cancellationToken)
            ?? throw new KeyNotFoundException($"No provider {providerId}.");

        // +-1 day covers every timezone offset a provider's local calendar date could fall on.
        var from = DateOnly.FromDateTime(startUtc.AddDays(-1));
        var to = DateOnly.FromDateTime(startUtc.AddDays(1));
        var openSlots = await availability.GetOpenSlotsUtcAsync(providerId, from, to, cancellationToken);
        if (!openSlots.Contains(startUtc))
        {
            return BookingResult.SlotNotAvailable;
        }

        var appointment = new Appointment
        {
            Id = Guid.NewGuid(),
            ProviderId = providerId,
            PatientId = patientId,
            StartUtc = startUtc,
            EndUtc = startUtc.AddMinutes(provider.AppointmentLengthMinutes),
            Reason = reason,
            CreatedAtUtc = DateTime.UtcNow,
        };

        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        db.Appointments.Add(appointment);
        try
        {
            await db.SaveChangesAsync(cancellationToken);
        }
        catch (DbUpdateException ex) when (IsUniqueConstraintViolation(ex))
        {
            // Another request won the race for this exact slot between our
            // availability check above and this insert.
            return BookingResult.Conflict;
        }

        try
        {
            await notifications.NotifyBookedAsync(appointment.Id, cancellationToken);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Best-effort: a notification failure shouldn't undo a real booking.
            logger.LogWarning(ex, "Could not create booking notification for appointment {AppointmentId}", appointment.Id);
        }

        return BookingResult.Booked(appointment);
    }

    /// <summary>Cancel an appointment. Only the patient who booked it or the owning provider may.</summary>
    public async Task<bool> CancelAsync(Guid appointmentId, Guid actingUserId, CancellationToken cancellationToken = default)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        var appointment = await db.Appointments.SingleOrDefaultAsync(a => a.Id == appointmentId, cancellationToken);
        if (appointment is null)
        {
            return false;
        }

        var isPatient = appointment.PatientId == actingUserId;
        var isProvider = appointment.ProviderId == actingUserId;
        if (!isPatient && !isProvider)
        {
            throw new UnauthorizedAccessException("Only the patient or the provider on an appointment may cancel it.");
        }

        if (appointment.Status != AppointmentStatus.Cancelled)
        {
            appointment.Status = AppointmentStatus.Cancelled;
            appointment.CancelledAtUtc = DateTime.UtcNow;
            appointment.CancelledBy = isPatient ? "patient" : "provider";
            await db.SaveChangesAsync(cancellationToken);

            try
            {
                await notifications.NotifyCancelledAsync(appointmentId, appointment.CancelledBy, cancellationToken);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                logger.LogWarning(ex, "Could not create cancellation notification for appointment {AppointmentId}", appointmentId);
            }
        }

        return true;
    }

    /// <summary>One appointment, for the video room -- null if it doesn't exist, or throws if the caller is neither party to it.</summary>
    public async Task<AppointmentView?> FindForUserAsync(Guid appointmentId, Guid actingUserId, CancellationToken cancellationToken = default)
    {
        var view = (await QueryAppointmentsAsync(a => a.Id == appointmentId, cancellationToken)).SingleOrDefault();
        if (view is null)
        {
            return null;
        }

        if (view.ProviderId != actingUserId && view.PatientId != actingUserId)
        {
            throw new UnauthorizedAccessException("That visit belongs to someone else.");
        }

        return view;
    }

    /// <summary>Mark a visit completed. Only the owning provider may -- a patient can't close out their own visit.</summary>
    public async Task<bool> CompleteAsync(Guid appointmentId, Guid providerId, CancellationToken cancellationToken = default)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        var appointment = await db.Appointments.SingleOrDefaultAsync(a => a.Id == appointmentId, cancellationToken);
        if (appointment is null)
        {
            return false;
        }

        if (appointment.ProviderId != providerId)
        {
            throw new UnauthorizedAccessException("Only the provider on a visit may mark it complete.");
        }

        if (appointment.Status == AppointmentStatus.Booked)
        {
            appointment.Status = AppointmentStatus.Completed;
            await db.SaveChangesAsync(cancellationToken);
        }

        return true;
    }

    public Task<IReadOnlyList<AppointmentView>> ListForPatientAsync(Guid patientId, CancellationToken cancellationToken = default) =>
        QueryAppointmentsAsync(a => a.PatientId == patientId, cancellationToken);

    public Task<IReadOnlyList<AppointmentView>> ListForProviderAsync(Guid providerId, CancellationToken cancellationToken = default) =>
        QueryAppointmentsAsync(a => a.ProviderId == providerId, cancellationToken);

    private async Task<IReadOnlyList<AppointmentView>> QueryAppointmentsAsync(
        Expression<Func<Appointment, bool>> predicate, CancellationToken cancellationToken)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);

        // Projecting straight into the AppointmentView record from inside the join
        // doesn't translate to SQL (EF throws at query time) -- project to an
        // anonymous type instead and build the record after materializing.
        var rows = await db.Appointments.AsNoTracking()
            .Where(predicate)
            .OrderBy(a => a.StartUtc)
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
                x.a.Status,
                x.a.Reason,
            })
            .ToListAsync(cancellationToken);

        return [.. rows.Select(r => new AppointmentView(
            r.Id, r.ProviderId, r.ProviderName, r.PatientId, r.PatientName, r.StartUtc, r.EndUtc, r.Status, r.Reason))];
    }

    private async Task<Provider?> GetProviderAsync(Guid providerId, CancellationToken cancellationToken)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        return await db.Providers.AsNoTracking().SingleOrDefaultAsync(p => p.Id == providerId, cancellationToken);
    }

    private static bool IsUniqueConstraintViolation(DbUpdateException ex) =>
        ex.InnerException is SqliteException { SqliteErrorCode: 19 }; // SQLITE_CONSTRAINT
}
