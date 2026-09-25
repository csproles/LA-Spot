using MelanomaDetection.Web.Data;
using Microsoft.EntityFrameworkCore;

namespace MelanomaDetection.Web.Services.Scheduling;

public sealed record ProviderSummary(
    Guid Id, string DisplayName, string Specialty, string? Credentials, string? Bio, string? PhotoUrl, string? HubCity = null);

/// <summary>
/// DB-backed wrapper around <see cref="SlotGenerator"/>: loads one provider's
/// rules, exceptions and booked appointments, then hands them to the pure
/// slot-generation logic.
/// </summary>
public sealed class AvailabilityService(IDbContextFactory<AppDbContext> dbFactory)
{
    public async Task<IReadOnlyList<ProviderSummary>> ListProvidersAsync(CancellationToken cancellationToken = default)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        var providers = await db.Providers.AsNoTracking()
            .Join(db.Users.AsNoTracking(), p => p.Id, u => u.Id, (p, u) =>
                new ProviderSummary(p.Id, u.DisplayName, p.Specialty, p.Credentials, p.Bio, p.PhotoUrl, p.HubCity))
            .ToListAsync(cancellationToken);
        return [.. providers.OrderBy(p => p.DisplayName, StringComparer.OrdinalIgnoreCase)];
    }

    /// <summary>Open, bookable slot start times (UTC) for a provider between two dates, inclusive.</summary>
    public async Task<IReadOnlyList<DateTime>> GetOpenSlotsUtcAsync(
        Guid providerId, DateOnly from, DateOnly to, CancellationToken cancellationToken = default)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        var provider = await db.Providers.AsNoTracking().SingleOrDefaultAsync(p => p.Id == providerId, cancellationToken)
            ?? throw new KeyNotFoundException($"No provider {providerId}.");

        var rules = await db.AvailabilityRules.AsNoTracking()
            .Where(r => r.ProviderId == providerId)
            .ToListAsync(cancellationToken);
        var exceptions = await db.AvailabilityExceptions.AsNoTracking()
            .Where(e => e.ProviderId == providerId && e.Date >= from && e.Date <= to)
            .ToListAsync(cancellationToken);

        // Widen by a day either side: a local evening slot near UTC's date boundary
        // can land on the UTC day before or after the requested range.
        var rangeStartUtc = from.AddDays(-1).ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc);
        var rangeEndUtc = to.AddDays(1).ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc);
        var booked = await db.Appointments.AsNoTracking()
            .Where(a => a.ProviderId == providerId
                && a.Status != AppointmentStatus.Cancelled
                && a.StartUtc < rangeEndUtc && a.EndUtc > rangeStartUtc)
            .Select(a => new { a.StartUtc, a.EndUtc })
            .ToListAsync(cancellationToken);

        return SlotGenerator.GenerateOpenSlotsUtc(
            rules,
            exceptions,
            booked.Select(a => (a.StartUtc, a.EndUtc)).ToList(),
            provider.TimeZoneId,
            provider.AppointmentLengthMinutes,
            provider.BufferMinutes,
            from,
            to,
            DateTime.UtcNow);
    }

    public async Task<Provider?> FindProviderAsync(Guid providerId, CancellationToken cancellationToken = default)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        return await db.Providers.AsNoTracking().SingleOrDefaultAsync(p => p.Id == providerId, cancellationToken);
    }

    /// <summary>Update a provider's own bookable-slot settings. Throws <see cref="TimeZoneNotFoundException"/> for an unrecognized zone id.</summary>
    public async Task UpdateProviderSettingsAsync(
        Guid providerId, int appointmentLengthMinutes, int bufferMinutes, string timeZoneId, CancellationToken cancellationToken = default)
    {
        if (appointmentLengthMinutes < 5 || appointmentLengthMinutes > 240)
        {
            throw new ArgumentOutOfRangeException(nameof(appointmentLengthMinutes), "Visit length must be between 5 and 240 minutes.");
        }

        if (bufferMinutes < 0 || bufferMinutes > 120)
        {
            throw new ArgumentOutOfRangeException(nameof(bufferMinutes), "Buffer must be between 0 and 120 minutes.");
        }

        TimeZoneInfo.FindSystemTimeZoneById(timeZoneId); // throws TimeZoneNotFoundException if bad

        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        var provider = await db.Providers.SingleOrDefaultAsync(p => p.Id == providerId, cancellationToken)
            ?? throw new KeyNotFoundException($"No provider {providerId}.");
        provider.AppointmentLengthMinutes = appointmentLengthMinutes;
        provider.BufferMinutes = bufferMinutes;
        provider.TimeZoneId = timeZoneId;
        await db.SaveChangesAsync(cancellationToken);
    }

    /// <summary>Update the self-reported credentialing fields patients see when browsing
    /// providers. Nothing here is verified against a licensing board -- see Provider.LicenseNumber.</summary>
    public async Task UpdateProviderProfileAsync(
        Guid providerId, string specialty, string? licenseNumber, string? credentials, string? bio, string? photoUrl,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(specialty))
        {
            throw new ArgumentException("Specialty can't be blank.");
        }

        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        var provider = await db.Providers.SingleOrDefaultAsync(p => p.Id == providerId, cancellationToken)
            ?? throw new KeyNotFoundException($"No provider {providerId}.");
        provider.Specialty = specialty.Trim();
        provider.LicenseNumber = string.IsNullOrWhiteSpace(licenseNumber) ? null : licenseNumber.Trim();
        provider.Credentials = string.IsNullOrWhiteSpace(credentials) ? null : credentials.Trim();
        provider.Bio = string.IsNullOrWhiteSpace(bio) ? null : bio.Trim();
        provider.PhotoUrl = string.IsNullOrWhiteSpace(photoUrl) ? null : photoUrl.Trim();
        await db.SaveChangesAsync(cancellationToken);
    }

    public async Task<IReadOnlyList<AvailabilityRule>> ListRulesAsync(Guid providerId, CancellationToken cancellationToken = default)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        return await db.AvailabilityRules.AsNoTracking()
            .Where(r => r.ProviderId == providerId)
            .OrderBy(r => r.Weekday).ThenBy(r => r.StartTime)
            .ToListAsync(cancellationToken);
    }

    public async Task AddRuleAsync(Guid providerId, DayOfWeek weekday, TimeOnly start, TimeOnly end, CancellationToken cancellationToken = default)
    {
        if (start >= end)
        {
            throw new ArgumentException("Start time must be before end time.");
        }

        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        db.AvailabilityRules.Add(new AvailabilityRule { ProviderId = providerId, Weekday = weekday, StartTime = start, EndTime = end });
        await db.SaveChangesAsync(cancellationToken);
    }

    /// <summary>Delete a rule. Throws if it doesn't belong to <paramref name="providerId"/>.</summary>
    public async Task RemoveRuleAsync(Guid providerId, int ruleId, CancellationToken cancellationToken = default)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        var rule = await db.AvailabilityRules.SingleOrDefaultAsync(r => r.Id == ruleId, cancellationToken);
        if (rule is null)
        {
            return;
        }

        if (rule.ProviderId != providerId)
        {
            throw new UnauthorizedAccessException("That availability rule belongs to a different provider.");
        }

        db.AvailabilityRules.Remove(rule);
        await db.SaveChangesAsync(cancellationToken);
    }

    public async Task<IReadOnlyList<AvailabilityException>> ListExceptionsAsync(
        Guid providerId, DateOnly from, CancellationToken cancellationToken = default)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        return await db.AvailabilityExceptions.AsNoTracking()
            .Where(e => e.ProviderId == providerId && e.Date >= from)
            .OrderBy(e => e.Date)
            .ToListAsync(cancellationToken);
    }

    public async Task AddExceptionAsync(
        Guid providerId, DateOnly date, bool isBlocked, TimeOnly? start, TimeOnly? end, CancellationToken cancellationToken = default)
    {
        // A whole-day block passes both as null; every other kind (a partial block or
        // extra hours) needs both. One set and the other missing is never valid.
        if (start is null != end is null)
        {
            throw new ArgumentException("Start and end time must both be set, or both left blank for a whole-day block.");
        }

        if (start is not null && end is not null && start >= end)
        {
            throw new ArgumentException("Start time must be before end time.");
        }

        if (!isBlocked && start is null)
        {
            throw new ArgumentException("Extra availability needs a start and end time.");
        }

        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        db.AvailabilityExceptions.Add(new AvailabilityException
        {
            ProviderId = providerId,
            Date = date,
            IsBlocked = isBlocked,
            StartTime = start,
            EndTime = end,
        });
        await db.SaveChangesAsync(cancellationToken);
    }

    /// <summary>Delete an exception. Throws if it doesn't belong to <paramref name="providerId"/>.</summary>
    public async Task RemoveExceptionAsync(Guid providerId, int exceptionId, CancellationToken cancellationToken = default)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        var exception = await db.AvailabilityExceptions.SingleOrDefaultAsync(e => e.Id == exceptionId, cancellationToken);
        if (exception is null)
        {
            return;
        }

        if (exception.ProviderId != providerId)
        {
            throw new UnauthorizedAccessException("That exception belongs to a different provider.");
        }

        db.AvailabilityExceptions.Remove(exception);
        await db.SaveChangesAsync(cancellationToken);
    }
}
