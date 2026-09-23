namespace MelanomaDetection.Web.Data;

/// <summary>
/// Scheduling config for a provider account. Shares its primary key with
/// <see cref="AppUser.Id"/> (one-to-one) rather than having its own -- a
/// provider is an AppUser first; this just adds the columns only providers need.
/// </summary>
public class Provider
{
    public Guid Id { get; set; }

    public string Specialty { get; set; } = "Dermatology";

    public int AppointmentLengthMinutes { get; set; } = 30;

    /// <summary>Gap kept free after each appointment before the next can start.</summary>
    public int BufferMinutes { get; set; } = 10;

    /// <summary>IANA id (e.g. "America/Chicago"). AvailabilityRule times are local to this zone.</summary>
    public string TimeZoneId { get; set; } = "America/Chicago";

    /// <summary>State medical license number, self-reported. Not verified against any
    /// licensing board -- see the profile form's own disclaimer for why that's out of
    /// scope here.</summary>
    public string? LicenseNumber { get; set; }

    /// <summary>Free text, e.g. "MD, Board-Certified Dermatologist".</summary>
    public string? Credentials { get; set; }

    public string? Bio { get; set; }

    /// <summary>URL of a profile photo. No upload/storage pipeline for this demo --
    /// a provider pastes a link (their Google avatar is offered as a default).</summary>
    public string? PhotoUrl { get; set; }
}

/// <summary>One recurring weekly open window, e.g. "Mon 09:00-12:00". Local to
/// <see cref="Provider.TimeZoneId"/>, not UTC -- a provider sets these once and
/// they should keep meaning "9am my time" across DST changes.</summary>
public class AvailabilityRule
{
    public int Id { get; set; }

    public Guid ProviderId { get; set; }

    public DayOfWeek Weekday { get; set; }

    public TimeOnly StartTime { get; set; }

    public TimeOnly EndTime { get; set; }
}

/// <summary>A one-off change to a specific date: either an extra open window outside
/// the normal weekly rules (IsBlocked false, times required), or the whole day taken
/// off (IsBlocked true, times null). Local to <see cref="Provider.TimeZoneId"/>.</summary>
public class AvailabilityException
{
    public int Id { get; set; }

    public Guid ProviderId { get; set; }

    public DateOnly Date { get; set; }

    public TimeOnly? StartTime { get; set; }

    public TimeOnly? EndTime { get; set; }

    public bool IsBlocked { get; set; }
}
