namespace MelanomaDetection.Web.Data;

public enum NotificationKind
{
    BookingConfirmed = 0,
    Cancelled = 1,
    Reminder24h = 2,
    Reminder1h = 3,
}

/// <summary>
/// A "sent" notification -- this demo never emails or texts anyone; booking,
/// cancellation and reminder notices are recorded here and shown in-app
/// instead (see NotificationService, Pages/Notifications.razor). A
/// (AppointmentId, RecipientUserId, Kind) unique index guarantees each kind
/// only ever reaches a given recipient once for a given appointment, the
/// same DB-level guard philosophy as the double-booking index.
/// </summary>
public class Notification
{
    public Guid Id { get; set; }

    public Guid AppointmentId { get; set; }

    public Guid RecipientUserId { get; set; }

    public NotificationKind Kind { get; set; }

    public string Subject { get; set; } = "";

    public string Body { get; set; } = "";

    /// <summary>The .ics calendar attachment text, for kinds that carry one (booking/cancellation).</summary>
    public string? IcsContent { get; set; }

    public DateTime CreatedAtUtc { get; set; }

    public bool IsRead { get; set; }
}
