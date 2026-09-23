namespace MelanomaDetection.Web.Data;

public enum AppointmentStatus
{
    Booked = 0,
    Cancelled = 1,
    Completed = 2,
}

/// <summary>One booked video visit. Double-booking is prevented at the DB level by a
/// unique index on (ProviderId, StartUtc) filtered to non-cancelled rows -- see
/// AppDbContext -- not just by checking availability before the insert.</summary>
public class Appointment
{
    public Guid Id { get; set; }

    public Guid ProviderId { get; set; }

    public Guid PatientId { get; set; }

    public DateTime StartUtc { get; set; }

    public DateTime EndUtc { get; set; }

    public AppointmentStatus Status { get; set; } = AppointmentStatus.Booked;

    /// <summary>PHI -- never log this. See ILogger calls elsewhere in this project for the pattern.</summary>
    public string? Reason { get; set; }

    /// <summary>Opaque id for the video room (fake for this demo -- see VideoRoom).</summary>
    public string? MeetingId { get; set; }

    public string? MeetingUrl { get; set; }

    /// <summary>Flask-side processing id of the check the patient chose to attach, if any.
    /// Never used to re-fetch from Flask on the provider's behalf (there's no cross-user
    /// auth path for that) -- ScanRiskScore/ScanOverallVisualConcern/ScanExplanation below
    /// are a snapshot taken in the patient's own browser session at booking time.</summary>
    public string? ScanProcessingId { get; set; }

    public double? ScanRiskScore { get; set; }

    public string? ScanOverallVisualConcern { get; set; }

    /// <summary>PHI -- never log this. Snapshot of the plain-language explanation at booking time.</summary>
    public string? ScanExplanation { get; set; }

    public DateTime CreatedAtUtc { get; set; }

    public DateTime? CancelledAtUtc { get; set; }

    /// <summary>"patient" or "provider" -- who cancelled, for the notice shown to the other side.</summary>
    public string? CancelledBy { get; set; }
}
