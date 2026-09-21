using MelanomaDetection.Web.Models;

namespace MelanomaDetection.Web.Services;

/// <summary>User-facing copy for a spot's recheck status, derived from the backend's nextDueAt.</summary>
public static class Recheck
{
    /// <summary>Tone is "overdue", "soon", "later" or "none", for styling the badge.</summary>
    public record Status(string? Text, string Tone);

    public static Status StatusOf(Spot spot, DateTimeOffset? now = null)
    {
        if (spot.NextDueAt is null)
        {
            return new Status(null, "none");
        }

        var today = (now ?? DateTimeOffset.UtcNow).Date;
        var days = (spot.NextDueAt.Value.Date - today).Days;

        if (spot.CadenceDays == 0)
        {
            return new Status("See a dermatologist", "overdue");
        }

        return days switch
        {
            < -1 => new Status($"Overdue by {-days} days", "overdue"),
            -1 => new Status("Overdue by 1 day", "overdue"),
            0 => new Status("Due today", "overdue"),
            1 => new Status("Due tomorrow", "soon"),
            <= 14 => new Status($"Due in {days} days", "soon"),
            _ => new Status($"Due {spot.NextDueAt.Value.ToLocalTime():MMM d}", "later"),
        };
    }

    public static string CadenceSentence(Spot spot) => spot.CadenceDays switch
    {
        null => "Take a first photo to start tracking this spot.",
        0 => "This spot's last check points to seeing a dermatologist now, rather than waiting for a scheduled recheck.",
        var days => $"Rechecked every {days} days at this cadence.",
    };

    public static string Ago(DateTimeOffset? at, DateTimeOffset? now = null)
    {
        if (at is null)
        {
            return "never";
        }

        var elapsed = (now ?? DateTimeOffset.UtcNow) - at.Value;
        return elapsed.TotalDays switch
        {
            < 1 => "today",
            < 2 => "yesterday",
            < 30 => $"{(int)elapsed.TotalDays} days ago",
            < 365 => $"{(int)(elapsed.TotalDays / 30)} months ago",
            _ => $"{(int)(elapsed.TotalDays / 365)} years ago",
        };
    }
}
