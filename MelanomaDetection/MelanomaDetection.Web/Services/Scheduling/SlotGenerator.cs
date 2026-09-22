using MelanomaDetection.Web.Data;

namespace MelanomaDetection.Web.Services.Scheduling;

/// <summary>
/// Turns a provider's weekly availability + one-off exceptions + already-booked
/// appointments into a list of bookable slot start times, in UTC. Pure and
/// DB-free on purpose -- <see cref="AvailabilityService"/> is the thin DB-backed
/// wrapper, this is the part worth unit-testing directly (see
/// MelanomaDetection.Web.Tests/Scheduling/SlotGeneratorTests.cs).
/// </summary>
public static class SlotGenerator
{
    public static IReadOnlyList<DateTime> GenerateOpenSlotsUtc(
        IReadOnlyList<AvailabilityRule> rules,
        IReadOnlyList<AvailabilityException> exceptions,
        IReadOnlyList<(DateTime StartUtc, DateTime EndUtc)> bookedAppointments,
        string timeZoneId,
        int appointmentLengthMinutes,
        int bufferMinutes,
        DateOnly fromDate,
        DateOnly toDate,
        DateTime nowUtc)
    {
        var tz = TimeZoneInfo.FindSystemTimeZoneById(timeZoneId);
        var slots = new List<DateTime>();

        for (var date = fromDate; date <= toDate; date = date.AddDays(1))
        {
            foreach (var window in OpenWindowsForDate(date, rules, exceptions))
            {
                slots.AddRange(SlotsInWindow(date, window, tz, appointmentLengthMinutes, bufferMinutes));
            }
        }

        return slots
            .Where(s => s >= nowUtc)
            .Where(s => !Overlaps(s, s.AddMinutes(appointmentLengthMinutes), bookedAppointments))
            .OrderBy(s => s)
            .ToList();
    }

    private static IEnumerable<DateTime> SlotsInWindow(
        DateOnly date, (int Start, int End) window, TimeZoneInfo tz, int lengthMinutes, int bufferMinutes)
    {
        var step = lengthMinutes + bufferMinutes;
        for (var minute = window.Start; minute + lengthMinutes <= window.End; minute += step)
        {
            var local = new DateTime(date.Year, date.Month, date.Day, 0, 0, 0, DateTimeKind.Unspecified)
                .AddMinutes(minute);

            // Spring-forward gap (e.g. 2:30am on the day clocks jump 2am -> 3am):
            // this local time never happened, so there's no slot to offer.
            if (tz.IsInvalidTime(local))
            {
                continue;
            }

            // Ambiguous fall-back times (a local time that occurs twice) resolve to
            // the standard-time offset -- ConvertTimeToUtc's documented default.
            yield return TimeZoneInfo.ConvertTimeToUtc(local, tz);
        }
    }

    /// <summary>Open local-time windows (minutes since midnight) for one calendar date.</summary>
    private static List<(int Start, int End)> OpenWindowsForDate(
        DateOnly date, IReadOnlyList<AvailabilityRule> rules, IReadOnlyList<AvailabilityException> exceptions)
    {
        var dayExceptions = exceptions.Where(e => e.Date == date).ToList();
        if (dayExceptions.Any(e => e.IsBlocked && e.StartTime is null))
        {
            return []; // whole day blocked
        }

        var windows = rules
            .Where(r => r.Weekday == date.DayOfWeek)
            .Select(r => (Start: ToMinutes(r.StartTime), End: ToMinutes(r.EndTime)))
            .ToList();

        foreach (var ex in dayExceptions.Where(e => e.StartTime is not null && e.EndTime is not null))
        {
            var range = (Start: ToMinutes(ex.StartTime!.Value), End: ToMinutes(ex.EndTime!.Value));
            windows = ex.IsBlocked ? Subtract(windows, range) : [.. windows, range];
        }

        return Merge(windows);
    }

    private static int ToMinutes(TimeOnly t) => t.Hour * 60 + t.Minute;

    private static List<(int Start, int End)> Merge(List<(int Start, int End)> windows)
    {
        // ponytail: a window with Start >= End (e.g. a rule spanning midnight) is
        // dropped rather than wrapped to the next day. Split overnight availability
        // into two same-day rules if that's ever needed.
        var sorted = windows.Where(w => w.Start < w.End).OrderBy(w => w.Start).ToList();
        var merged = new List<(int Start, int End)>();
        foreach (var w in sorted)
        {
            if (merged.Count > 0 && w.Start <= merged[^1].End)
            {
                merged[^1] = (merged[^1].Start, Math.Max(merged[^1].End, w.End));
            }
            else
            {
                merged.Add(w);
            }
        }

        return merged;
    }

    private static List<(int Start, int End)> Subtract(List<(int Start, int End)> windows, (int Start, int End) block)
    {
        var result = new List<(int Start, int End)>();
        foreach (var w in windows)
        {
            if (block.End <= w.Start || block.Start >= w.End)
            {
                result.Add(w);
                continue;
            }

            if (block.Start > w.Start)
            {
                result.Add((w.Start, Math.Min(block.Start, w.End)));
            }

            if (block.End < w.End)
            {
                result.Add((Math.Max(block.End, w.Start), w.End));
            }
        }

        return result;
    }

    private static bool Overlaps(
        DateTime start, DateTime end, IReadOnlyList<(DateTime StartUtc, DateTime EndUtc)> existing) =>
        existing.Any(a => start < a.EndUtc && a.StartUtc < end);
}
