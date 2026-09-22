using MelanomaDetection.Web.Data;
using MelanomaDetection.Web.Services.Scheduling;

namespace MelanomaDetection.Web.Tests.Scheduling;

public class SlotGeneratorTests
{
    private static readonly DateTime FarPast = new(2020, 1, 1, 0, 0, 0, DateTimeKind.Utc);

    private static AvailabilityRule Rule(DayOfWeek day, string start, string end) => new()
    {
        Weekday = day,
        StartTime = TimeOnly.Parse(start),
        EndTime = TimeOnly.Parse(end),
    };

    [Fact]
    public void ConvertsLocalRuleTimeToUtc()
    {
        // 2026-01-05 is a Monday, CST (UTC-6) year-round in January -- no DST involved.
        var rules = new[] { Rule(DayOfWeek.Monday, "09:00", "09:30") };
        var date = new DateOnly(2026, 1, 5);

        var slots = SlotGenerator.GenerateOpenSlotsUtc(
            rules, [], [], "America/Chicago", appointmentLengthMinutes: 30, bufferMinutes: 0,
            date, date, FarPast);

        Assert.Equal([new DateTime(2026, 1, 5, 15, 0, 0, DateTimeKind.Utc)], slots);
    }

    [Fact]
    public void SkipsTheSpringForwardGapButKeepsSlotsEitherSideOfIt()
    {
        // America/Chicago springs forward on 2026-03-08: 02:00 local never happens.
        var rules = new[] { Rule(DayOfWeek.Sunday, "01:00", "04:00") };
        var date = new DateOnly(2026, 3, 8);

        var slots = SlotGenerator.GenerateOpenSlotsUtc(
            rules, [], [], "America/Chicago", appointmentLengthMinutes: 60, bufferMinutes: 0,
            date, date, FarPast);

        Assert.Equal(
            [
                new DateTime(2026, 3, 8, 7, 0, 0, DateTimeKind.Utc),  // 01:00 CST (UTC-6)
                new DateTime(2026, 3, 8, 8, 0, 0, DateTimeKind.Utc),  // 03:00 CDT (UTC-5); 02:00 doesn't exist
            ],
            slots);
    }

    [Fact]
    public void SpacesSlotsByAppointmentLengthPlusBuffer()
    {
        var rules = new[] { Rule(DayOfWeek.Monday, "09:00", "11:00") };
        var date = new DateOnly(2026, 1, 5);

        var slots = SlotGenerator.GenerateOpenSlotsUtc(
            rules, [], [], "UTC", appointmentLengthMinutes: 30, bufferMinutes: 15,
            date, date, FarPast);

        Assert.Equal(
            [
                new DateTime(2026, 1, 5, 9, 0, 0, DateTimeKind.Utc),
                new DateTime(2026, 1, 5, 9, 45, 0, DateTimeKind.Utc),
                new DateTime(2026, 1, 5, 10, 30, 0, DateTimeKind.Utc),
            ],
            slots);
    }

    [Fact]
    public void ExcludesSlotsThatOverlapAnExistingBooking()
    {
        var rules = new[] { Rule(DayOfWeek.Monday, "09:00", "11:00") };
        var date = new DateOnly(2026, 1, 5);
        var booked = new[]
        {
            (new DateTime(2026, 1, 5, 9, 30, 0, DateTimeKind.Utc), new DateTime(2026, 1, 5, 10, 0, 0, DateTimeKind.Utc)),
        };

        var slots = SlotGenerator.GenerateOpenSlotsUtc(
            rules, [], booked, "UTC", appointmentLengthMinutes: 30, bufferMinutes: 0,
            date, date, FarPast);

        Assert.Equal(
            [
                new DateTime(2026, 1, 5, 9, 0, 0, DateTimeKind.Utc),
                new DateTime(2026, 1, 5, 10, 0, 0, DateTimeKind.Utc),
                new DateTime(2026, 1, 5, 10, 30, 0, DateTimeKind.Utc),
            ],
            slots);
    }

    [Fact]
    public void WholeDayBlockExceptionRemovesAllSlotsThatDay()
    {
        var rules = new[] { Rule(DayOfWeek.Monday, "09:00", "11:00") };
        var date = new DateOnly(2026, 1, 5);
        var exceptions = new[] { new AvailabilityException { Date = date, IsBlocked = true } };

        var slots = SlotGenerator.GenerateOpenSlotsUtc(
            rules, exceptions, [], "UTC", appointmentLengthMinutes: 30, bufferMinutes: 0,
            date, date, FarPast);

        Assert.Empty(slots);
    }

    [Fact]
    public void PartialBlockExceptionCarvesOutOnlyThatRange()
    {
        var rules = new[] { Rule(DayOfWeek.Monday, "09:00", "11:00") };
        var date = new DateOnly(2026, 1, 5);
        var exceptions = new[]
        {
            new AvailabilityException { Date = date, IsBlocked = true, StartTime = TimeOnly.Parse("09:30"), EndTime = TimeOnly.Parse("10:00") },
        };

        var slots = SlotGenerator.GenerateOpenSlotsUtc(
            rules, exceptions, [], "UTC", appointmentLengthMinutes: 30, bufferMinutes: 0,
            date, date, FarPast);

        Assert.Equal(
            [
                new DateTime(2026, 1, 5, 9, 0, 0, DateTimeKind.Utc),
                new DateTime(2026, 1, 5, 10, 0, 0, DateTimeKind.Utc),
                new DateTime(2026, 1, 5, 10, 30, 0, DateTimeKind.Utc),
            ],
            slots);
    }

    [Fact]
    public void OneOffOpeningExceptionAddsSlotsOnADayWithNoWeeklyRule()
    {
        var date = new DateOnly(2026, 1, 3); // a Saturday; no weekly rule covers it
        var exceptions = new[]
        {
            new AvailabilityException { Date = date, IsBlocked = false, StartTime = TimeOnly.Parse("10:00"), EndTime = TimeOnly.Parse("10:30") },
        };

        var slots = SlotGenerator.GenerateOpenSlotsUtc(
            [], exceptions, [], "UTC", appointmentLengthMinutes: 30, bufferMinutes: 0,
            date, date, FarPast);

        Assert.Equal([new DateTime(2026, 1, 3, 10, 0, 0, DateTimeKind.Utc)], slots);
    }

    [Fact]
    public void ExcludesSlotsThatHaveAlreadyPassed()
    {
        var rules = new[] { Rule(DayOfWeek.Monday, "09:00", "10:00") };
        var date = new DateOnly(2026, 1, 5);
        var now = new DateTime(2026, 1, 5, 9, 30, 0, DateTimeKind.Utc);

        var slots = SlotGenerator.GenerateOpenSlotsUtc(
            rules, [], [], "UTC", appointmentLengthMinutes: 30, bufferMinutes: 0,
            date, date, now);

        Assert.Equal([new DateTime(2026, 1, 5, 9, 30, 0, DateTimeKind.Utc)], slots);
    }
}
