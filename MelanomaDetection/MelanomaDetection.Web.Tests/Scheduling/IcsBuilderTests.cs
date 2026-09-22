using MelanomaDetection.Web.Services.Scheduling;

namespace MelanomaDetection.Web.Tests.Scheduling;

public class IcsBuilderTests
{
    private static readonly DateTime NowUtc = new(2027, 6, 1, 12, 0, 0, DateTimeKind.Utc);
    private static readonly DateTime StartUtc = new(2027, 6, 7, 15, 0, 0, DateTimeKind.Utc);
    private static readonly DateTime EndUtc = new(2027, 6, 7, 15, 30, 0, DateTimeKind.Utc);
    private static readonly Guid AppointmentId = Guid.Parse("11111111-1111-1111-1111-111111111111");

    [Fact]
    public void ProducesAWellFormedVeventWithStartAndEndTimes()
    {
        var ics = IcsBuilder.BuildVisitEvent(AppointmentId, StartUtc, EndUtc, "LA Spot video visit", null, IcsStatus.Confirmed, NowUtc);

        Assert.Contains("BEGIN:VCALENDAR\r\n", ics);
        Assert.Contains("BEGIN:VEVENT\r\n", ics);
        Assert.Contains("DTSTART:20270607T150000Z\r\n", ics);
        Assert.Contains("DTEND:20270607T153000Z\r\n", ics);
        Assert.Contains("STATUS:CONFIRMED\r\n", ics);
        Assert.Contains("END:VEVENT\r\n", ics);
        Assert.Contains("END:VCALENDAR\r\n", ics);
    }

    [Fact]
    public void CancelledStatusUsesCancelMethodAndStatus()
    {
        var ics = IcsBuilder.BuildVisitEvent(AppointmentId, StartUtc, EndUtc, "Visit", null, IcsStatus.Cancelled, NowUtc);

        Assert.Contains("METHOD:CANCEL\r\n", ics);
        Assert.Contains("STATUS:CANCELLED\r\n", ics);
    }

    [Theory]
    [InlineData("Reason: itchy, red; worried", "Reason: itchy\\, red\\; worried")]
    [InlineData(@"C:\path\like\text", @"C:\\path\\like\\text")]
    public void EscapesReservedTextCharacters(string input, string expectedEscaped)
    {
        var ics = IcsBuilder.BuildVisitEvent(AppointmentId, StartUtc, EndUtc, "Visit", input, IcsStatus.Confirmed, NowUtc);

        Assert.Contains($"DESCRIPTION:{expectedEscaped}\r\n", ics);
    }

    [Fact]
    public void ANewlineInTheDescriptionCannotInjectAnExtraCalendarComponent()
    {
        // A visit reason is untrusted patient input. If a real newline reached the
        // file unescaped, "BEGIN:VEVENT" on its own line would start a second,
        // attacker-controlled event inside the same .ics. Confirm that never happens:
        // the newline must stay encoded as the two-character escape sequence "\n"
        // inside DESCRIPTION's single value, not become a second raw line.
        var maliciousReason = "normal text\nBEGIN:VEVENT\r\nSUMMARY:Injected\r\nEND:VEVENT";

        var ics = IcsBuilder.BuildVisitEvent(AppointmentId, StartUtc, EndUtc, "Visit", maliciousReason, IcsStatus.Confirmed, NowUtc);

        var lines = ics.Split("\r\n", StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(1, lines.Count(l => l is "BEGIN:VEVENT" or " BEGIN:VEVENT"));
        Assert.DoesNotContain(lines, l => l.TrimStart() == "SUMMARY:Injected");
    }

    [Fact]
    public void FoldsLinesLongerThan75CharactersWithASpaceContinuation()
    {
        var longDescription = new string('a', 200);

        var ics = IcsBuilder.BuildVisitEvent(AppointmentId, StartUtc, EndUtc, "Visit", longDescription, IcsStatus.Confirmed, NowUtc);

        var rawLines = ics.Split("\r\n", StringSplitOptions.RemoveEmptyEntries);
        Assert.All(rawLines, line => Assert.True(line.Length <= 75, $"Line exceeded 75 chars: '{line}'"));
        Assert.Contains(rawLines, l => l.StartsWith(' ')); // a folded continuation line
    }
}
