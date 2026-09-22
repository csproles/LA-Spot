using System.Text;

namespace MelanomaDetection.Web.Services.Scheduling;

public enum IcsStatus { Confirmed, Cancelled }

/// <summary>
/// Builds a minimal RFC 5545 .ics calendar event for a visit. Pure and
/// DB-free so it's directly unit-testable -- see
/// MelanomaDetection.Web.Tests/Scheduling/IcsBuilderTests.cs, which is where
/// the escaping is exercised: SUMMARY/DESCRIPTION come from a provider's
/// specialty and a patient's free-text visit reason, both untrusted input
/// that ends up inside a file a real calendar app will parse, so an
/// unescaped ';', ',', '\' or newline could corrupt the file or splice in
/// an attacker-controlled ICS property.
/// </summary>
public static class IcsBuilder
{
    public static string BuildVisitEvent(
        Guid appointmentId, DateTime startUtc, DateTime endUtc, string summary, string? description, IcsStatus status, DateTime nowUtc)
    {
        var lines = new List<string>
        {
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//LA Spot//Telehealth//EN",
            "METHOD:" + (status == IcsStatus.Cancelled ? "CANCEL" : "REQUEST"),
            "BEGIN:VEVENT",
            $"UID:{appointmentId}@laspot.local",
            $"DTSTAMP:{FormatUtc(nowUtc)}",
            $"DTSTART:{FormatUtc(startUtc)}",
            $"DTEND:{FormatUtc(endUtc)}",
            $"SUMMARY:{Escape(summary)}",
        };

        if (!string.IsNullOrWhiteSpace(description))
        {
            lines.Add($"DESCRIPTION:{Escape(description)}");
        }

        lines.Add("STATUS:" + (status == IcsStatus.Cancelled ? "CANCELLED" : "CONFIRMED"));
        lines.Add("SEQUENCE:" + (status == IcsStatus.Cancelled ? "1" : "0"));
        lines.Add("END:VEVENT");
        lines.Add("END:VCALENDAR");

        var sb = new StringBuilder();
        foreach (var line in lines)
        {
            foreach (var folded in Fold(line))
            {
                sb.Append(folded).Append("\r\n");
            }
        }

        return sb.ToString();
    }

    private static string FormatUtc(DateTime value) => value.ToUniversalTime().ToString("yyyyMMdd'T'HHmmss'Z'");

    /// <summary>Escapes RFC 5545 TEXT value special characters: backslash, semicolon, comma, and newline.
    /// This is what stops a visit reason containing e.g. a real newline from becoming a second ICS
    /// content line instead of staying inside this property's value.</summary>
    private static string Escape(string value) => value
        .Replace("\\", "\\\\")
        .Replace(";", "\\;")
        .Replace(",", "\\,")
        .Replace("\r\n", "\\n")
        .Replace("\n", "\\n");

    /// <summary>RFC 5545 line folding: no content line may exceed 75 octets; continuations start with a space.</summary>
    private static IEnumerable<string> Fold(string line)
    {
        const int limit = 75;
        if (line.Length <= limit)
        {
            yield return line;
            yield break;
        }

        yield return line[..limit];
        var rest = line[limit..];
        while (rest.Length > limit - 1)
        {
            yield return " " + rest[..(limit - 1)];
            rest = rest[(limit - 1)..];
        }

        if (rest.Length > 0)
        {
            yield return " " + rest;
        }
    }
}
