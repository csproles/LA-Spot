namespace MelanomaDetection.Web.Services.Scheduling;

/// <summary>
/// Converts a UTC instant to the visitor's own timezone -- the IANA id JS
/// interop reads from the browser via scheduling.js's getTimeZone(). Falls
/// back to displaying UTC when the id hasn't arrived yet (first render,
/// before OnAfterRenderAsync's JS call resolves) or isn't recognized.
/// </summary>
public static class ClientTimeZone
{
    public static DateTime ToLocal(DateTime utc, string? timeZoneId)
    {
        if (timeZoneId is null)
        {
            return utc;
        }

        try
        {
            return TimeZoneInfo.ConvertTimeFromUtc(utc, TimeZoneInfo.FindSystemTimeZoneById(timeZoneId));
        }
        catch (TimeZoneNotFoundException)
        {
            return utc;
        }
    }

    /// <summary>Formats a UTC instant in the visitor's local time, appending " UTC" only when it couldn't be converted.</summary>
    public static string Format(DateTime utc, string? timeZoneId, string format) =>
        ToLocal(utc, timeZoneId).ToString(format) + (timeZoneId is null ? " UTC" : "");
}
