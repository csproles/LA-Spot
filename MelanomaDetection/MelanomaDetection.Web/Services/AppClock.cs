namespace MelanomaDetection.Web.Services;

/// <summary>
/// The clock for anything shown to a person as "today" or a time of day. The app
/// serves Louisiana, but a container's own clock is normally UTC -- so the raw
/// <c>DateTime.Now</c> would say "Good morning" and tomorrow's date on a Louisiana
/// evening. This pins those displays to US Central Time, falling back to the
/// machine's own clock if the zone can't be found (e.g. an image without tzdata).
/// </summary>
public static class AppClock
{
    private static readonly TimeZoneInfo? Zone = FindCentral();

    public static DateTime Now => Zone is null ? DateTime.Now : TimeZoneInfo.ConvertTime(DateTime.UtcNow, Zone);

    /// <summary>Morning until noon, afternoon until 6 pm, evening after that.</summary>
    public static string GreetingFor(int hour) => hour switch
    {
        < 12 => "Good morning",
        < 18 => "Good afternoon",
        _ => "Good evening",
    };

    public static string Greeting => GreetingFor(Now.Hour);

    private static TimeZoneInfo? FindCentral()
    {
        foreach (var id in new[] { "America/Chicago", "Central Standard Time" })
        {
            try
            {
                return TimeZoneInfo.FindSystemTimeZoneById(id);
            }
            catch (Exception ex) when (ex is TimeZoneNotFoundException or InvalidTimeZoneException)
            {
            }
        }

        return null;
    }
}
