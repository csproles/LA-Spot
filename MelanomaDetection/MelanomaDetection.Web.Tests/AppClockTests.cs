using MelanomaDetection.Web.Services;

namespace MelanomaDetection.Web.Tests;

public class AppClockTests
{
    [Theory]
    [InlineData(0, "Good morning")]
    [InlineData(4, "Good morning")]
    [InlineData(11, "Good morning")]
    [InlineData(12, "Good afternoon")]
    [InlineData(17, "Good afternoon")]
    [InlineData(18, "Good evening")]
    [InlineData(20, "Good evening")]
    [InlineData(23, "Good evening")]
    public void GreetingFollowsTheHourOfDay(int hour, string expected)
    {
        Assert.Equal(expected, AppClock.GreetingFor(hour));
    }

    [Fact]
    public void NowIsCentralTimeRegardlessOfTheMachineZone()
    {
        var central = TimeZoneInfo.FindSystemTimeZoneById(
            OperatingSystem.IsWindows() ? "Central Standard Time" : "America/Chicago");
        var expected = TimeZoneInfo.ConvertTime(DateTime.UtcNow, central);

        Assert.InRange((AppClock.Now - expected).TotalSeconds, -5, 5);
    }
}
