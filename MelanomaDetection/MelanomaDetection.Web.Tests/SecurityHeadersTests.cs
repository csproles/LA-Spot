using MelanomaDetection.Web.Services;
using MelanomaDetection.Web.Services.Scheduling;

namespace MelanomaDetection.Web.Tests;

public class SecurityHeadersTests
{
    private static string Directive(string policy, string name) =>
        policy.Split(';', StringSplitOptions.TrimEntries).Single(d => d.StartsWith(name + " ", StringComparison.Ordinal));

    [Fact]
    public void TheVideoVisitHostCanBeFramedButNothingElseCan()
    {
        var frameSrc = Directive(SecurityHeaders.BuildContentSecurityPolicy("n"), "frame-src");

        Assert.Equal($"frame-src {JitsiVideoRoomProvider.Origin}", frameSrc);
    }

    [Theory]
    [InlineData("camera")]
    [InlineData("microphone")]
    [InlineData("display-capture")]
    public void TheVideoVisitHostGetsTheDevicesACallNeeds(string feature)
    {
        var entry = SecurityHeaders.PermissionsPolicy.Split(", ").Single(e => e.StartsWith(feature + "=", StringComparison.Ordinal));

        Assert.Contains($"\"{JitsiVideoRoomProvider.Origin}\"", entry);
        Assert.Contains("self", entry);
    }

    [Fact]
    public void RoomLinksPointAtTheAllowedHost()
    {
        var room = new JitsiVideoRoomProvider().CreateRoom(Guid.NewGuid());

        Assert.StartsWith(JitsiVideoRoomProvider.Origin + "/", room.JoinUrl);
    }
}
