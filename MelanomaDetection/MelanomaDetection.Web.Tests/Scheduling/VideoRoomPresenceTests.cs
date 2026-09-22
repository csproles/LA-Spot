using MelanomaDetection.Web.Services.Scheduling;

namespace MelanomaDetection.Web.Tests.Scheduling;

public class VideoRoomPresenceTests
{
    [Fact]
    public void JoinThenLeaveRoundTrips()
    {
        var presence = new VideoRoomPresence();
        var appointmentId = Guid.NewGuid();

        Assert.False(presence.IsPresent(appointmentId, "patient"));
        presence.Join(appointmentId, "patient");
        Assert.True(presence.IsPresent(appointmentId, "patient"));
        Assert.False(presence.IsPresent(appointmentId, "provider"));

        presence.Leave(appointmentId, "patient");
        Assert.False(presence.IsPresent(appointmentId, "patient"));
    }

    [Fact]
    public void PresenceChangedFiresForTheAffectedAppointmentOnly()
    {
        var presence = new VideoRoomPresence();
        var watched = Guid.NewGuid();
        var other = Guid.NewGuid();
        var notifiedIds = new List<Guid>();
        presence.PresenceChanged += id => notifiedIds.Add(id);

        presence.Join(other, "provider");
        presence.Join(watched, "provider");

        Assert.Equal([other, watched], notifiedIds);
    }

    [Fact]
    public void DifferentRolesCanBothBePresentAtOnce()
    {
        var presence = new VideoRoomPresence();
        var appointmentId = Guid.NewGuid();

        presence.Join(appointmentId, "patient");
        presence.Join(appointmentId, "provider");

        Assert.True(presence.IsPresent(appointmentId, "patient"));
        Assert.True(presence.IsPresent(appointmentId, "provider"));
    }
}
