using MelanomaDetection.Web.Services.Scheduling;

namespace MelanomaDetection.Web.Tests.Scheduling;

public class JitsiVideoRoomProviderTests
{
    [Fact]
    public void CreateRoomBuildsAJoinUrlOnTheJitsiRoomName()
    {
        var provider = new JitsiVideoRoomProvider();
        var appointmentId = Guid.NewGuid();

        var room = provider.CreateRoom(appointmentId);

        Assert.StartsWith($"lionspot-{appointmentId:N}-", room.RoomName);
        Assert.Equal($"https://meet.jit.si/{room.RoomName}", room.JoinUrl);
    }

    [Fact]
    public void CreateRoomNeverReusesARoomNameForTheSameAppointment()
    {
        var provider = new JitsiVideoRoomProvider();
        var appointmentId = Guid.NewGuid();

        var first = provider.CreateRoom(appointmentId);
        var second = provider.CreateRoom(appointmentId);

        Assert.NotEqual(first.RoomName, second.RoomName);
    }
}
