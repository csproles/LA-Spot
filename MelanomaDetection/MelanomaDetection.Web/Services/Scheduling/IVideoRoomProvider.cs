namespace MelanomaDetection.Web.Services.Scheduling;

/// <summary>A room to embed for one appointment.</summary>
public sealed record VideoRoom(string RoomName, string JoinUrl);

/// <summary>
/// Creates the video room for a newly booked appointment. Jitsi (see
/// <see cref="JitsiVideoRoomProvider"/>) is the only implementation today, but
/// callers (BookingService) depend on this interface, not on Jitsi directly,
/// so swapping in Google Meet later (via the Calendar API's conferenceData)
/// means adding a new implementation and changing one DI registration in
/// Program.cs -- not touching BookingService or any page.
/// </summary>
public interface IVideoRoomProvider
{
    VideoRoom CreateRoom(Guid appointmentId);
}
