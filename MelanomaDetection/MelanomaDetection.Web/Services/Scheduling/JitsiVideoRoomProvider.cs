using System.Security.Cryptography;

namespace MelanomaDetection.Web.Services.Scheduling;

/// <summary>
/// Free, keyless video rooms on the public meet.jit.si server, embedded via
/// iframe (see VideoVisit.razor) -- no API key, no OAuth, no custom WebRTC.
/// The room name mixes in a random token so a leaked appointment id alone
/// isn't enough to guess the room; meet.jit.si itself has no further access
/// control, which is a fine trade for a demo and NOT how this would ship for
/// real patient data (that would need a self-hosted Jitsi with JWT auth, or
/// the Google Meet swap this interface exists for).
/// </summary>
public sealed class JitsiVideoRoomProvider : IVideoRoomProvider
{
    /// <summary>The video host's origin; SecurityHeaders allows it to be framed and to use the camera and mic.</summary>
    public const string Origin = "https://meet.jit.si";

    private const string BaseUrl = Origin + "/";

    public VideoRoom CreateRoom(Guid appointmentId)
    {
        var token = Convert.ToHexString(RandomNumberGenerator.GetBytes(6)).ToLowerInvariant();
        var roomName = $"lionspot-{appointmentId:N}-{token}";
        return new VideoRoom(roomName, BaseUrl + roomName);
    }
}
