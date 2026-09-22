using System.Collections.Concurrent;

namespace MelanomaDetection.Web.Services.Scheduling;

/// <summary>
/// Tracks who has the video room open, in memory, for the fake in-app room:
/// no media or signaling ever crosses between participants, this only lets
/// each side see that the other has joined. Registered as a singleton.
/// ponytail: in-process only -- a real deployment behind multiple app
/// instances would need this in a shared store (e.g. Redis) instead.
/// </summary>
public sealed class VideoRoomPresence
{
    private readonly ConcurrentDictionary<Guid, HashSet<string>> present = new();
    private readonly object gate = new();

    public event Action<Guid>? PresenceChanged;

    public void Join(Guid appointmentId, string role)
    {
        lock (gate)
        {
            present.GetOrAdd(appointmentId, static _ => []).Add(role);
        }

        PresenceChanged?.Invoke(appointmentId);
    }

    public void Leave(Guid appointmentId, string role)
    {
        lock (gate)
        {
            if (present.TryGetValue(appointmentId, out var roles))
            {
                roles.Remove(role);
                if (roles.Count == 0)
                {
                    present.TryRemove(appointmentId, out _);
                }
            }
        }

        PresenceChanged?.Invoke(appointmentId);
    }

    public bool IsPresent(Guid appointmentId, string role)
    {
        lock (gate)
        {
            return present.TryGetValue(appointmentId, out var roles) && roles.Contains(role);
        }
    }
}
