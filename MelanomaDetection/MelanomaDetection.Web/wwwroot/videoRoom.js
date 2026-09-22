// Fake in-app video room: shows the visitor their own camera so the room
// feels real, but never transmits it anywhere -- there is no signaling
// server, no peer connection, and no external video service (Zoom/Meet).
// See Services/Scheduling/VideoRoomPresence.cs for how each side learns the
// other has joined the room.
window.skinCheckVideoRoom = (() => {
    let stream = null;

    async function start(videoElementId) {
        const video = document.getElementById(videoElementId);
        if (!video || !navigator.mediaDevices?.getUserMedia) {
            return;
        }

        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            video.srcObject = stream;
        } catch {
            // Camera/mic denied or unavailable -- the room still works, just without a self-preview.
        }
    }

    function setAudioEnabled(enabled) {
        stream?.getAudioTracks().forEach(track => { track.enabled = enabled; });
    }

    function setVideoEnabled(enabled) {
        stream?.getVideoTracks().forEach(track => { track.enabled = enabled; });
    }

    function stop() {
        stream?.getTracks().forEach(track => track.stop());
        stream = null;
    }

    return { start, setAudioEnabled, setVideoEnabled, stop };
})();
