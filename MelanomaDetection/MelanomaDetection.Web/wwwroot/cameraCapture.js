// In-app camera viewfinder for the Upload flow's "Take a photo" step, backing
// Components/CheckFlow/CameraCapture.razor. Kept as plain functions on
// window.skinCheckCamera (not an ES module) to match theme.js's convention
// for the handful of small interop helpers this app needs.
//
// Why this exists instead of the plain <input type="file" capture="environment">
// the app used before: that hands the whole camera UI to the OS, so there is
// no in-app framing/retake before the shot is taken. getUserMedia gives a
// live preview inside the app instead. It only works in a secure context
// (HTTPS, or localhost) and on browsers that support it, so isSupported()
// lets the component fall back to the old file-input flow everywhere else.
(function () {
    function isSupported() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }

    // Friendly text for the handful of getUserMedia failures worth telling
    // the person apart -- everything else collapses to a generic message.
    function friendlyError(error) {
        switch (error && error.name) {
            case 'NotAllowedError':
            case 'SecurityError':
                return 'Camera access was denied. Allow camera access for this site, or use "Choose from gallery" instead.';
            case 'NotFoundError':
            case 'OverconstrainedError':
                return "No usable camera was found on this device. Use \"Choose from gallery\" instead.";
            case 'NotReadableError':
                return 'The camera is already in use by another app. Close it and try again, or use "Choose from gallery" instead.';
            default:
                return 'Could not start the camera. Use "Choose from gallery" instead.';
        }
    }

    // Stops whatever stream is currently attached to a <video>, if any. Safe
    // to call on an element that was never started.
    function stop(videoElement) {
        const stream = videoElement && videoElement.srcObject;
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            videoElement.srcObject = null;
        }
    }

    // Requests the given camera (facingMode is a hint, not a guarantee -- most
    // laptops only have one camera and ignore it) and starts it playing into
    // videoElement. Replaces any stream already attached to that element.
    // Returns null on success, or a message to show the person on failure.
    async function start(videoElement, facingMode) {
        stop(videoElement);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: { ideal: facingMode || 'environment' } },
                audio: false,
            });
            videoElement.srcObject = stream;
            await videoElement.play();
            return null;
        } catch (error) {
            return friendlyError(error);
        }
    }

    // Draws the video's current frame to canvasElement and returns it as a
    // base64 JPEG (no "data:" prefix -- the caller already knows the type).
    function capture(videoElement, canvasElement, quality) {
        const width = videoElement.videoWidth;
        const height = videoElement.videoHeight;
        if (!width || !height) {
            return null;
        }
        canvasElement.width = width;
        canvasElement.height = height;
        canvasElement.getContext('2d').drawImage(videoElement, 0, 0, width, height);
        const dataUrl = canvasElement.toDataURL('image/jpeg', quality || 0.92);
        const commaIndex = dataUrl.indexOf(',');
        return commaIndex >= 0 ? dataUrl.slice(commaIndex + 1) : dataUrl;
    }

    window.skinCheckCamera = { isSupported, start, capture, stop };
})();
