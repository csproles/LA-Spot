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
//
// The captured photo goes back to .NET as a Uint8Array, which Blazor hands over
// as an IJSStreamReference (streamed in chunks). It must never be returned as a
// base64 string: a Blazor Server circuit accepts at most 32 KB per message from
// the browser, a real photo is far larger, and going over the limit makes the
// server drop the whole connection.
(function () {
    // A phone sensor is 4000 px or more on the long side. That is more than the
    // analysis needs and often more than the 5 MB upload cap allows.
    const MAX_LONG_EDGE = 2560;
    const JPEG_QUALITIES = [0.92, 0.85, 0.75, 0.6, 0.45];

    function isSupported() {
        return !!(navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === 'function');
    }

    // Friendly text for the handful of getUserMedia failures worth telling
    // the person apart -- everything else collapses to a generic message.
    function friendlyError(error) {
        switch (error && error.name) {
            case 'NotAllowedError':
            case 'SecurityError':
            case 'PermissionDeniedError':
                return 'Camera access was denied. Allow camera access for this site, or use "Choose from gallery" instead.';
            case 'NotFoundError':
            case 'DevicesNotFoundError':
            case 'OverconstrainedError':
                return "No usable camera was found on this device. Use \"Choose from gallery\" instead.";
            case 'NotReadableError':
            case 'TrackStartError':
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
    // Asks for a high resolution first, because browsers otherwise default to
    // 640x480, then loosens the request if a browser or device rejects it.
    // Returns null on success, or a message to show the person on failure.
    async function start(videoElement, facingMode) {
        const hadStream = !!(videoElement && videoElement.srcObject);
        stop(videoElement);
        if (hadStream) {
            // iOS Safari can refuse the next camera ("could not start video source") when it is
            // requested the instant the previous one was stopped, as when switching cameras.
            await new Promise((resolve) => setTimeout(resolve, 250));
        }
        const facing = { ideal: facingMode || 'environment' };
        const attempts = [
            { video: { facingMode: facing, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false },
            { video: { facingMode: facing }, audio: false },
            { video: true, audio: false },
        ];

        let lastError = null;
        for (const constraints of attempts) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                // Safari on iOS only plays a video inline, without a fullscreen takeover
                // or an autoplay refusal, when these are set on the element itself.
                videoElement.muted = true;
                videoElement.playsInline = true;
                videoElement.setAttribute('playsinline', '');
                videoElement.srcObject = stream;
                await videoElement.play();
                return null;
            } catch (error) {
                lastError = error;
                stop(videoElement);
                // A refused permission or a camera in use isn't fixed by asking more
                // loosely, and asking again would only show the prompt twice.
                const name = error && error.name;
                if (name === 'NotAllowedError' || name === 'SecurityError' || name === 'PermissionDeniedError'
                    || name === 'NotReadableError' || name === 'TrackStartError') {
                    break;
                }
            }
        }
        return friendlyError(lastError);
    }

    function readBlob(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(new Uint8Array(reader.result));
            reader.onerror = () => reject(reader.error);
            reader.readAsArrayBuffer(blob);
        });
    }

    function dataUrlToBytes(dataUrl) {
        const binary = atob(dataUrl.slice(dataUrl.indexOf(',') + 1));
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return bytes;
    }

    // JPEG bytes of the canvas, or null if the browser couldn't encode it.
    function encodeJpeg(canvasElement, quality) {
        return new Promise((resolve) => {
            try {
                if (typeof canvasElement.toBlob === 'function') {
                    canvasElement.toBlob((blob) => {
                        if (!blob) {
                            resolve(null);
                            return;
                        }
                        readBlob(blob).then(resolve, () => resolve(null));
                    }, 'image/jpeg', quality);
                } else {
                    resolve(dataUrlToBytes(canvasElement.toDataURL('image/jpeg', quality)));
                }
            } catch (error) {
                resolve(null);
            }
        });
    }

    // While a camera starts up, browsers can report a placeholder frame (2x2 in
    // Chrome) before real frames arrive; a shutter tap then would "capture" a
    // photo too small to be of any use. .NET calls waitForFrame first and only
    // captures once it says a real frame is there -- "not ready" can't travel as
    // the capture's own result, because .NET reads that as a stream and neither a
    // null nor an empty one can be read.
    const MIN_FRAME_EDGE = 64;
    const FRAME_WAIT_MS = 2000;

    function hasRealFrame(videoElement) {
        return videoElement.videoWidth >= MIN_FRAME_EDGE && videoElement.videoHeight >= MIN_FRAME_EDGE;
    }

    async function waitForRealFrame(videoElement) {
        const deadline = Date.now() + FRAME_WAIT_MS;
        while (!hasRealFrame(videoElement) && Date.now() < deadline) {
            await new Promise((resolve) => setTimeout(resolve, 50));
        }
        return hasRealFrame(videoElement);
    }

    // Draws the video's current frame to canvasElement and returns it as JPEG
    // bytes (a Uint8Array, which .NET receives as an IJSStreamReference), or null
    // when there is no real frame or it couldn't be encoded. The frame is scaled
    // down to MAX_LONG_EDGE and the quality lowered step by step until it fits
    // maxBytes, so a big sensor doesn't produce a photo the upload will refuse.
    async function capture(videoElement, canvasElement, maxBytes) {
        if (!hasRealFrame(videoElement)) {
            return null;
        }
        let width = videoElement.videoWidth;
        let height = videoElement.videoHeight;

        const scale = Math.min(1, MAX_LONG_EDGE / Math.max(width, height));
        width = Math.round(width * scale);
        height = Math.round(height * scale);
        canvasElement.width = width;
        canvasElement.height = height;
        canvasElement.getContext('2d').drawImage(videoElement, 0, 0, width, height);

        let bytes = null;
        for (const quality of JPEG_QUALITIES) {
            bytes = await encodeJpeg(canvasElement, quality);
            if (!bytes) {
                return null;
            }
            if (!maxBytes || bytes.length <= maxBytes) {
                break;
            }
        }
        return bytes;
    }

    // Opens the viewfinder <dialog> as a modal so it covers the screen from the top layer.
    function showOverlay(dialog) {
        if (dialog && !dialog.open && typeof dialog.showModal === 'function') {
            dialog.showModal();
        }
    }

    window.skinCheckCamera = { isSupported, start, waitForFrame: waitForRealFrame, capture, stop, showOverlay };
})();
