// Kept as a file (not an inline <script>) so the Content-Security-Policy can
// disallow inline scripts.
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('service-worker.js');
    });
}
