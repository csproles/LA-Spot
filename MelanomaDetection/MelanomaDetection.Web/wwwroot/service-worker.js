// This app runs on Blazor Server (interactive components need a live
// SignalR connection), so this worker does not attempt full offline
// interactivity. It precaches the static app shell for fast repeat loads,
// serves those assets cache-first, and falls back to offline.html when a
// page navigation fails with no network.
const CACHE_NAME = "skin-check-shell-v2";

const PRECACHE_URLS = [
    "offline.html",
    "manifest.json",
    "favicon.png",
    "icons/icon-192.png",
    "icons/icon-512.png",
    "icons/icon-maskable-512.png",
    "icons/apple-touch-icon.png",
];

self.addEventListener("install", (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(PRECACHE_URLS))
            .then(() => self.skipWaiting())
    );
});

self.addEventListener("activate", (event) => {
    event.waitUntil(
        caches.keys()
            .then((keys) => Promise.all(
                keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
            ))
            .then(() => self.clients.claim())
    );
});

function isShellAsset(url) {
    return url.origin === self.location.origin && (
        url.pathname.startsWith("/icons/") ||
        url.pathname.startsWith("/lib/") ||
        url.pathname === "/app.css" ||
        url.pathname === "/favicon.png" ||
        url.pathname === "/manifest.json"
    );
}

self.addEventListener("fetch", (event) => {
    const request = event.request;
    if (request.method !== "GET") {
        return;
    }

    const url = new URL(request.url);

    // Never intercept the SignalR circuit -- it relies on live
    // websocket/long-polling traffic, not cacheable HTTP responses.
    if (url.pathname.startsWith("/_blazor")) {
        return;
    }

    if (request.mode === "navigate") {
        event.respondWith(
            fetch(request).catch(() => caches.match("offline.html"))
        );
        return;
    }

    if (isShellAsset(url)) {
        event.respondWith(
            caches.match(request).then((cached) => {
                const network = fetch(request).then((response) => {
                    if (response.ok) {
                        const copy = response.clone();
                        caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
                    }
                    return response;
                }).catch(() => cached);
                return cached || network;
            })
        );
    }
});
