const STATEWIDE_CENTER = { lat: 30.98, lng: -91.96 };
const STATEWIDE_ZOOM = 7;
const FOCUS_ZOOM = 15;
const BOUNCE_MS = 1400;

let map;
let infoWindow;
let selectedMarker;
// Provider id (its index in the list Blazor renders) -> { marker, provider }, so
// a click on a card can find the pin that belongs to it.
let markers = new Map();

function loadMapsScript(apiKey) {
    return new Promise((resolve, reject) => {
        if (window.google?.maps) {
            resolve();
            return;
        }
        const script = document.createElement('script');
        script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}`;
        script.async = true;
        script.onload = () => resolve();
        script.onerror = () => reject(new Error('Failed to load Google Maps script'));
        document.head.appendChild(script);
    });
}

function clearMarkers() {
    infoWindow?.close();
    selectedMarker = null;
    markers.forEach(({ marker }) => marker.setMap(null));
    markers = new Map();
}

function showProviders(providers) {
    clearMarkers();
    (providers || [])
        .filter((provider) => provider.lat != null && provider.lng != null)
        .forEach((provider) => {
            const marker = new google.maps.Marker({
                position: { lat: provider.lat, lng: provider.lng },
                map,
                title: provider.name,
            });
            markers.set(provider.id, { marker, provider });
            // Clicking the pin itself is how most people expect to open a provider,
            // not just clicking its card in the list below the map.
            marker.addListener('click', () => focusProvider(provider.id));
        });
}

// The card that opens above the selected pin. Built from DOM nodes with
// textContent because every field comes from an external registry and must
// never be parsed as HTML. Inline styles because Google renders the window
// outside this component's scoped CSS, always on a white background.
function buildInfoContent(provider) {
    const card = document.createElement('div');
    card.style.cssText = 'max-width:240px;padding:2px 4px 4px;font-family:inherit;color:#1f2937;';

    const addLine = (text, css) => {
        if (!text) {
            return;
        }
        const line = document.createElement('div');
        line.style.cssText = css;
        line.textContent = text;
        card.appendChild(line);
    };

    addLine('Selected provider', 'font-size:11px;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;color:#b91c1c;margin-bottom:2px;');
    addLine(provider.name, 'font-size:15px;font-weight:700;line-height:1.25;margin-bottom:4px;');
    addLine(provider.address, 'font-size:13px;color:#4b5563;');
    addLine(provider.phone, 'font-size:13px;color:#4b5563;margin-top:2px;');

    if (provider.website) {
        const link = document.createElement('a');
        link.href = provider.website;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = 'Visit website';
        link.style.cssText = 'display:inline-block;font-size:13px;font-weight:600;color:#1d4ed8;margin-top:4px;';
        card.appendChild(link);
    }

    return card;
}

function focusProvider(id) {
    const entry = markers.get(id);
    if (!entry) {
        return false;
    }
    const { marker, provider } = entry;

    map.panTo(marker.getPosition());
    if ((map.getZoom() ?? 0) < FOCUS_ZOOM) {
        map.setZoom(FOCUS_ZOOM);
    }

    // Lift the chosen pin above any neighbours it overlaps, and drop the last one back.
    selectedMarker?.setZIndex(null);
    selectedMarker = marker;
    marker.setZIndex(google.maps.Marker.MAX_ZINDEX + 1);

    marker.setAnimation(google.maps.Animation.BOUNCE);
    setTimeout(() => marker.setAnimation(null), BOUNCE_MS);

    infoWindow ??= new google.maps.InfoWindow();
    infoWindow.setContent(buildInfoContent(provider));
    infoWindow.open({ map, anchor: marker });

    // On a narrow screen the list sits below the map, so bring the map back into view.
    map.getDiv().scrollIntoView({ behavior: 'smooth', block: 'center' });
    return true;
}

function renderMap(container) {
    map = new google.maps.Map(container, {
        center: STATEWIDE_CENTER,
        zoom: STATEWIDE_ZOOM,
    });
}

let mapReady;

// Exposed for Blazor interop (see MapCanvas.razor). init() is called explicitly from
// OnAfterRenderAsync rather than running at script-parse time — this component
// prerenders, and grabbing the container element too early binds the map to the
// pre-interactive DOM node, which Blazor's circuit attach then discards (the map
// keeps building silently onto an orphaned element).
//
// Region/parish data and the NPI Registry + Census geocoder lookups all live
// server-side (see NpiProviderService.cs); this module only ever renders
// whatever center/zoom/providers Blazor hands it.
window.skinCheckMap = {
    init(elementId, apiKey) {
        const container = document.getElementById(elementId);
        if (!container) {
            return Promise.reject(new Error('Map container not found'));
        }
        mapReady = loadMapsScript(apiKey).then(() => renderMap(container));
        mapReady.catch((err) => console.error(err));
        return mapReady;
    },
    showResults(payload) {
        if (!mapReady) {
            return Promise.resolve();
        }
        return mapReady.then(() => {
            if (payload?.center) {
                map.setCenter(payload.center);
                map.setZoom(payload.zoom ?? STATEWIDE_ZOOM);
            }
            showProviders(payload?.providers);
        });
    },
    // Pans and zooms to the provider's pin, bounces it and opens its label.
    // Resolves false when that provider has no pin (it couldn't be geocoded).
    focusProvider(id) {
        if (!mapReady) {
            return Promise.resolve(false);
        }
        return mapReady.then(() => focusProvider(id));
    },
    reset() {
        if (!mapReady) {
            return Promise.resolve();
        }
        return mapReady.then(() => {
            map.setCenter(STATEWIDE_CENTER);
            map.setZoom(STATEWIDE_ZOOM);
            clearMarkers();
        });
    },
};
