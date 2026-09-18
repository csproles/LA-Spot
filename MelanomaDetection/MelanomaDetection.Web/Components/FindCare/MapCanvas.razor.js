const STATEWIDE_CENTER = { lat: 30.98, lng: -91.96 };
const STATEWIDE_ZOOM = 7;

let map;
let markers = [];

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
    markers.forEach((marker) => marker.setMap(null));
    markers = [];
}

function showProviders(providers) {
    clearMarkers();
    (providers || [])
        .filter((provider) => provider.lat != null && provider.lng != null)
        .forEach((provider) => {
            markers.push(new google.maps.Marker({
                position: { lat: provider.lat, lng: provider.lng },
                map,
                title: provider.name,
            }));
        });
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
