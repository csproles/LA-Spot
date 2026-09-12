// Placeholder markers — swap for real provider data once there's a data source.
const providers = [
    { name: 'Baton Rouge Dermatology', lat: 30.4515, lng: -91.1871 },
    { name: 'New Orleans Skin Clinic', lat: 29.9511, lng: -90.0715 },
    { name: 'Shreveport Dermatology Center', lat: 32.5252, lng: -93.7502 },
];

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

function renderMap(container) {
    const map = new google.maps.Map(container, {
        center: { lat: 30.98, lng: -91.96 },
        zoom: 7,
    });

    providers.forEach((provider) => {
        new google.maps.Marker({
            position: { lat: provider.lat, lng: provider.lng },
            map,
            title: provider.name,
        });
    });
}

const container = document.getElementById('skin-check-map');
if (container) {
    loadMapsScript(container.dataset.apiKey)
        .then(() => renderMap(container))
        .catch((err) => console.error(err));
}
