namespace MelanomaDetection.Web.Services;

/// <summary>
/// Louisiana's metro hubs, each with its own NPI Registry city query -- most
/// zip codes have no dermatologist of their own, so searching the zip's exact
/// city would often return nothing. <see cref="NearestHub"/> picks the
/// closest hub to a geocoded point (see NpiProviderService.GeocodeZipAsync),
/// so a zip code always resolves to a hub with real results.
/// </summary>
public static class LouisianaRegions
{
    public record Region(string CityLabel, string NpiCityQuery, double Lat, double Lng, int Zoom);

    private static readonly Region[] Hubs =
    [
        new("Shreveport", "Shreveport", 32.5252, -93.7502, 9),
        new("Monroe", "Monroe", 32.5093, -92.1193, 9),
        new("Alexandria", "Alexandria", 31.3113, -92.4451, 9),
        new("Lake Charles", "Lake Charles", 30.2266, -93.2174, 9),
        new("Lafayette", "Lafayette", 30.2241, -92.0198, 9),
        new("Baton Rouge", "Baton Rouge", 30.4515, -91.1871, 9),
        new("Hammond", "Hammond", 30.5044, -90.4615, 9),
        new("Houma", "Houma", 29.5958, -90.7195, 9),
        new("New Orleans", "New Orleans", 29.9511, -90.0715, 9),
    ];

    /// <summary>The hub closest (straight-line) to a geocoded point. Always
    /// returns a result -- Hubs is a fixed, non-empty list.</summary>
    public static Region NearestHub(double lat, double lng) =>
        Hubs.MinBy(hub => HaversineMiles(lat, lng, hub.Lat, hub.Lng))!;

    public static IReadOnlyList<Region> AllHubs => Hubs;

    /// <summary>A hub by its city name (case-insensitive), or null for an unknown or empty name.</summary>
    public static Region? FindHub(string? cityLabel) =>
        string.IsNullOrWhiteSpace(cityLabel)
            ? null
            : Hubs.FirstOrDefault(hub => string.Equals(hub.CityLabel, cityLabel.Trim(), StringComparison.OrdinalIgnoreCase));

    /// <summary>Straight-line miles between two hubs, 0 for the same hub.</summary>
    public static double MilesBetween(Region a, Region b) => HaversineMiles(a.Lat, a.Lng, b.Lat, b.Lng);

    private static double HaversineMiles(double lat1, double lng1, double lat2, double lng2)
    {
        const double earthRadiusMiles = 3958.8;
        double ToRadians(double degrees) => degrees * Math.PI / 180;

        var dLat = ToRadians(lat2 - lat1);
        var dLng = ToRadians(lng2 - lng1);
        var a = Math.Sin(dLat / 2) * Math.Sin(dLat / 2) +
                Math.Cos(ToRadians(lat1)) * Math.Cos(ToRadians(lat2)) *
                Math.Sin(dLng / 2) * Math.Sin(dLng / 2);
        var c = 2 * Math.Atan2(Math.Sqrt(a), Math.Sqrt(1 - a));
        return earthRadiusMiles * c;
    }
}
