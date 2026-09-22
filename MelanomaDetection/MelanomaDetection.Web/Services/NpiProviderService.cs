using MelanomaDetection.Web.Models;
using Microsoft.Extensions.Caching.Memory;

namespace MelanomaDetection.Web.Services;

/// <summary>
/// Looks up real dermatology providers for a city from the NPI Registry (CMS's free,
/// keyless public database of every licensed US healthcare provider) and geocodes their
/// addresses via the free US Census geocoder. Both are unauthenticated government APIs --
/// no billing account, no API key.
/// </summary>
public class NpiProviderService
{
    private readonly HttpClient _httpClient;
    private readonly IMemoryCache _cache;
    private readonly ILogger<NpiProviderService> _logger;
    private readonly string? _googleMapsApiKey;

    public NpiProviderService(HttpClient httpClient, IMemoryCache cache, ILogger<NpiProviderService> logger, IConfiguration configuration)
    {
        _httpClient = httpClient;
        _cache = cache;
        _logger = logger;
        _googleMapsApiKey = configuration["GoogleMaps:ApiKey"];
    }

    /// <summary>
    /// Returns up to <paramref name="limit"/> dermatology providers registered in the given
    /// city/state. Results (including geocoding) are cached for 12 hours per city, since both
    /// upstream APIs are free/keyless and should be used politely. Returns an empty list --
    /// never throws -- if either upstream API is unreachable or returns no matches, since a
    /// missing "find care" panel shouldn't take down the map page.
    /// </summary>
    public async Task<IReadOnlyList<DermatologyProvider>> GetProvidersAsync(
        string city, string state = "LA", int limit = 8, CancellationToken cancellationToken = default)
    {
        var cacheKey = $"npi-providers:{state}:{city}";
        if (_cache.TryGetValue(cacheKey, out IReadOnlyList<DermatologyProvider>? cached) && cached is not null)
        {
            return cached;
        }

        var providers = await FetchProvidersAsync(city, state, limit, cancellationToken);
        _cache.Set(cacheKey, providers, TimeSpan.FromHours(12));
        return providers;
    }

    private async Task<IReadOnlyList<DermatologyProvider>> FetchProvidersAsync(
        string city, string state, int limit, CancellationToken cancellationToken)
    {
        NpiSearchResponse? search;
        try
        {
            var url = "https://npiregistry.cms.hhs.gov/api/" +
                $"?version=2.1&state={Uri.EscapeDataString(state)}&city={Uri.EscapeDataString(city)}" +
                $"&taxonomy_description={Uri.EscapeDataString("Dermatology")}&limit={limit}";
            search = await _httpClient.GetFromJsonAsync<NpiSearchResponse>(url, cancellationToken);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or System.Text.Json.JsonException)
        {
            _logger.LogWarning(ex, "NPI Registry lookup failed for {City}, {State}", city, state);
            return [];
        }

        if (search?.Results is not { Count: > 0 })
        {
            return [];
        }

        // Geocode addresses in parallel -- each is a separate, independent request to the
        // Census geocoder, so there's no benefit to doing this sequentially.
        var geocodeTasks = search.Results.Select(r => BuildProviderAsync(r, cancellationToken));
        var providers = await Task.WhenAll(geocodeTasks);

        return providers.Where(p => p is not null).Select(p => p!).ToList();
    }

    private async Task<DermatologyProvider?> BuildProviderAsync(NpiResult result, CancellationToken cancellationToken)
    {
        var address = result.Addresses.FirstOrDefault(a => a.AddressPurpose == "LOCATION")
            ?? result.Addresses.FirstOrDefault();
        if (address is null || string.IsNullOrWhiteSpace(address.AddressLine1))
        {
            return null;
        }

        var name = !string.IsNullOrWhiteSpace(result.Basic.OrganizationName)
            ? result.Basic.OrganizationName!
            : string.Join(' ', new[] { result.Basic.NamePrefix, result.Basic.FirstName, result.Basic.LastName }
                    .Select(CleanNpiField).Where(s => s is not null))
                + (CleanNpiField(result.Basic.Credential) is { } credential ? $", {credential}" : "");

        if (string.IsNullOrWhiteSpace(name))
        {
            return null;
        }

        var fullAddress = $"{address.AddressLine1}, {address.City}, {address.State} {FormatZip(address.PostalCode)}";
        var coordinatesTask = GeocodeAsync(fullAddress, cancellationToken);
        var ratingTask = GetRatingAsync($"{name}, {fullAddress}", cancellationToken);
        await Task.WhenAll(coordinatesTask, ratingTask);
        var coordinates = coordinatesTask.Result;
        var rating = ratingTask.Result;

        return new DermatologyProvider(name, fullAddress, address.TelephoneNumber, coordinates?.Lat, coordinates?.Lng, rating?.Rating, rating?.Count);
    }

    /// <summary>Google Places rating for a provider, via one Find Place from Text call (its
    /// "fields" can return rating/user_ratings_total directly, so no second Place Details
    /// call is needed). Returns null -- never throws -- if no key is configured, no match is
    /// found, or the lookup fails; a missing rating just means the card shows none.</summary>
    private async Task<(double Rating, int Count)?> GetRatingAsync(string query, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(_googleMapsApiKey))
        {
            return null;
        }

        try
        {
            var url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json" +
                $"?input={Uri.EscapeDataString(query)}&inputtype=textquery&fields=rating,user_ratings_total&key={_googleMapsApiKey}";
            var response = await _httpClient.GetFromJsonAsync<GoogleFindPlaceResponse>(url, cancellationToken);
            var candidate = response?.Candidates.FirstOrDefault();
            return candidate?.Rating is { } value ? (value, candidate.UserRatingsTotal ?? 0) : null;
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or System.Text.Json.JsonException)
        {
            _logger.LogWarning(ex, "Google Places rating lookup failed for {Query}", query);
            return null;
        }
    }

    private async Task<(double Lat, double Lng)?> GeocodeAsync(string address, CancellationToken cancellationToken)
    {
        try
        {
            var url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress" +
                $"?address={Uri.EscapeDataString(address)}&benchmark=Public_AR_Current&format=json";
            var response = await _httpClient.GetFromJsonAsync<CensusGeocodeResponse>(url, cancellationToken);
            var match = response?.Result.AddressMatches.FirstOrDefault();
            return match is null ? null : (match.Coordinates.Lat, match.Coordinates.Lng);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or System.Text.Json.JsonException)
        {
            _logger.LogWarning(ex, "Census geocoding failed for address {Address}", address);
            return null;
        }
    }

    /// <summary>NPI Registry uses the literal string "--" as a "no value" placeholder for
    /// name-part fields (prefix, suffix, etc.) instead of omitting them.</summary>
    private static string? CleanNpiField(string? value) =>
        string.IsNullOrWhiteSpace(value) || value == "--" ? null : value;

    /// <summary>NPI Registry returns ZIP+4 as a contiguous 9-digit string with no separator.</summary>
    private static string FormatZip(string postalCode) =>
        postalCode.Length == 9 ? $"{postalCode[..5]}-{postalCode[5..]}" : postalCode;
}
