using System.Text.Json.Serialization;

namespace MelanomaDetection.Web.Models;

/// <summary>A dermatology provider resolved from the NPI Registry and geocoded via the Census geocoder.
/// Rating/RatingCount/Website come from Google Places and are null if that lookup found no match
/// or failed -- never blocks the rest of the provider from showing.</summary>
public record DermatologyProvider(string Name, string Address, string? Phone, double? Lat, double? Lng, double? Rating = null, int? RatingCount = null, string? Website = null);

/// <summary>Shape of the "candidates" entries in Google's Find Place from Text response.</summary>
public class GooglePlaceCandidate
{
    [JsonPropertyName("place_id")]
    public string? PlaceId { get; set; }

    [JsonPropertyName("rating")]
    public double? Rating { get; set; }

    [JsonPropertyName("user_ratings_total")]
    public int? UserRatingsTotal { get; set; }
}

public class GoogleFindPlaceResponse
{
    [JsonPropertyName("candidates")]
    public List<GooglePlaceCandidate> Candidates { get; set; } = [];
}

/// <summary>Shape of the "result" object in Google's Place Details response, narrowed to the one
/// field (website) Find Place from Text can't return directly.</summary>
public class GooglePlaceDetailsResponse
{
    [JsonPropertyName("result")]
    public GooglePlaceDetailsResult? Result { get; set; }
}

public class GooglePlaceDetailsResult
{
    [JsonPropertyName("website")]
    public string? Website { get; set; }
}

/// <summary>Shape of https://npiregistry.cms.hhs.gov/api/ (version 2.1) responses.</summary>
public class NpiSearchResponse
{
    [JsonPropertyName("results")]
    public List<NpiResult> Results { get; set; } = [];
}

public class NpiResult
{
    [JsonPropertyName("basic")]
    public NpiBasic Basic { get; set; } = new();

    [JsonPropertyName("addresses")]
    public List<NpiAddress> Addresses { get; set; } = [];
}

public class NpiBasic
{
    [JsonPropertyName("organization_name")]
    public string? OrganizationName { get; set; }

    [JsonPropertyName("name_prefix")]
    public string? NamePrefix { get; set; }

    [JsonPropertyName("first_name")]
    public string? FirstName { get; set; }

    [JsonPropertyName("last_name")]
    public string? LastName { get; set; }

    [JsonPropertyName("credential")]
    public string? Credential { get; set; }
}

public class NpiAddress
{
    [JsonPropertyName("address_purpose")]
    public string AddressPurpose { get; set; } = string.Empty;

    [JsonPropertyName("address_1")]
    public string AddressLine1 { get; set; } = string.Empty;

    [JsonPropertyName("city")]
    public string City { get; set; } = string.Empty;

    [JsonPropertyName("state")]
    public string State { get; set; } = string.Empty;

    [JsonPropertyName("postal_code")]
    public string PostalCode { get; set; } = string.Empty;

    [JsonPropertyName("telephone_number")]
    public string? TelephoneNumber { get; set; }
}

/// <summary>Shape of https://geocoding.geo.census.gov/geocoder/locations/onelineaddress responses.</summary>
public class CensusGeocodeResponse
{
    [JsonPropertyName("result")]
    public CensusGeocodeResult Result { get; set; } = new();
}

public class CensusGeocodeResult
{
    [JsonPropertyName("addressMatches")]
    public List<CensusAddressMatch> AddressMatches { get; set; } = [];
}

public class CensusAddressMatch
{
    [JsonPropertyName("coordinates")]
    public CensusCoordinates Coordinates { get; set; } = new();
}

public class CensusCoordinates
{
    [JsonPropertyName("x")]
    public double Lng { get; set; }

    [JsonPropertyName("y")]
    public double Lat { get; set; }
}
