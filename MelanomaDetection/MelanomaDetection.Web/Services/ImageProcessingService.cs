using System.Net.Http.Headers;
using System.Net.Http.Json;
using MelanomaDetection.Web.Models;

namespace MelanomaDetection.Web.Services;

/// <summary>
/// Thrown for any failure talking to the Flask API (network, timeout, or an error
/// response). <see cref="Exception.Message"/> is always safe to show directly to the user.
/// </summary>
public class ImageProcessingApiException : Exception
{
    public ImageProcessingApiException(string message) : base(message)
    {
    }
}

/// <summary>
/// Talks to the Python Flask image-processing API (see MelanomaDetection.Python/main.py).
/// </summary>
public class ImageProcessingService
{
    private readonly HttpClient _httpClient;

    public ImageProcessingService(HttpClient httpClient)
    {
        _httpClient = httpClient;
    }

    /// <summary>
    /// Maps to POST /api/image/process. When spotId is given the backend files
    /// the check under that spot and scores "Evolving" against the spot's
    /// previous check. location/symptoms/notes are optional tags carried
    /// alongside the image.
    /// </summary>
    public async Task<ProcessImageResponse> ProcessImageAsync(
        byte[] data, string filename, string? spotId = null, string? location = null,
        IEnumerable<string>? symptoms = null, string? notes = null)
    {
        using var content = new MultipartFormDataContent();
        using var fileContent = new ByteArrayContent(data);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue(GetContentType(filename));
        content.Add(fileContent, "file", filename);

        if (!string.IsNullOrWhiteSpace(spotId))
        {
            content.Add(new StringContent(spotId), "spot_id");
        }

        if (!string.IsNullOrWhiteSpace(location))
        {
            content.Add(new StringContent(location), "location");
        }

        foreach (var symptom in symptoms ?? Enumerable.Empty<string>())
        {
            content.Add(new StringContent(symptom), "symptoms");
        }

        if (!string.IsNullOrWhiteSpace(notes))
        {
            content.Add(new StringContent(notes), "notes");
        }

        using var response = await SendAsync(() => _httpClient.PostAsync("/api/image/process", content));

        var result = await response.Content.ReadFromJsonAsync<ProcessImageResponse>();
        return result ?? throw new ImageProcessingApiException("The analysis service returned an empty response.");
    }

    /// <summary>Maps to GET /api/image/results/{id}.</summary>
    public async Task<ImageProcessingResults> GetResultsAsync(string processingId)
    {
        using var response = await SendAsync(() =>
            _httpClient.GetAsync($"/api/image/results/{Uri.EscapeDataString(processingId)}"));

        var result = await response.Content.ReadFromJsonAsync<ImageProcessingResults>();
        return result ?? throw new ImageProcessingApiException("The analysis service returned an empty response.");
    }

    /// <summary>
    /// Maps to POST /api/image/explain/{id} -- an on-demand, per-processingId-cached
    /// call to the OpenAI-backed plain-language explainer. Only call this in response
    /// to an explicit user action (e.g. a button click), never automatically on page
    /// load, since each first call for a given id costs a real OpenAI API call.
    /// </summary>
    public async Task<string> ExplainAsync(string processingId)
    {
        using var response = await SendAsync(() =>
            _httpClient.PostAsync($"/api/image/explain/{Uri.EscapeDataString(processingId)}", null));

        var result = await response.Content.ReadFromJsonAsync<ExplainResponse>();
        return result?.Explanation ?? throw new ImageProcessingApiException("The analysis service returned an empty response.");
    }

    /// <summary>
    /// Maps to POST /api/image/save/{id} -- persists a processed check to history.
    /// Symptoms and notes are collected after the photo is analyzed and its
    /// outline confirmed, so they're attached here rather than at process time.
    /// </summary>
    public async Task SaveToHistoryAsync(string processingId, IEnumerable<string>? symptoms = null, string? notes = null)
    {
        using var response = await SendAsync(() =>
            _httpClient.PostAsJsonAsync(
                $"/api/image/save/{Uri.EscapeDataString(processingId)}",
                new { symptoms = symptoms?.ToList(), notes }));
    }

    /// <summary>Maps to GET /api/image/history -- all saved checks, newest first.</summary>
    public async Task<List<HistoryEntry>> GetHistoryAsync()
    {
        using var response = await SendAsync(() => _httpClient.GetAsync("/api/image/history"));

        var result = await response.Content.ReadFromJsonAsync<HistoryResponse>();
        return result?.Entries ?? new List<HistoryEntry>();
    }

    /// <summary>Maps to GET /api/spots -- every tracked spot with its aggregates and next-due date.</summary>
    public async Task<List<Spot>> GetSpotsAsync()
    {
        using var response = await SendAsync(() => _httpClient.GetAsync("/api/spots"));

        var result = await response.Content.ReadFromJsonAsync<SpotsResponse>();
        return result?.Spots ?? new List<Spot>();
    }

    /// <summary>Maps to GET /api/spots/{id} -- one spot with its full check timeline.</summary>
    public async Task<SpotDetail> GetSpotAsync(string spotId)
    {
        using var response = await SendAsync(() =>
            _httpClient.GetAsync($"/api/spots/{Uri.EscapeDataString(spotId)}"));

        var result = await response.Content.ReadFromJsonAsync<SpotDetail>();
        return result ?? throw new ImageProcessingApiException("The analysis service returned an empty response.");
    }

    /// <summary>Maps to POST /api/spots.</summary>
    public async Task<Spot> CreateSpotAsync(string label, string bodyRegion)
    {
        using var response = await SendAsync(() =>
            _httpClient.PostAsJsonAsync("/api/spots", new { label, bodyRegion }));

        var result = await response.Content.ReadFromJsonAsync<Spot>();
        return result ?? throw new ImageProcessingApiException("The analysis service returned an empty response.");
    }

    /// <summary>Maps to PATCH /api/spots/{id} -- rename and/or archive.</summary>
    public async Task<Spot> UpdateSpotAsync(string spotId, string? label = null, bool? archived = null)
    {
        using var response = await SendAsync(() =>
            _httpClient.PatchAsJsonAsync($"/api/spots/{Uri.EscapeDataString(spotId)}", new { label, archived }));

        var result = await response.Content.ReadFromJsonAsync<Spot>();
        return result ?? throw new ImageProcessingApiException("The analysis service returned an empty response.");
    }

    /// <summary>Maps to GET /api/profile. Configured is false until the user fills it in.</summary>
    public async Task<RiskProfile> GetProfileAsync()
    {
        using var response = await SendAsync(() => _httpClient.GetAsync("/api/profile"));

        var result = await response.Content.ReadFromJsonAsync<RiskProfile>();
        return result ?? new RiskProfile();
    }

    /// <summary>Maps to PUT /api/profile.</summary>
    public async Task<RiskProfile> SaveProfileAsync(RiskProfile profile)
    {
        using var response = await SendAsync(() =>
            _httpClient.PutAsJsonAsync("/api/profile", new
            {
                fullName = profile.FullName,
                location = profile.Location,
                sunExposure = profile.SunExposure,
                fitzpatrick = profile.Fitzpatrick,
                familyHistory = profile.FamilyHistory,
                blisteringSunburns = profile.BlisteringSunburns,
                manyMoles = profile.ManyMoles,
            }));

        var result = await response.Content.ReadFromJsonAsync<RiskProfile>();
        return result ?? throw new ImageProcessingApiException("The analysis service returned an empty response.");
    }

    /// <summary>
    /// Sends a request and converts every failure mode (unreachable server, the
    /// HttpClient's configured timeout, or a non-2xx response) into a single
    /// <see cref="ImageProcessingApiException"/> carrying a user-friendly message.
    /// </summary>
    private async Task<HttpResponseMessage> SendAsync(Func<Task<HttpResponseMessage>> send)
    {
        HttpResponseMessage response;
        try
        {
            response = await send();
        }
        catch (TaskCanceledException)
        {
            // We never pass our own CancellationToken, so this can only be HttpClient.Timeout firing.
            throw new ImageProcessingApiException(
                "The analysis service took too long to respond (30s timeout). Please try again.");
        }
        catch (HttpRequestException)
        {
            throw new ImageProcessingApiException(
                "Could not reach the analysis service. Make sure the backend is running and try again.");
        }

        if (!response.IsSuccessStatusCode)
        {
            var message = await TryReadErrorMessageAsync(response);
            response.Dispose();
            throw new ImageProcessingApiException(message ?? $"The analysis service returned an unexpected error ({(int)response.StatusCode}).");
        }

        return response;
    }

    private static async Task<string?> TryReadErrorMessageAsync(HttpResponseMessage response)
    {
        try
        {
            var payload = await response.Content.ReadFromJsonAsync<ErrorResponse>();
            return payload?.Error;
        }
        catch
        {
            return null;
        }
    }

    private static string GetContentType(string filename)
    {
        return Path.GetExtension(filename).ToLowerInvariant() switch
        {
            ".png" => "image/png",
            ".jpg" or ".jpeg" => "image/jpeg",
            ".bmp" => "image/bmp",
            ".webp" => "image/webp",
            _ => "application/octet-stream",
        };
    }
}
