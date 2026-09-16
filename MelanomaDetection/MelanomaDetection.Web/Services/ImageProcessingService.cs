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
    /// Maps to POST /api/image/process. location/symptoms/notes are optional tags
    /// carried alongside the image so a later "Save to history" shows a properly
    /// labeled entry in the History list.
    /// </summary>
    public async Task<ProcessImageResponse> ProcessImageAsync(
        byte[] data, string filename, string? location = null,
        IEnumerable<string>? symptoms = null, string? notes = null)
    {
        using var content = new MultipartFormDataContent();
        using var fileContent = new ByteArrayContent(data);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue(GetContentType(filename));
        content.Add(fileContent, "file", filename);

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

    /// <summary>Maps to POST /api/image/save/{id} -- marks a processed check as saved.</summary>
    public async Task SaveToHistoryAsync(string processingId)
    {
        using var response = await SendAsync(() =>
            _httpClient.PostAsync($"/api/image/save/{Uri.EscapeDataString(processingId)}", null));
    }

    /// <summary>Maps to GET /api/image/history -- all checks that have been saved.</summary>
    public async Task<List<HistoryEntry>> GetHistoryAsync()
    {
        using var response = await SendAsync(() => _httpClient.GetAsync("/api/image/history"));

        var result = await response.Content.ReadFromJsonAsync<HistoryResponse>();
        return result?.Entries ?? new List<HistoryEntry>();
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
