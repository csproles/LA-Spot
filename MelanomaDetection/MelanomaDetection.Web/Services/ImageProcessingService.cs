using System.Net.Http.Headers;
using System.Text.Json;
using MelanomaDetection.Web.Models;
using MelanomaDetection.Web.Services.Account;
using MelanomaDetection.Web.Services.RateLimiting;
using Microsoft.Extensions.Caching.Memory;

namespace MelanomaDetection.Web.Services;

/// <summary>One page of GetHistoryPageAsync -- the entries for that page, and the account's total check count.</summary>
public record HistoryPage(List<HistoryEntry> Entries, int Total);

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
///
/// Every request is made on behalf of the signed-in person: their account id
/// travels in <see cref="UserIdHeader"/> and the API scopes every row it reads
/// or writes by it. The browser never talks to Flask directly, so this header
/// (together with the shared key set on the HttpClient) is the whole identity
/// story between the two services.
///
/// Inputs are checked here against <see cref="InputLimits"/> before anything is
/// sent, and the costly calls (analysis, explanation) are capped per person by
/// <see cref="OperationRateLimiter"/>. Both surface as
/// <see cref="ImageProcessingApiException"/>, so pages show them like any other error.
/// </summary>
public class ImageProcessingService
{
    public const string UserIdHeader = "X-User-Id";
    public const string InternalKeyHeader = "X-Internal-Api-Key";

    private static readonly TimeSpan ProfileCacheDuration = TimeSpan.FromSeconds(30);

    private readonly HttpClient _httpClient;
    private readonly CurrentUser _currentUser;
    private readonly OperationRateLimiter _limiter;
    private readonly IMemoryCache _cache;

    public ImageProcessingService(HttpClient httpClient, CurrentUser currentUser, OperationRateLimiter limiter, IMemoryCache cache)
    {
        _httpClient = httpClient;
        _currentUser = currentUser;
        _limiter = limiter;
        _cache = cache;
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
        if (data.Length == 0)
        {
            throw new ImageProcessingApiException("The photo is empty. Please choose another one.");
        }

        if (data.Length > InputLimits.ImageMaxBytes)
        {
            throw new ImageProcessingApiException("The photo is too large. Maximum allowed size is 5 MB.");
        }

        if (!InputLimits.LooksLikeImage(data))
        {
            throw new ImageProcessingApiException("That file doesn't look like a photo. Please use a JPEG, PNG or BMP image.");
        }

        location = RequireText(location, "Location", InputLimits.LocationMax);
        notes = RequireText(notes, "Notes", InputLimits.NotesMax, multiline: true);
        var checkedSymptoms = RequireSymptoms(symptoms);

        await ThrottleAsync(LimitedOperation.AnalyzePhoto);

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

        foreach (var symptom in checkedSymptoms)
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

    /// <summary>
    /// Maps to POST /predict -- the Kaggle-trained CNN+CatBoost melanoma risk
    /// model (risk_model.py), run alongside (not instead of) V5's own
    /// /api/image/process pipeline. age/sex/bodySite are optional: the model
    /// was trained tolerating missing metadata (see risk_model.py).
    /// </summary>
    public async Task<PredictResponse> PredictRiskAsync(
        byte[] data, string filename, int? age, string? sex, string? bodySite, string? linkedProcessingId = null)
    {
        if (data.Length == 0)
        {
            throw new ImageProcessingApiException("The photo is empty. Please choose another one.");
        }

        if (data.Length > InputLimits.ImageMaxBytes)
        {
            throw new ImageProcessingApiException("The photo is too large. Maximum allowed size is 5 MB.");
        }

        if (!InputLimits.LooksLikeImage(data))
        {
            throw new ImageProcessingApiException("That file doesn't look like a photo. Please use a JPEG, PNG or BMP image.");
        }

        await ThrottleAsync(LimitedOperation.AnalyzePhoto);

        using var content = new MultipartFormDataContent();
        using var fileContent = new ByteArrayContent(data);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue(GetContentType(filename));
        content.Add(fileContent, "image", filename);

        if (age is not null)
        {
            content.Add(new StringContent(age.Value.ToString()), "age");
        }

        if (!string.IsNullOrWhiteSpace(sex))
        {
            content.Add(new StringContent(sex), "sex");
        }

        if (!string.IsNullOrWhiteSpace(bodySite))
        {
            content.Add(new StringContent(bodySite), "body_site");
        }

        // The check this same photo was already processed as, so the service can give it the
        // risk model's own A, B and C (see PredictResponse.ApplyTo).
        if (!string.IsNullOrWhiteSpace(linkedProcessingId))
        {
            content.Add(new StringContent(linkedProcessingId), "linked_processing_id");
        }

        using var response = await SendAsync(() => _httpClient.PostAsync("/predict", content));

        var result = await response.Content.ReadFromJsonAsync<PredictResponse>();
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
        await ThrottleAsync(LimitedOperation.ExplainResults);

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
        var checkedSymptoms = symptoms is null ? null : RequireSymptoms(symptoms);
        var checkedNotes = notes is null ? null : RequireText(notes, "Notes", InputLimits.NotesMax, multiline: true);

        using var response = await SendAsync(() =>
            _httpClient.PostAsJsonAsync(
                $"/api/image/save/{Uri.EscapeDataString(processingId)}",
                new { symptoms = checkedSymptoms, notes = checkedNotes }));
    }

    /// <summary>Maps to GET /api/image/history -- all saved checks, newest first.</summary>
    public async Task<List<HistoryEntry>> GetHistoryAsync()
    {
        using var response = await SendAsync(() => _httpClient.GetAsync("/api/image/history"));

        var result = await response.Content.ReadFromJsonAsync<HistoryResponse>();
        return result?.Entries ?? new List<HistoryEntry>();
    }

    /// <summary>
    /// Maps to GET /api/image/history?limit=&amp;offset= -- one page of saved checks,
    /// newest first, plus the account's total check count. Use this instead of
    /// <see cref="GetHistoryAsync()"/> for a list that can grow without bound
    /// (the "All checks" page); callers that need the account's complete history
    /// at once (dashboard recent-checks widget, chat context, data export) should
    /// keep using the parameterless overload.
    /// </summary>
    public async Task<HistoryPage> GetHistoryPageAsync(int limit, int offset)
    {
        using var response = await SendAsync(() =>
            _httpClient.GetAsync($"/api/image/history?limit={limit}&offset={offset}"));

        var result = await response.Content.ReadFromJsonAsync<HistoryResponse>();
        return new HistoryPage(result?.Entries ?? new List<HistoryEntry>(), result?.Total ?? 0);
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
        label = RequireText(label, "Label", InputLimits.SpotLabelMax, required: true)!;
        bodyRegion = RequireText(bodyRegion, "Body region", InputLimits.BodyRegionMax, required: true)!;

        using var response = await SendAsync(() =>
            _httpClient.PostAsJsonAsync("/api/spots", new { label, bodyRegion }));

        var result = await response.Content.ReadFromJsonAsync<Spot>();
        return result ?? throw new ImageProcessingApiException("The analysis service returned an empty response.");
    }

    /// <summary>Maps to PATCH /api/spots/{id} -- rename and/or archive.</summary>
    public async Task<Spot> UpdateSpotAsync(string spotId, string? label = null, bool? archived = null)
    {
        label = RequireText(label, "Label", InputLimits.SpotLabelMax);

        using var response = await SendAsync(() =>
            _httpClient.PatchAsJsonAsync($"/api/spots/{Uri.EscapeDataString(spotId)}", new { label, archived }));

        var result = await response.Content.ReadFromJsonAsync<Spot>();
        return result ?? throw new ImageProcessingApiException("The analysis service returned an empty response.");
    }

    /// <summary>
    /// Maps to GET /api/profile. Configured is false until the user fills it in.
    /// Cached briefly per account -- Home, Onboarding and Profile all fetch it on
    /// load, and it rarely changes between those visits -- and invalidated by
    /// <see cref="SaveProfileAsync"/> so an edit is never served stale.
    /// </summary>
    public async Task<RiskProfile> GetProfileAsync()
    {
        var userId = await _currentUser.GetUserIdAsync();
        if (userId is { } id && _cache.TryGetValue(ProfileCacheKey(id), out RiskProfile? cached) && cached is not null)
        {
            return cached;
        }

        using var response = await SendAsync(() => _httpClient.GetAsync("/api/profile"));

        var result = await response.Content.ReadFromJsonAsync<RiskProfile>() ?? new RiskProfile();
        if (userId is { } cacheableId)
        {
            _cache.Set(ProfileCacheKey(cacheableId), result, ProfileCacheDuration);
        }

        return result;
    }

    private static string ProfileCacheKey(Guid userId) => $"profile:{userId}";

    /// <summary>Maps to PUT /api/profile.</summary>
    public async Task<RiskProfile> SaveProfileAsync(RiskProfile profile)
    {
        var fullName = RequireText(profile.FullName, "Name", InputLimits.FullNameMax);
        var location = RequireText(profile.Location, "Location", InputLimits.LocationMax);

        using var response = await SendAsync(() =>
            _httpClient.PutAsJsonAsync("/api/profile", new
            {
                fullName,
                location,
                sunExposure = profile.SunExposure,
                fitzpatrick = profile.Fitzpatrick,
                familyHistory = profile.FamilyHistory,
                blisteringSunburns = profile.BlisteringSunburns,
                manyMoles = profile.ManyMoles,
                recheckReminders = profile.RecheckReminders,
                highRiskAlerts = profile.HighRiskAlerts,
                shareWithDermatologist = profile.ShareWithDermatologist,
                anonymousAnalytics = profile.AnonymousAnalytics,
            }));

        var result = await response.Content.ReadFromJsonAsync<RiskProfile>();
        if (result is null)
        {
            throw new ImageProcessingApiException("The analysis service returned an empty response.");
        }

        if (await _currentUser.GetUserIdAsync() is { } userId)
        {
            _cache.Set(ProfileCacheKey(userId), result, ProfileCacheDuration);
        }

        return result;
    }

    /// <summary>
    /// Maps to GET /api/account/export -- every spot, check and the risk profile
    /// for one account, as the raw JSON document the API produced. Takes the
    /// account id explicitly because it is called from an endpoint, where there
    /// is no component authentication state to read it from.
    /// </summary>
    public async Task<JsonElement> ExportUserDataAsync(Guid userId)
    {
        using var response = await SendAsync(() => _httpClient.GetAsync("/api/account/export"), userId);

        var document = await response.Content.ReadFromJsonAsync<JsonElement>();
        return document.ValueKind == JsonValueKind.Undefined
            ? throw new ImageProcessingApiException("The analysis service returned an empty response.")
            : document;
    }

    /// <summary>Maps to DELETE /api/account -- erases everything the API holds for one account.</summary>
    public async Task DeleteUserDataAsync(Guid userId)
    {
        using var response = await SendAsync(() => _httpClient.DeleteAsync("/api/account"), userId);
        _cache.Remove(ProfileCacheKey(userId));
    }

    /// <summary>
    /// Trimmed text, or null when blank. Throws (with a message fit to show) for a
    /// value that is too long, has control characters, or is blank when required.
    /// </summary>
    private static string? RequireText(string? value, string field, int maxLength, bool required = false, bool multiline = false)
    {
        var text = value?.Trim();

        if (string.IsNullOrEmpty(text))
        {
            return required ? throw new ImageProcessingApiException($"{field} is required.") : null;
        }

        if (text.Length > maxLength)
        {
            throw new ImageProcessingApiException($"{field} must be {maxLength} characters or fewer.");
        }

        var allowed = multiline ? (Func<char, bool>)(c => c is '\n' or '\r' or '\t') : c => c == '\t';
        if (text.Any(c => char.IsControl(c) && !allowed(c)))
        {
            throw new ImageProcessingApiException($"{field} contains characters that aren't allowed.");
        }

        return text;
    }

    private static List<string> RequireSymptoms(IEnumerable<string>? symptoms)
    {
        var list = (symptoms ?? Enumerable.Empty<string>()).ToList();
        if (list.Count > InputLimits.SymptomsMaxCount)
        {
            throw new ImageProcessingApiException($"Please choose at most {InputLimits.SymptomsMaxCount} symptoms.");
        }

        return list
            .Select(symptom => RequireText(symptom, "Each symptom", InputLimits.SymptomMax))
            .OfType<string>()
            .ToList();
    }

    /// <summary>Counts one use of a costly action against the signed-in person's cap, or throws asking them to wait.</summary>
    private async Task ThrottleAsync(LimitedOperation operation)
    {
        var userId = await _currentUser.GetUserIdAsync()
            ?? throw new ImageProcessingApiException("Your session has ended. Sign in again to continue.");

        if (_limiter.TryAcquire(operation, userId) is { } retryAfter)
        {
            throw new ImageProcessingApiException(OperationRateLimiter.WaitMessage(retryAfter));
        }
    }

    /// <summary>
    /// Sends a request and converts every failure mode (unreachable server, the
    /// HttpClient's configured timeout, or a non-2xx response) into a single
    /// <see cref="ImageProcessingApiException"/> carrying a user-friendly message.
    /// The signed-in account id is attached first; pass <paramref name="userId"/>
    /// only from code that runs outside a component, where it has to be supplied.
    /// </summary>
    private async Task<HttpResponseMessage> SendAsync(Func<Task<HttpResponseMessage>> send, Guid? userId = null)
    {
        await AttachUserAsync(userId);

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

    /// <summary>
    /// Stamp the account id onto this client. The HttpClient instance is ours
    /// alone (IHttpClientFactory hands each typed-client instance its own), so
    /// a default header is safe and set once per instance.
    /// </summary>
    private async ValueTask AttachUserAsync(Guid? explicitUserId)
    {
        if (_httpClient.DefaultRequestHeaders.Contains(UserIdHeader))
        {
            return;
        }

        var userId = explicitUserId ?? await _currentUser.GetUserIdAsync()
            ?? throw new ImageProcessingApiException("Your session has ended. Sign in again to continue.");

        _httpClient.DefaultRequestHeaders.Add(UserIdHeader, userId.ToString());
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
