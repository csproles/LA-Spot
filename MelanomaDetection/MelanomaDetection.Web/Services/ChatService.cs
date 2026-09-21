using System.Net.Http.Json;
using MelanomaDetection.Web.Models;
using MelanomaDetection.Web.Services.Account;
using MelanomaDetection.Web.Services.Chat;
using MelanomaDetection.Web.Services.RateLimiting;

namespace MelanomaDetection.Web.Services;

/// <summary>
/// Thrown for any failure talking to the chat API (network, timeout, or an error
/// response). <see cref="Exception.Message"/> is always safe to show directly to the user.
/// </summary>
public class ChatApiException : Exception
{
    public ChatApiException(string message) : base(message)
    {
    }
}

/// <summary>
/// Talks to POST /api/chat (see MelanomaDetection.Python/textbook_chat.py) -- the
/// textbook Q&amp;A chat widget shown on every page. Same identity and error-handling
/// shape as <see cref="ImageProcessingService"/> (a second, small typed client rather
/// than sharing that class, since the two talk to unrelated endpoints); the request/
/// response plumbing (SendAsync/AttachUserAsync) is intentionally duplicated in
/// miniature here rather than factored out, since this is the only other caller --
/// worth revisiting if a third Flask-calling service shows up.
/// </summary>
public class ChatService
{
    private readonly HttpClient _httpClient;
    private readonly CurrentUser _currentUser;
    private readonly OperationRateLimiter _limiter;
    private readonly ChatPageContext _pageContext;

    public ChatService(HttpClient httpClient, CurrentUser currentUser, OperationRateLimiter limiter, ChatPageContext pageContext)
    {
        _httpClient = httpClient;
        _currentUser = currentUser;
        _limiter = limiter;
        _pageContext = pageContext;
    }

    /// <summary>
    /// Asks the textbook chat one question, with up to the last few prior turns for
    /// conversational context and the current page's already-loaded data (see
    /// <see cref="ChatPageContext"/>) attached automatically. Only call in response to
    /// an explicit user action -- each call costs a real OpenAI request on the Flask side.
    /// </summary>
    public async Task<ChatAnswerResponse> AskAsync(string question, IReadOnlyList<ChatMessage> history)
    {
        question = RequireText(question, "Question", InputLimits.ChatQuestionMax, required: true)!;
        await ThrottleAsync();

        var historyPayload = history
            .TakeLast(6)
            .Select(m => new ChatHistoryTurn { Role = m.Role == ChatRole.User ? "user" : "assistant", Text = m.Text })
            .ToList();

        ChatPageContextPayload? pageContextPayload = _pageContext.Data is { Count: > 0 } data
            ? new ChatPageContextPayload { Page = _pageContext.PageName ?? "unknown", Data = data }
            : null;

        using var response = await SendAsync(() =>
            _httpClient.PostAsJsonAsync("/api/chat", new { question, history = historyPayload, pageContext = pageContextPayload }));

        var result = await response.Content.ReadFromJsonAsync<ChatAnswerResponse>();
        return result ?? throw new ChatApiException("The chat service returned an empty response.");
    }

    private static string? RequireText(string? value, string field, int maxLength, bool required = false)
    {
        var text = value?.Trim();

        if (string.IsNullOrEmpty(text))
        {
            return required ? throw new ChatApiException($"{field} is required.") : null;
        }

        if (text.Length > maxLength)
        {
            throw new ChatApiException($"{field} must be {maxLength} characters or fewer.");
        }

        return text;
    }

    private async Task ThrottleAsync()
    {
        var userId = await _currentUser.GetUserIdAsync()
            ?? throw new ChatApiException("Your session has ended. Sign in again to continue.");

        if (_limiter.TryAcquire(LimitedOperation.Chat, userId) is { } retryAfter)
        {
            throw new ChatApiException(OperationRateLimiter.WaitMessage(retryAfter));
        }
    }

    private async Task<HttpResponseMessage> SendAsync(Func<Task<HttpResponseMessage>> send)
    {
        await AttachUserAsync();

        HttpResponseMessage response;
        try
        {
            response = await send();
        }
        catch (TaskCanceledException)
        {
            // We never pass our own CancellationToken, so this can only be HttpClient.Timeout firing.
            throw new ChatApiException("The chat service took too long to respond (30s timeout). Please try again.");
        }
        catch (HttpRequestException)
        {
            throw new ChatApiException("Could not reach the chat service. Make sure the backend is running and try again.");
        }

        if (!response.IsSuccessStatusCode)
        {
            var message = await TryReadErrorMessageAsync(response);
            response.Dispose();
            throw new ChatApiException(message ?? $"The chat service returned an unexpected error ({(int)response.StatusCode}).");
        }

        return response;
    }

    /// <summary>Stamp the account id onto this client once, the same way ImageProcessingService does.</summary>
    private async ValueTask AttachUserAsync()
    {
        if (_httpClient.DefaultRequestHeaders.Contains(ImageProcessingService.UserIdHeader))
        {
            return;
        }

        var userId = await _currentUser.GetUserIdAsync()
            ?? throw new ChatApiException("Your session has ended. Sign in again to continue.");

        _httpClient.DefaultRequestHeaders.Add(ImageProcessingService.UserIdHeader, userId.ToString());
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
}
