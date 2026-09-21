using System.Text.Json.Serialization;

namespace MelanomaDetection.Web.Models;

public enum ChatRole
{
    User,
    Assistant,
}

/// <summary>One turn in the chat widget's conversation, held only in the Blazor circuit's memory.</summary>
public class ChatMessage
{
    public ChatRole Role { get; set; }
    public string Text { get; set; } = string.Empty;
    public bool Refused { get; set; }
    public IReadOnlyList<string> Sources { get; set; } = Array.Empty<string>();
}

/// <summary>The wire shape POST /api/chat expects for one prior turn.</summary>
public class ChatHistoryTurn
{
    [JsonPropertyName("role")]
    public string Role { get; set; } = "user";

    [JsonPropertyName("text")]
    public string Text { get; set; } = string.Empty;
}

/// <summary>The wire shape POST /api/chat expects for the caller's current-screen summary.</summary>
public class ChatPageContextPayload
{
    [JsonPropertyName("page")]
    public string Page { get; set; } = string.Empty;

    [JsonPropertyName("data")]
    public IReadOnlyDictionary<string, object?> Data { get; set; } = new Dictionary<string, object?>();
}

/// <summary>POST /api/chat's response shape (textbook_chat.render_answer in the Flask API).</summary>
public class ChatAnswerResponse
{
    [JsonPropertyName("refused")]
    public bool Refused { get; set; }

    [JsonPropertyName("answer")]
    public string? Answer { get; set; }

    [JsonPropertyName("refusalReason")]
    public string? RefusalReason { get; set; }

    [JsonPropertyName("sources")]
    public List<string> Sources { get; set; } = new();
}
