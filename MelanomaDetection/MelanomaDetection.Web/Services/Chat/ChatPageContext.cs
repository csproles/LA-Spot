namespace MelanomaDetection.Web.Services.Chat;

/// <summary>
/// What the currently-viewed page has already loaded, for the chat widget to
/// forward as pageContext on each question -- so "what does my score mean" can
/// be answered from the data the person is already looking at, without the
/// widget scraping the page itself.
///
/// Scoped per Blazor circuit, so it never crosses between users. A page that
/// sets it must clear it on dispose (implement IDisposable and call Clear() --
/// see Pages/Results.razor) so navigating to a page that never sets context
/// doesn't leave the previous page's data attached to new questions.
/// </summary>
public sealed class ChatPageContext
{
    public string? PageName { get; private set; }
    public IReadOnlyDictionary<string, object?>? Data { get; private set; }

    public void Set(string pageName, IReadOnlyDictionary<string, object?> data)
    {
        PageName = pageName;
        Data = data;
    }

    public void Clear()
    {
        PageName = null;
        Data = null;
    }
}
