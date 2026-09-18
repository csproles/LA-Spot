namespace MelanomaDetection.Web.Services.Account;

/// <summary>
/// Guards every post-sign-in / post-sign-out destination against open
/// redirects: only a path on this site is honoured, anything else falls back
/// to the app's home page.
/// </summary>
public static class LocalRedirect
{
    public const string Home = "/";

    public static string Sanitize(string? returnUrl, string fallback = Home)
    {
        return IsLocalUrl(returnUrl) ? returnUrl! : fallback;
    }

    /// <summary>
    /// Same rule as ASP.NET Core MVC's Url.IsLocalUrl: a single leading slash
    /// (so "//evil.example" and "/\evil.example" are rejected), no scheme, and
    /// no control characters that could smuggle a header break.
    /// </summary>
    public static bool IsLocalUrl(string? url)
    {
        if (string.IsNullOrEmpty(url) || url[0] != '/')
        {
            return false;
        }

        if (url.Length == 1)
        {
            return true;
        }

        if (url[1] == '/' || url[1] == '\\')
        {
            return false;
        }

        return !url.Any(char.IsControl);
    }
}
