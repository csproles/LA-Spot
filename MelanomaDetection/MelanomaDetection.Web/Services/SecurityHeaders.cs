using System.Security.Cryptography;

namespace MelanomaDetection.Web.Services;

/// <summary>
/// Browser-side hardening headers, sent on every response.
///
/// The Content-Security-Policy is what limits the damage of any script that
/// slips into a page: only this site's own scripts run, and only the hosts the
/// app genuinely uses (Google Maps, Google Fonts, Google profile pictures) can
/// be loaded from. The camera is allowed for this site alone because the check
/// flow uses it; nothing else in the browser is.
/// </summary>
public static class SecurityHeaders
{
    /// <summary>
    /// Where the per-request script nonce is kept, for the one inline script the
    /// framework emits (the import map, see App.razor). Any other inline script is refused.
    /// </summary>
    public const string NonceKey = "csp-nonce";

    private const string GoogleMaps = "https://*.googleapis.com https://*.gstatic.com https://*.google.com https://*.ggpht.com https://*.googleusercontent.com";

    /// <summary>
    /// <c>ws:</c>/<c>wss:</c> are for the Blazor circuit's WebSocket; some browsers don't
    /// treat 'self' as covering it. Inline styles are needed by Blazor's scoped CSS and
    /// by Google Maps. Inline <em>scripts</em> are not allowed, except the nonce'd one.
    /// </summary>
    private static string BuildContentSecurityPolicy(string nonce) => string.Join("; ",
        "default-src 'self'",
        $"script-src 'self' 'nonce-{nonce}' {GoogleMaps} blob:",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        $"img-src 'self' data: blob: {GoogleMaps}",
        "font-src 'self' https://fonts.gstatic.com",
        $"connect-src 'self' ws: wss: {GoogleMaps} data: blob:",
        "worker-src 'self' blob:",
        "manifest-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self' https://accounts.google.com",
        "frame-ancestors 'self'");

    public static IApplicationBuilder UseSkinCheckSecurityHeaders(this IApplicationBuilder app)
    {
        return app.Use((context, next) =>
        {
            var nonce = Convert.ToBase64String(RandomNumberGenerator.GetBytes(16));
            context.Items[NonceKey] = nonce;

            // OnStarting so these are applied last -- the framework adds its own
            // frame-ancestors-only CSP to some responses and would otherwise win.
            context.Response.OnStarting(() =>
            {
                var headers = context.Response.Headers;
                headers.ContentSecurityPolicy = BuildContentSecurityPolicy(nonce);
                headers.XContentTypeOptions = "nosniff";
                headers["Referrer-Policy"] = "strict-origin-when-cross-origin";
                headers["Permissions-Policy"] = "camera=(self), microphone=(), geolocation=(), payment=()";
                return Task.CompletedTask;
            });

            return next();
        });
    }
}
