using System.Net;
using MelanomaDetection.Web.Components;
using MelanomaDetection.Web.Services;
using MelanomaDetection.Web.Services.Account;
using MelanomaDetection.Web.Services.Chat;
using MelanomaDetection.Web.Services.RateLimiting;
using Microsoft.AspNetCore.Antiforgery;
using Microsoft.AspNetCore.HttpOverrides;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

// Accounts database, session cookie and Google sign-in.
builder.Services.AddSkinCheckAccounts(builder.Configuration, builder.Environment);

// Behind a reverse proxy the real client address is only in X-Forwarded-For, and
// the rate limits are keyed on it (without this, everyone shares the proxy's
// address). That header is trusted only from the proxies named here, never from
// arbitrary callers, so it can't be used to dodge a limit. Leave both lists empty
// when nothing sits in front of the app.
var trustedProxies = builder.Configuration.GetSection("ReverseProxy:KnownProxies").Get<string[]>() ?? [];
var trustedNetworks = builder.Configuration.GetSection("ReverseProxy:KnownNetworks").Get<string[]>() ?? [];
var behindProxy = trustedProxies.Length + trustedNetworks.Length > 0;
if (behindProxy)
{
    builder.Services.Configure<ForwardedHeadersOptions>(options =>
    {
        options.ForwardedHeaders = ForwardedHeaders.XForwardedFor | ForwardedHeaders.XForwardedProto;
        options.KnownProxies.Clear();
        options.KnownIPNetworks.Clear();
        foreach (var proxy in trustedProxies)
        {
            options.KnownProxies.Add(IPAddress.Parse(proxy));
        }

        foreach (var network in trustedNetworks)
        {
            options.KnownIPNetworks.Add(System.Net.IPNetwork.Parse(network));
        }
    });
}

// Request rate limits (sign-in, account endpoints, a global ceiling) plus the
// per-person caps on expensive actions that happen over the Blazor circuit.
builder.Services.AddSkinCheckRateLimiting();

// Flask image-processing API (MelanomaDetection.Python/main.py), default port 5002.
// FlaskApi:InternalKey is the shared secret Flask uses to trust the X-User-Id
// header; required outside development because Flask's port is reachable on
// the host.
var flaskInternalKey = builder.Configuration["FlaskApi:InternalKey"];
if (string.IsNullOrWhiteSpace(flaskInternalKey) && !builder.Environment.IsDevelopment())
{
    throw new InvalidOperationException("'FlaskApi:InternalKey' must be set so the analysis service can trust this app.");
}

builder.Services.AddHttpClient<ImageProcessingService>(client =>
{
    var baseUrl = builder.Configuration["FlaskApi:BaseUrl"] ?? "http://localhost:5002";
    client.BaseAddress = new Uri(baseUrl);
    client.Timeout = TimeSpan.FromSeconds(30);
    if (!string.IsNullOrWhiteSpace(flaskInternalKey))
    {
        client.DefaultRequestHeaders.Add(ImageProcessingService.InternalKeyHeader, flaskInternalKey);
    }
});

// The chat widget's per-page "what's on screen" context (see ChatPageContext's summary);
// scoped like CurrentUser so it never crosses between people sharing the server.
builder.Services.AddScoped<ChatPageContext>();

builder.Services.AddHttpClient<ChatService>(client =>
{
    var baseUrl = builder.Configuration["FlaskApi:BaseUrl"] ?? "http://localhost:5002";
    client.BaseAddress = new Uri(baseUrl);
    client.Timeout = TimeSpan.FromSeconds(30);
    if (!string.IsNullOrWhiteSpace(flaskInternalKey))
    {
        client.DefaultRequestHeaders.Add(ImageProcessingService.InternalKeyHeader, flaskInternalKey);
    }
});

// NPI Registry + Census geocoder (both free, keyless government APIs) for the Map page's
// "nearby dermatologists" lookup.
builder.Services.AddMemoryCache();
builder.Services.AddHttpClient<NpiProviderService>(client =>
{
    client.Timeout = TimeSpan.FromSeconds(15);
});

var app = builder.Build();

if (behindProxy)
{
    // First, so everything after it sees the real client address and scheme.
    app.UseForwardedHeaders();
}

app.UseSkinCheckSecurityHeaders();

// A stale page -- almost always the browser's back/forward cache showing an
// old sign-in, sign-out, or delete-account form -- carries an antiforgery
// token bound to whoever was signed in when that page was rendered. If the
// session has since changed (signed in, signed out, the demo account got
// swept), posting that stale form throws here instead of silently acting
// under the wrong identity. Cache-Control: no-store (see SecurityHeaders)
// stops the browser from resurrecting the stale page in the first place;
// this is the fallback for whatever slips through that anyway (multiple
// tabs, multiple devices). Registered this early so it wraps every later
// middleware, including UseAntiforgery and the endpoints themselves.
app.Use(async (context, next) =>
{
    try
    {
        await next();
    }
    catch (Exception ex) when (ex is AntiforgeryValidationException || ex.InnerException is AntiforgeryValidationException)
    {
        context.Response.Redirect("/login?status=session-expired");
    }
});

await app.MigrateAccountsDatabaseAsync();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}
app.UseStatusCodePagesWithReExecute("/not-found", createScopeForStatusCodePages: true);
app.UseHttpsRedirection();

app.UseAuthentication();

// After authentication so limits can be per account rather than per address.
app.UseRateLimiter();

app.UseAuthorization();

// Antiforgery has to follow authentication: tokens are bound to the signed-in identity.
app.UseAntiforgery();

app.MapStaticAssets();
app.MapAccountEndpoints();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
