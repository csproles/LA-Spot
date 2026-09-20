using MelanomaDetection.Web.Components;
using MelanomaDetection.Web.Services;
using MelanomaDetection.Web.Services.Account;
using MelanomaDetection.Web.Services.RateLimiting;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

// Accounts database, session cookie and Google sign-in.
builder.Services.AddSkinCheckAccounts(builder.Configuration, builder.Environment);

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

// NPI Registry + Census geocoder (both free, keyless government APIs) for the Map page's
// "nearby dermatologists" lookup.
builder.Services.AddMemoryCache();
builder.Services.AddHttpClient<NpiProviderService>(client =>
{
    client.Timeout = TimeSpan.FromSeconds(15);
});

var app = builder.Build();

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
