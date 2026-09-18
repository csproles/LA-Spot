using System.Security.Claims;
using MelanomaDetection.Web.Data;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Authentication.Google;
using Microsoft.AspNetCore.DataProtection;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;

namespace MelanomaDetection.Web.Services.Account;

/// <summary>
/// Registers everything sign-in needs: the accounts database, the cookie that
/// carries a session, and Google as the (only) way to obtain one.
/// </summary>
public static class AuthenticationSetup
{
    public const string ConnectionStringName = "SkinCheck";
    public const string GoogleClientIdKey = "Authentication:Google:ClientId";
    public const string GoogleClientSecretKey = "Authentication:Google:ClientSecret";
    public const string DataProtectionKeysDirectoryKey = "DataProtection:KeysDirectory";

    public static IServiceCollection AddSkinCheckAccounts(
        this IServiceCollection services, IConfiguration configuration, IHostEnvironment environment)
    {
        var connectionString = configuration.GetConnectionString(ConnectionStringName)
            ?? "Data Source=skincheck-users.db";
        EnsureDatabaseDirectoryExists(connectionString);

        // A factory, not AddDbContext: Blazor Server scopes are per circuit, so a
        // scoped DbContext would otherwise live for the whole browser session.
        services.AddDbContextFactory<AppDbContext>(options => options.UseSqlite(connectionString));

        // Sign-in cookies are encrypted with data-protection keys. In a container
        // those default to an ephemeral in-memory ring, which would sign everyone
        // out on every restart; pointing them at the data volume keeps them stable.
        var keysDirectory = configuration[DataProtectionKeysDirectoryKey];
        if (!string.IsNullOrWhiteSpace(keysDirectory))
        {
            Directory.CreateDirectory(keysDirectory);
            services.AddDataProtection()
                .SetApplicationName("SkinCheck")
                .PersistKeysToFileSystem(new DirectoryInfo(keysDirectory));
        }

        services.AddMemoryCache();
        services.AddScoped<UserAccountService>();
        services.AddScoped<CurrentUser>();
        services.AddCascadingAuthenticationState();
        services.AddAuthorization();

        // "Try the demo": on by default only in Development (see DemoSettings).
        services.Configure<DemoSettings>(options => options.Enabled = environment.IsDevelopment());
        services.Configure<DemoSettings>(configuration.GetSection(DemoSettings.SectionName));
        services.AddHostedService<DemoAccountSweeper>();

        var authentication = services
            .AddAuthentication(CookieAuthenticationDefaults.AuthenticationScheme)
            .AddCookie(options => ConfigureCookie(options, environment));

        var clientId = configuration[GoogleClientIdKey];
        var clientSecret = configuration[GoogleClientSecretKey];
        var googleConfigured = !string.IsNullOrWhiteSpace(clientId) && !string.IsNullOrWhiteSpace(clientSecret);

        if (googleConfigured)
        {
            authentication.AddGoogle(options => ConfigureGoogle(options, clientId!, clientSecret!));
        }
        else if (!environment.IsDevelopment())
        {
            // Outside development an app nobody can sign in to is a deployment
            // mistake; fail at startup with the exact keys to set.
            throw new InvalidOperationException(
                $"Google sign-in is not configured. Set '{GoogleClientIdKey}' and '{GoogleClientSecretKey}'.");
        }

        return services;
    }

    /// <summary>
    /// Bring the accounts database up to date. SQLite runs single-instance
    /// here, which is the case Microsoft's guidance says migrate-on-startup
    /// is appropriate for.
    /// </summary>
    public static async Task MigrateAccountsDatabaseAsync(this WebApplication app)
    {
        var factory = app.Services.GetRequiredService<IDbContextFactory<AppDbContext>>();
        await using var db = await factory.CreateDbContextAsync();
        await db.Database.MigrateAsync();
    }

    private static void ConfigureCookie(CookieAuthenticationOptions options, IHostEnvironment environment)
    {
        options.Cookie.Name = "SkinCheck.Auth";
        options.Cookie.HttpOnly = true;
        options.Cookie.SameSite = SameSiteMode.Lax;
        // The Docker image serves plain HTTP behind localhost:7001, so
        // Secure-only cookies would never come back there. Everywhere else the
        // cookie is HTTPS-only.
        options.Cookie.SecurePolicy = environment.IsDevelopment()
            ? CookieSecurePolicy.SameAsRequest
            : CookieSecurePolicy.Always;
        // Strictly necessary for the service the person asked for, so it is
        // exempt from consent under the ePrivacy rules and no banner is needed.
        options.Cookie.IsEssential = true;

        options.LoginPath = "/login";
        options.LogoutPath = "/auth/logout";
        options.AccessDeniedPath = "/login";
        options.ExpireTimeSpan = TimeSpan.FromDays(14);
        options.SlidingExpiration = true;

        // Every endpoint here is reached by a browser navigation or form post,
        // never by script, so an expired session should always land on the
        // sign-in page. Without this, .NET 10 answers minimal-API endpoints
        // (the export download, the delete form) with a bare 401 instead.
        options.Events.OnRedirectToLogin = context =>
        {
            context.Response.Redirect(context.RedirectUri);
            return Task.CompletedTask;
        };
        options.Events.OnRedirectToAccessDenied = context =>
        {
            context.Response.Redirect(context.RedirectUri);
            return Task.CompletedTask;
        };

        // A cookie outlives the account it was issued for (deleted from another
        // device, or by us). Re-check the row on each request so it stops
        // working everywhere at once.
        options.Events.OnValidatePrincipal = async context =>
        {
            var accounts = context.HttpContext.RequestServices.GetRequiredService<UserAccountService>();
            var userId = CurrentUser.ReadUserId(context.Principal!);

            if (userId is null || !await accounts.ExistsAsync(userId.Value, context.HttpContext.RequestAborted))
            {
                context.RejectPrincipal();
                await context.HttpContext.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
            }
        };
    }

    private static void ConfigureGoogle(GoogleOptions options, string clientId, string clientSecret)
    {
        options.ClientId = clientId;
        options.ClientSecret = clientSecret;

        // Data minimisation: the default scopes are openid, profile and email --
        // exactly what an account needs and nothing more -- and Google's access
        // and refresh tokens are not kept, because we never call Google again
        // on the person's behalf.
        options.SaveTokens = false;
        options.ClaimActions.MapJsonKey(AppClaimTypes.Picture, "picture");

        // Google returns to us with a top-level GET, which Lax cookies are sent
        // on. The framework's default of SameSite=None exists for form-post
        // callbacks and, without HTTPS, Chrome would drop that cookie entirely
        // and every sign-in would fail with "Correlation failed".
        options.CorrelationCookie.SameSite = SameSiteMode.Lax;

        options.Events.OnTicketReceived = async context =>
        {
            var accounts = context.HttpContext.RequestServices.GetRequiredService<UserAccountService>();
            var user = await accounts.SignInWithGoogleAsync(context.Principal!, context.HttpContext.RequestAborted);

            // Issue our own minimal identity rather than persisting the whole
            // Google ticket: the cookie carries only what the UI needs to show
            // and the key every data request is scoped by.
            context.Principal = CreatePrincipal(user);
        };
    }

    /// <summary>
    /// The principal a session cookie carries for an account: our id, what the
    /// UI shows, and whether it is a demo. Shared by Google sign-in and the demo
    /// entry point so the two can never drift.
    /// </summary>
    public static ClaimsPrincipal CreatePrincipal(AppUser user)
    {
        var identity = new ClaimsIdentity(
            [
                new Claim(AppClaimTypes.UserId, user.Id.ToString()),
                new Claim(ClaimTypes.Name, user.DisplayName),
                new Claim(ClaimTypes.Email, user.Email),
            ],
            CookieAuthenticationDefaults.AuthenticationScheme,
            ClaimTypes.Name,
            ClaimTypes.Role);

        if (user.PictureUrl is not null)
        {
            identity.AddClaim(new Claim(AppClaimTypes.Picture, user.PictureUrl));
        }

        if (user.IsDemo)
        {
            identity.AddClaim(new Claim(AppClaimTypes.Demo, "true"));
        }

        return new ClaimsPrincipal(identity);
    }

    private static void EnsureDatabaseDirectoryExists(string connectionString)
    {
        var dataSource = new SqliteConnectionStringBuilder(connectionString).DataSource;
        var directory = Path.GetDirectoryName(Path.GetFullPath(dataSource));
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }
    }
}
