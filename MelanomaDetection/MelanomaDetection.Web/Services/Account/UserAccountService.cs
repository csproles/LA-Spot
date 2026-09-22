using System.Security.Claims;
using MelanomaDetection.Web.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Caching.Memory;

namespace MelanomaDetection.Web.Services.Account;

/// <summary>
/// Account rows in the web app's own database. Sign-in upserts one from the
/// Google ticket; every later request re-checks that it still exists so a
/// deleted account's cookies stop working everywhere, not just on the device
/// that deleted it.
/// </summary>
public sealed class UserAccountService(
    IDbContextFactory<AppDbContext> dbFactory,
    IMemoryCache cache,
    ILogger<UserAccountService> logger,
    IConfiguration configuration)
{
    /// <summary>Emails granted a provider account, comma-separated (Provider:AllowedEmails).
    /// Re-checked on every sign-in, not just creation, so adding/removing an email takes
    /// effect on that person's next login -- no manual DB edit needed.</summary>
    private bool IsAllowedProviderEmail(string email) =>
        (configuration["Provider:AllowedEmails"] ?? "")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Contains(email, StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// How long a "this account exists" answer is trusted before hitting the
    /// database again. Deleting an account evicts the entry immediately, so
    /// this only bounds staleness for deletions made by another process.
    /// </summary>
    private static readonly TimeSpan ExistsCacheTtl = TimeSpan.FromMinutes(1);

    /// <summary>
    /// Create or refresh the account for a Google principal and return it.
    /// Only the claims we store are read; nothing else from the ticket is kept.
    /// </summary>
    public async Task<AppUser> SignInWithGoogleAsync(ClaimsPrincipal principal, CancellationToken cancellationToken = default)
    {
        var subject = principal.FindFirstValue(ClaimTypes.NameIdentifier)
            ?? throw new InvalidOperationException("The Google ticket has no subject (sub) claim.");
        var email = principal.FindFirstValue(ClaimTypes.Email)
            ?? throw new InvalidOperationException("The Google ticket has no email claim.");
        var displayName = principal.FindFirstValue(ClaimTypes.Name) ?? email;
        var picture = principal.FindFirstValue(AppClaimTypes.Picture);
        var now = DateTime.UtcNow;

        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        var user = await db.Users.SingleOrDefaultAsync(u => u.GoogleSubject == subject, cancellationToken);

        if (user is null)
        {
            user = new AppUser
            {
                Id = Guid.NewGuid(),
                GoogleSubject = subject,
                Email = email,
                DisplayName = displayName,
                PictureUrl = picture,
                CreatedAtUtc = now,
                LastSignInAtUtc = now,
            };
            db.Users.Add(user);
            logger.LogInformation("Created account {UserId}", user.Id);
        }
        else
        {
            user.Email = email;
            user.DisplayName = displayName;
            user.PictureUrl = picture;
            user.LastSignInAtUtc = now;
        }

        user.IsProvider = IsAllowedProviderEmail(email);
        if (user.IsProvider && !await db.Providers.AnyAsync(p => p.Id == user.Id, cancellationToken))
        {
            db.Providers.Add(new Provider { Id = user.Id });
        }

        await db.SaveChangesAsync(cancellationToken);
        cache.Set(ExistsKey(user.Id), true, ExistsCacheTtl);
        return user;
    }

    /// <summary>
    /// A fresh, empty demo account. Nothing links it to a real person, and
    /// <see cref="DeleteAsync"/> is called when the session ends.
    /// </summary>
    public async Task<AppUser> CreateDemoAsync(CancellationToken cancellationToken = default)
    {
        var now = DateTime.UtcNow;
        var user = new AppUser
        {
            Id = Guid.NewGuid(),
            GoogleSubject = $"demo:{Guid.NewGuid():N}",
            Email = string.Empty,
            DisplayName = "Demo account",
            CreatedAtUtc = now,
            LastSignInAtUtc = now,
            IsDemo = true,
        };

        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        db.Users.Add(user);
        await db.SaveChangesAsync(cancellationToken);
        cache.Set(ExistsKey(user.Id), true, ExistsCacheTtl);
        logger.LogInformation("Created demo account {UserId}", user.Id);
        return user;
    }

    /// <summary>Demo accounts whose session was never ended, older than <paramref name="olderThan"/>.</summary>
    public async Task<List<Guid>> FindStaleDemoAccountsAsync(TimeSpan olderThan, CancellationToken cancellationToken = default)
    {
        var cutoff = DateTime.UtcNow - olderThan;
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        return await db.Users
            .Where(u => u.IsDemo && u.CreatedAtUtc < cutoff)
            .Select(u => u.Id)
            .ToListAsync(cancellationToken);
    }

    public async Task<bool> ExistsAsync(Guid userId, CancellationToken cancellationToken = default)
    {
        if (cache.TryGetValue(ExistsKey(userId), out bool cached))
        {
            return cached;
        }

        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        var exists = await db.Users.AnyAsync(u => u.Id == userId, cancellationToken);
        cache.Set(ExistsKey(userId), exists, ExistsCacheTtl);
        return exists;
    }

    public async Task<AppUser?> FindAsync(Guid userId, CancellationToken cancellationToken = default)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        return await db.Users.AsNoTracking().SingleOrDefaultAsync(u => u.Id == userId, cancellationToken);
    }

    /// <summary>
    /// Remove the account row. Callers delete the Flask-side data first, so a
    /// failure there leaves the account intact and the person can retry.
    /// </summary>
    public async Task DeleteAsync(Guid userId, CancellationToken cancellationToken = default)
    {
        await using var db = await dbFactory.CreateDbContextAsync(cancellationToken);
        await db.Users.Where(u => u.Id == userId).ExecuteDeleteAsync(cancellationToken);
        cache.Remove(ExistsKey(userId));
        logger.LogInformation("Deleted account {UserId}", userId);
    }

    private static string ExistsKey(Guid userId) => $"account-exists:{userId}";
}
