using Microsoft.EntityFrameworkCore;

namespace MelanomaDetection.Web.Data;

/// <summary>
/// The web app's own SQLite database: accounts only. Resolved through
/// <see cref="IDbContextFactory{TContext}"/> rather than injected directly, as
/// Microsoft recommends for Blazor Server, because a scoped context would
/// otherwise live as long as the user's circuit.
/// </summary>
public class AppDbContext(DbContextOptions<AppDbContext> options) : DbContext(options)
{
    public DbSet<AppUser> Users => Set<AppUser>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<AppUser>(user =>
        {
            user.HasKey(u => u.Id);
            user.HasIndex(u => u.GoogleSubject).IsUnique();
            user.Property(u => u.GoogleSubject).HasMaxLength(255);
            user.Property(u => u.Email).HasMaxLength(320);
            user.Property(u => u.DisplayName).HasMaxLength(200);
            user.Property(u => u.PictureUrl).HasMaxLength(2048);
        });
    }
}
