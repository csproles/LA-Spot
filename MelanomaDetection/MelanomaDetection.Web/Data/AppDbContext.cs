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

    public DbSet<Provider> Providers => Set<Provider>();

    public DbSet<AvailabilityRule> AvailabilityRules => Set<AvailabilityRule>();

    public DbSet<AvailabilityException> AvailabilityExceptions => Set<AvailabilityException>();

    public DbSet<Appointment> Appointments => Set<Appointment>();

    public DbSet<Notification> Notifications => Set<Notification>();

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

        modelBuilder.Entity<Provider>(provider =>
        {
            provider.HasKey(p => p.Id);
            provider.Property(p => p.Specialty).HasMaxLength(100);
            provider.Property(p => p.TimeZoneId).HasMaxLength(100);
            provider.Property(p => p.LicenseNumber).HasMaxLength(50);
            provider.Property(p => p.Credentials).HasMaxLength(200);
            provider.Property(p => p.Bio).HasMaxLength(2000);
            provider.Property(p => p.PhotoUrl).HasMaxLength(2048);
        });

        modelBuilder.Entity<AvailabilityRule>(rule =>
        {
            rule.HasIndex(r => new { r.ProviderId, r.Weekday });
        });

        modelBuilder.Entity<AvailabilityException>(exception =>
        {
            exception.HasIndex(e => new { e.ProviderId, e.Date });
        });

        modelBuilder.Entity<Appointment>(appointment =>
        {
            appointment.Property(a => a.MeetingId).HasMaxLength(100);
            appointment.Property(a => a.MeetingUrl).HasMaxLength(2048);
            appointment.Property(a => a.CancelledBy).HasMaxLength(20);
            appointment.Property(a => a.ScanProcessingId).HasMaxLength(100);
            appointment.Property(a => a.ScanOverallVisualConcern).HasMaxLength(50);
            appointment.HasIndex(a => a.PatientId);
            // The double-booking guard: SQLite supports a filtered (partial) unique
            // index, so two non-cancelled rows can never share a provider+start time,
            // even under a concurrent insert race -- the second one throws at the DB,
            // not just fails a pre-insert availability check.
            appointment.HasIndex(a => new { a.ProviderId, a.StartUtc })
                .IsUnique()
                .HasFilter($"\"Status\" <> {(int)AppointmentStatus.Cancelled}");
        });

        modelBuilder.Entity<Notification>(notification =>
        {
            notification.Property(n => n.Subject).HasMaxLength(200);
            notification.HasIndex(n => new { n.RecipientUserId, n.CreatedAtUtc });
            // The double-send guard: a sweep or a retry can run twice, but the
            // same notification kind can never reach the same recipient for the
            // same appointment more than once -- enforced here, not just by the
            // sender checking first.
            notification.HasIndex(n => new { n.AppointmentId, n.RecipientUserId, n.Kind }).IsUnique();
        });
    }
}
