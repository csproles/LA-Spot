using MelanomaDetection.Web.Data;
using MelanomaDetection.Web.Services.Scheduling;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;

namespace MelanomaDetection.Web.Tests.Scheduling;

public class NotificationServiceTests
{
    private static async Task<(NotificationService Notifications, Guid ProviderId, Guid PatientId, Guid AppointmentId, string DbPath)> SeedAsync(
        DateTime startUtc)
    {
        var dbPath = Path.Combine(Path.GetTempPath(), $"telehealth-notify-{Guid.NewGuid():N}.db");
        var options = new DbContextOptionsBuilder<AppDbContext>().UseSqlite($"Data Source={dbPath}").Options;

        var providerId = Guid.NewGuid();
        var patientId = Guid.NewGuid();
        var appointmentId = Guid.NewGuid();

        await using (var seed = new AppDbContext(options))
        {
            await seed.Database.EnsureCreatedAsync();
            seed.Users.Add(new AppUser { Id = providerId, GoogleSubject = "p", Email = "dr@example.com", DisplayName = "Dr. Example" });
            seed.Users.Add(new AppUser { Id = patientId, GoogleSubject = "q", Email = "pat@example.com", DisplayName = "Pat Patient" });
            seed.Providers.Add(new Provider { Id = providerId, TimeZoneId = "UTC" });
            seed.Appointments.Add(new Appointment
            {
                Id = appointmentId,
                ProviderId = providerId,
                PatientId = patientId,
                StartUtc = startUtc,
                EndUtc = startUtc.AddMinutes(30),
                Reason = "itchy spot",
                CreatedAtUtc = DateTime.UtcNow,
            });
            await seed.SaveChangesAsync();
        }

        var notifications = new NotificationService(new SingleOptionsDbContextFactory(options));
        return (notifications, providerId, patientId, appointmentId, dbPath);
    }

    private static void Cleanup(string dbPath)
    {
        SqliteConnection.ClearAllPools();
        File.Delete(dbPath);
    }

    [Fact]
    public async Task NotifyBookedCreatesOneNotificationForEachPartyWithAMatchingIcs()
    {
        var (notifications, providerId, patientId, appointmentId, dbPath) = await SeedAsync(DateTime.UtcNow.AddDays(1));
        try
        {
            await notifications.NotifyBookedAsync(appointmentId);

            var patientNotice = Assert.Single(await notifications.ListForUserAsync(patientId));
            var providerNotice = Assert.Single(await notifications.ListForUserAsync(providerId));

            Assert.Equal(NotificationKind.BookingConfirmed, patientNotice.Kind);
            Assert.Equal(NotificationKind.BookingConfirmed, providerNotice.Kind);
            Assert.NotNull(patientNotice.IcsContent);
            Assert.Contains("STATUS:CONFIRMED", patientNotice.IcsContent);
            Assert.Contains("Dr. Example", patientNotice.Body);
            Assert.Contains("Pat Patient", providerNotice.Body);
        }
        finally
        {
            Cleanup(dbPath);
        }
    }

    [Fact]
    public async Task NotifyCancelledMarksTheIcsCancelled()
    {
        var (notifications, providerId, _, appointmentId, dbPath) = await SeedAsync(DateTime.UtcNow.AddDays(1));
        try
        {
            await notifications.NotifyCancelledAsync(appointmentId, "patient");

            var providerNotice = Assert.Single(await notifications.ListForUserAsync(providerId));
            Assert.Equal(NotificationKind.Cancelled, providerNotice.Kind);
            Assert.Contains("STATUS:CANCELLED", providerNotice.IcsContent);
            Assert.Contains("cancelled by the patient", providerNotice.Body);
        }
        finally
        {
            Cleanup(dbPath);
        }
    }

    [Fact]
    public async Task SendRemindersOnlyNotifiesAppointmentsInsideTheWindow()
    {
        var (notifications, providerId, _, appointmentId, dbPath) = await SeedAsync(DateTime.UtcNow.AddHours(24));
        try
        {
            // Window that excludes it.
            await notifications.SendRemindersAsync(NotificationKind.Reminder24h, DateTime.UtcNow.AddHours(1), DateTime.UtcNow.AddHours(2));
            Assert.Empty(await notifications.ListForUserAsync(providerId));

            // Window that includes it.
            await notifications.SendRemindersAsync(NotificationKind.Reminder24h, DateTime.UtcNow.AddHours(23), DateTime.UtcNow.AddHours(25));
            var notice = Assert.Single(await notifications.ListForUserAsync(providerId));
            Assert.Equal(NotificationKind.Reminder24h, notice.Kind);
        }
        finally
        {
            Cleanup(dbPath);
        }
    }

    [Fact]
    public async Task SendRemindersIsIdempotentAcrossOverlappingSweepPasses()
    {
        var (notifications, providerId, patientId, appointmentId, dbPath) = await SeedAsync(DateTime.UtcNow.AddHours(1));
        try
        {
            var window = (Start: DateTime.UtcNow.AddMinutes(50), End: DateTime.UtcNow.AddMinutes(70));

            // Two sweep passes covering the same appointment -- as would happen if
            // the sweep interval is shorter than the window width.
            await notifications.SendRemindersAsync(NotificationKind.Reminder1h, window.Start, window.End);
            await notifications.SendRemindersAsync(NotificationKind.Reminder1h, window.Start, window.End);

            Assert.Single(await notifications.ListForUserAsync(providerId));
            Assert.Single(await notifications.ListForUserAsync(patientId));
        }
        finally
        {
            Cleanup(dbPath);
        }
    }
}
