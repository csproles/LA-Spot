using MelanomaDetection.Web.Data;
using MelanomaDetection.Web.Services.Scheduling;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging.Abstractions;

namespace MelanomaDetection.Web.Tests.Scheduling;

/// <summary>Booking always stamps a video room on the appointment, and carries through
/// an attached scan snapshot when one is given -- see Appointment.ScanProcessingId
/// for why this is a snapshot rather than a live cross-service reference.</summary>
public class BookingServiceScanAndVideoRoomTests
{
    [Fact]
    public async Task BookingAttachesAVideoRoomAndAnOptionalScanSnapshot()
    {
        var dbPath = Path.Combine(Path.GetTempPath(), $"telehealth-scan-{Guid.NewGuid():N}.db");
        var options = new DbContextOptionsBuilder<AppDbContext>().UseSqlite($"Data Source={dbPath}").Options;

        try
        {
            var providerId = Guid.NewGuid();
            var patientId = Guid.NewGuid();
            var date = new DateOnly(2027, 6, 7);

            await using (var seed = new AppDbContext(options))
            {
                await seed.Database.EnsureCreatedAsync();
                seed.Users.Add(new AppUser { Id = providerId, GoogleSubject = "provider-sub", Email = "dr@example.com", DisplayName = "Dr. Example" });
                seed.Users.Add(new AppUser { Id = patientId, GoogleSubject = "patient-sub", Email = "patient@example.com", DisplayName = "Pat Patient" });
                seed.Providers.Add(new Provider
                {
                    Id = providerId,
                    TimeZoneId = "UTC",
                    AppointmentLengthMinutes = 30,
                    BufferMinutes = 0,
                });
                seed.AvailabilityRules.Add(new AvailabilityRule
                {
                    ProviderId = providerId,
                    Weekday = date.DayOfWeek,
                    StartTime = TimeOnly.Parse("09:00"),
                    EndTime = TimeOnly.Parse("10:00"),
                });
                await seed.SaveChangesAsync();
            }

            var factory = new SingleOptionsDbContextFactory(options);
            var booking = new BookingService(
                factory, new AvailabilityService(factory), new NotificationService(factory),
                new JitsiVideoRoomProvider(), NullLogger<BookingService>.Instance);
            var slotStartUtc = date.ToDateTime(TimeOnly.Parse("09:00"), DateTimeKind.Utc);

            var scan = new AttachedScan("proc-123", 82, "ELEVATED VISUAL CONCERN", "Looks concerning, please see a dermatologist.");
            var result = await booking.BookAsync(providerId, patientId, slotStartUtc, "itchy spot", scan);

            Assert.Equal(BookingOutcome.Booked, result.Outcome);
            var appointment = result.Appointment!;
            Assert.False(string.IsNullOrWhiteSpace(appointment.MeetingUrl));
            Assert.StartsWith("https://meet.jit.si/lionspot-", appointment.MeetingUrl);
            Assert.Equal("proc-123", appointment.ScanProcessingId);
            Assert.Equal(82, appointment.ScanRiskScore);
            Assert.Equal("ELEVATED VISUAL CONCERN", appointment.ScanOverallVisualConcern);

            var view = Assert.Single(await booking.ListForPatientAsync(patientId));
            Assert.Equal(appointment.MeetingUrl, view.MeetingUrl);
            Assert.Equal("proc-123", view.ScanProcessingId);
        }
        finally
        {
            SqliteConnection.ClearAllPools();
            File.Delete(dbPath);
        }
    }
}
