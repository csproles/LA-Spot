using MelanomaDetection.Web.Data;
using MelanomaDetection.Web.Services.Scheduling;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging.Abstractions;

namespace MelanomaDetection.Web.Tests.Scheduling;

/// <summary>The video room's access control: only the patient or provider on a visit may open it or mark it complete.</summary>
public class VideoVisitAuthorizationTests
{
    private static async Task<(BookingService Booking, Guid ProviderId, Guid PatientId, Guid AppointmentId, string DbPath)> SeedAsync()
    {
        var dbPath = Path.Combine(Path.GetTempPath(), $"telehealth-video-{Guid.NewGuid():N}.db");
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
                StartUtc = DateTime.UtcNow,
                EndUtc = DateTime.UtcNow.AddMinutes(30),
                CreatedAtUtc = DateTime.UtcNow,
            });
            await seed.SaveChangesAsync();
        }

        var factory = new SingleOptionsDbContextFactory(options);
        var booking = new BookingService(factory, new AvailabilityService(factory), new NotificationService(factory), new JitsiVideoRoomProvider(), NullLogger<BookingService>.Instance);
        return (booking, providerId, patientId, appointmentId, dbPath);
    }

    [Fact]
    public async Task FindForUserReturnsTheVisitForEitherParty()
    {
        var (booking, providerId, patientId, appointmentId, dbPath) = await SeedAsync();
        try
        {
            var asProvider = await booking.FindForUserAsync(appointmentId, providerId);
            var asPatient = await booking.FindForUserAsync(appointmentId, patientId);

            Assert.NotNull(asProvider);
            Assert.NotNull(asPatient);
            Assert.Equal(appointmentId, asProvider!.Id);
        }
        finally
        {
            SqliteConnection.ClearAllPools();
            File.Delete(dbPath);
        }
    }

    [Fact]
    public async Task FindForUserRejectsAStranger()
    {
        var (booking, _, _, appointmentId, dbPath) = await SeedAsync();
        try
        {
            await Assert.ThrowsAsync<UnauthorizedAccessException>(() => booking.FindForUserAsync(appointmentId, Guid.NewGuid()));
        }
        finally
        {
            SqliteConnection.ClearAllPools();
            File.Delete(dbPath);
        }
    }

    [Fact]
    public async Task FindForUserReturnsNullForAnUnknownAppointment()
    {
        var (booking, providerId, _, _, dbPath) = await SeedAsync();
        try
        {
            Assert.Null(await booking.FindForUserAsync(Guid.NewGuid(), providerId));
        }
        finally
        {
            SqliteConnection.ClearAllPools();
            File.Delete(dbPath);
        }
    }

    [Fact]
    public async Task OnlyTheProviderCanCompleteAVisit()
    {
        var (booking, providerId, patientId, appointmentId, dbPath) = await SeedAsync();
        try
        {
            await Assert.ThrowsAsync<UnauthorizedAccessException>(() => booking.CompleteAsync(appointmentId, patientId));

            await booking.CompleteAsync(appointmentId, providerId);
            var view = await booking.FindForUserAsync(appointmentId, providerId);
            Assert.Equal(AppointmentStatus.Completed, view!.Status);
        }
        finally
        {
            SqliteConnection.ClearAllPools();
            File.Delete(dbPath);
        }
    }
}
