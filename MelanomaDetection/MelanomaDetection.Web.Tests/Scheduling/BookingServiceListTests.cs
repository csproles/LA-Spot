using MelanomaDetection.Web.Data;
using MelanomaDetection.Web.Services.Scheduling;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging.Abstractions;

namespace MelanomaDetection.Web.Tests.Scheduling;

/// <summary>
/// ListForPatientAsync/ListForProviderAsync join Appointments against Users
/// twice to attach display names. A first version of that query projected
/// straight into the AppointmentView record inside the join, which built and
/// unit-tested fine (SlotGenerator has no DB in the loop) but threw
/// InvalidOperationException ("could not be translated") the first time it
/// ran against a real SQLite provider -- only caught by hitting the live
/// endpoint by hand. This exercises the real translated query so a
/// regression fails a test instead of a manual click-through.
/// </summary>
public class BookingServiceListTests
{
    [Fact]
    public async Task ListForPatientAndListForProviderReturnTheBookingWithBothDisplayNames()
    {
        var dbPath = Path.Combine(Path.GetTempPath(), $"telehealth-list-{Guid.NewGuid():N}.db");
        var options = new DbContextOptionsBuilder<AppDbContext>().UseSqlite($"Data Source={dbPath}").Options;

        try
        {
            var providerId = Guid.NewGuid();
            var patientId = Guid.NewGuid();
            var startUtc = new DateTime(2027, 6, 7, 9, 0, 0, DateTimeKind.Utc);

            await using (var seed = new AppDbContext(options))
            {
                await seed.Database.EnsureCreatedAsync();
                seed.Users.Add(new AppUser { Id = providerId, GoogleSubject = "provider-sub", Email = "dr@example.com", DisplayName = "Dr. Example" });
                seed.Users.Add(new AppUser { Id = patientId, GoogleSubject = "patient-sub", Email = "patient@example.com", DisplayName = "Pat Patient" });
                seed.Providers.Add(new Provider { Id = providerId, TimeZoneId = "UTC" });
                seed.Appointments.Add(new Appointment
                {
                    Id = Guid.NewGuid(),
                    ProviderId = providerId,
                    PatientId = patientId,
                    StartUtc = startUtc,
                    EndUtc = startUtc.AddMinutes(30),
                    Reason = "itchy spot",
                    CreatedAtUtc = DateTime.UtcNow,
                });
                await seed.SaveChangesAsync();
            }

            var factory = new SingleOptionsDbContextFactory(options);
            var booking = new BookingService(factory, new AvailabilityService(factory), new NotificationService(factory), new JitsiVideoRoomProvider(), NullLogger<BookingService>.Instance);

            var forPatient = await booking.ListForPatientAsync(patientId);
            var forProvider = await booking.ListForProviderAsync(providerId);

            var patientView = Assert.Single(forPatient);
            Assert.Equal("Dr. Example", patientView.ProviderName);
            Assert.Equal("Pat Patient", patientView.PatientName);

            var providerView = Assert.Single(forProvider);
            Assert.Equal(patientView.Id, providerView.Id);
        }
        finally
        {
            SqliteConnection.ClearAllPools();
            File.Delete(dbPath);
        }
    }
}
