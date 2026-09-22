using MelanomaDetection.Web.Data;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;

namespace MelanomaDetection.Web.Tests.Data;

/// <summary>
/// Deterministic (no timing/thread-scheduling involved) proof that the
/// filtered unique index in AppDbContext -- (ProviderId, StartUtc) unique
/// among non-cancelled rows -- is what actually stops a double booking, not
/// just BookingService's own read-before-write check. This is the mechanism
/// BookingServiceConcurrencyTests relies on under real concurrency.
/// </summary>
public class AppointmentUniqueIndexTests
{
    private static DbContextOptions<AppDbContext> NewDbOptions(out string dbPath)
    {
        dbPath = Path.Combine(Path.GetTempPath(), $"telehealth-index-{Guid.NewGuid():N}.db");
        return new DbContextOptionsBuilder<AppDbContext>().UseSqlite($"Data Source={dbPath}").Options;
    }

    private static Appointment NewAppointment(Guid providerId, DateTime startUtc, AppointmentStatus status = AppointmentStatus.Booked) => new()
    {
        Id = Guid.NewGuid(),
        ProviderId = providerId,
        PatientId = Guid.NewGuid(),
        StartUtc = startUtc,
        EndUtc = startUtc.AddMinutes(30),
        Status = status,
        CreatedAtUtc = DateTime.UtcNow,
    };

    [Fact]
    public async Task SecondNonCancelledBookingForTheSameProviderAndStartIsRejectedAtTheDatabase()
    {
        var options = NewDbOptions(out var dbPath);
        try
        {
            var providerId = Guid.NewGuid();
            var startUtc = new DateTime(2027, 6, 7, 9, 0, 0, DateTimeKind.Utc);

            await using (var setup = new AppDbContext(options))
            {
                await setup.Database.EnsureCreatedAsync();
                setup.Appointments.Add(NewAppointment(providerId, startUtc));
                await setup.SaveChangesAsync();
            }

            await using var db = new AppDbContext(options);
            db.Appointments.Add(NewAppointment(providerId, startUtc));

            var ex = await Assert.ThrowsAsync<DbUpdateException>(() => db.SaveChangesAsync());
            var sqliteEx = Assert.IsType<SqliteException>(ex.InnerException);
            Assert.Equal(19, sqliteEx.SqliteErrorCode); // SQLITE_CONSTRAINT
        }
        finally
        {
            SqliteConnection.ClearAllPools();
            File.Delete(dbPath);
        }
    }

    [Fact]
    public async Task CancelledAppointmentDoesNotBlockRebookingTheSameSlot()
    {
        var options = NewDbOptions(out var dbPath);
        try
        {
            var providerId = Guid.NewGuid();
            var startUtc = new DateTime(2027, 6, 7, 9, 0, 0, DateTimeKind.Utc);

            await using (var setup = new AppDbContext(options))
            {
                await setup.Database.EnsureCreatedAsync();
                setup.Appointments.Add(NewAppointment(providerId, startUtc, AppointmentStatus.Cancelled));
                await setup.SaveChangesAsync();
            }

            await using var db = new AppDbContext(options);
            db.Appointments.Add(NewAppointment(providerId, startUtc));
            await db.SaveChangesAsync(); // should not throw

            Assert.Equal(2, await db.Appointments.CountAsync());
        }
        finally
        {
            SqliteConnection.ClearAllPools();
            File.Delete(dbPath);
        }
    }

    [Fact]
    public async Task DifferentStartTimesForTheSameProviderAreBothAllowed()
    {
        var options = NewDbOptions(out var dbPath);
        try
        {
            var providerId = Guid.NewGuid();
            var startUtc = new DateTime(2027, 6, 7, 9, 0, 0, DateTimeKind.Utc);

            await using var db = new AppDbContext(options);
            await db.Database.EnsureCreatedAsync();
            db.Appointments.Add(NewAppointment(providerId, startUtc));
            db.Appointments.Add(NewAppointment(providerId, startUtc.AddMinutes(30)));
            await db.SaveChangesAsync(); // should not throw

            Assert.Equal(2, await db.Appointments.CountAsync());
        }
        finally
        {
            SqliteConnection.ClearAllPools();
            File.Delete(dbPath);
        }
    }
}
