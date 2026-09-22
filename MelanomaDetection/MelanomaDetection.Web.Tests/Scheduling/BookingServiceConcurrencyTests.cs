using MelanomaDetection.Web.Data;
using MelanomaDetection.Web.Services.Scheduling;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging.Abstractions;

namespace MelanomaDetection.Web.Tests.Scheduling;

/// <summary>
/// Proves the double-booking guard holds under real concurrent requests for
/// the exact same provider+slot against a real SQLite file -- not mocked, and
/// not sequential: each attempt runs on its own thread-pool thread via
/// Task.Run so their availability reads can genuinely overlap before any of
/// them writes. Exactly one must win; see AppointmentUniqueIndexTests for a
/// deterministic proof of the DB constraint that makes that true regardless
/// of how the race actually interleaves.
/// </summary>
public class BookingServiceConcurrencyTests
{
    [Fact]
    public async Task ConcurrentBookingsForTheSameSlotYieldExactlyOneWinner()
    {
        var dbPath = Path.Combine(Path.GetTempPath(), $"telehealth-concurrency-{Guid.NewGuid():N}.db");
        var options = new DbContextOptionsBuilder<AppDbContext>()
            .UseSqlite($"Data Source={dbPath}")
            .Options;

        try
        {
            await using (var setup = new AppDbContext(options))
            {
                await setup.Database.EnsureCreatedAsync();
            }

            var providerId = Guid.NewGuid();
            var date = new DateOnly(2027, 6, 7); // a fixed future date, whenever this runs
            await using (var seed = new AppDbContext(options))
            {
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
            var availability = new AvailabilityService(factory);
            var booking = new BookingService(factory, availability, new NotificationService(factory), NullLogger<BookingService>.Instance);
            var slotStartUtc = date.ToDateTime(TimeOnly.Parse("09:00"), DateTimeKind.Utc);

            // Real thread-pool parallelism (not cooperative async interleaving on one
            // thread), so several attempts can genuinely be mid-availability-check at
            // the same instant -- the scenario only the DB constraint, not the
            // pre-check, can resolve safely.
            const int competitors = 8;
            var tasks = Enumerable.Range(0, competitors)
                .Select(_ => Task.Run(() => booking.BookAsync(providerId, Guid.NewGuid(), slotStartUtc, "patient", CancellationToken.None)))
                .ToArray();
            var results = await Task.WhenAll(tasks);

            Assert.Single(results, r => r.Outcome == BookingOutcome.Booked);
            Assert.All(results.Where(r => r.Outcome != BookingOutcome.Booked),
                r => Assert.True(r.Outcome is BookingOutcome.SlotNotAvailable or BookingOutcome.Conflict));

            await using var verify = new AppDbContext(options);
            var activeBookingsForSlot = await verify.Appointments
                .Where(a => a.ProviderId == providerId && a.StartUtc == slotStartUtc && a.Status != AppointmentStatus.Cancelled)
                .CountAsync();
            Assert.Equal(1, activeBookingsForSlot);
        }
        finally
        {
            SqliteConnection.ClearAllPools();
            File.Delete(dbPath);
        }
    }
}
