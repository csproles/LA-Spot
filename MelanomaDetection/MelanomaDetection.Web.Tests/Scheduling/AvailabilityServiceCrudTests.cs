using MelanomaDetection.Web.Data;
using MelanomaDetection.Web.Services.Scheduling;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;

namespace MelanomaDetection.Web.Tests.Scheduling;

/// <summary>Provider-facing CRUD on availability rules/exceptions/settings -- the dashboard's write path.</summary>
public class AvailabilityServiceCrudTests
{
    private static (AvailabilityService Service, string DbPath) NewService()
    {
        var dbPath = Path.Combine(Path.GetTempPath(), $"telehealth-crud-{Guid.NewGuid():N}.db");
        var options = new DbContextOptionsBuilder<AppDbContext>().UseSqlite($"Data Source={dbPath}").Options;
        using (var setup = new AppDbContext(options))
        {
            setup.Database.EnsureCreated();
        }

        return (new AvailabilityService(new SingleOptionsDbContextFactory(options)), dbPath);
    }

    private static void Cleanup(string dbPath)
    {
        SqliteConnection.ClearAllPools();
        File.Delete(dbPath);
    }

    [Fact]
    public async Task AddRuleThenRemoveRuleRoundTrips()
    {
        var (service, dbPath) = NewService();
        try
        {
            var providerId = Guid.NewGuid();
            await service.AddRuleAsync(providerId, DayOfWeek.Monday, new TimeOnly(9, 0), new TimeOnly(17, 0));

            var rules = await service.ListRulesAsync(providerId);
            var rule = Assert.Single(rules);
            Assert.Equal(DayOfWeek.Monday, rule.Weekday);

            await service.RemoveRuleAsync(providerId, rule.Id);
            Assert.Empty(await service.ListRulesAsync(providerId));
        }
        finally
        {
            Cleanup(dbPath);
        }
    }

    [Fact]
    public async Task AddRuleRejectsStartAtOrAfterEnd()
    {
        var (service, dbPath) = NewService();
        try
        {
            await Assert.ThrowsAsync<ArgumentException>(() =>
                service.AddRuleAsync(Guid.NewGuid(), DayOfWeek.Monday, new TimeOnly(17, 0), new TimeOnly(9, 0)));
        }
        finally
        {
            Cleanup(dbPath);
        }
    }

    [Fact]
    public async Task RemoveRuleRejectsARuleBelongingToAnotherProvider()
    {
        var (service, dbPath) = NewService();
        try
        {
            var ownerId = Guid.NewGuid();
            await service.AddRuleAsync(ownerId, DayOfWeek.Monday, new TimeOnly(9, 0), new TimeOnly(17, 0));
            var rule = Assert.Single(await service.ListRulesAsync(ownerId));

            await Assert.ThrowsAsync<UnauthorizedAccessException>(() =>
                service.RemoveRuleAsync(Guid.NewGuid(), rule.Id));

            // Untouched -- the rejected caller didn't own it.
            Assert.Single(await service.ListRulesAsync(ownerId));
        }
        finally
        {
            Cleanup(dbPath);
        }
    }

    [Fact]
    public async Task AddExceptionRejectsExtraHoursWithoutATimeRange()
    {
        var (service, dbPath) = NewService();
        try
        {
            await Assert.ThrowsAsync<ArgumentException>(() =>
                service.AddExceptionAsync(Guid.NewGuid(), new DateOnly(2027, 6, 7), isBlocked: false, start: null, end: null));
        }
        finally
        {
            Cleanup(dbPath);
        }
    }

    [Fact]
    public async Task AddExceptionAllowsAWholeDayBlockWithNoTimeRange()
    {
        var (service, dbPath) = NewService();
        try
        {
            var providerId = Guid.NewGuid();
            await service.AddExceptionAsync(providerId, new DateOnly(2027, 6, 7), isBlocked: true, start: null, end: null);

            var exception = Assert.Single(await service.ListExceptionsAsync(providerId, new DateOnly(2027, 1, 1)));
            Assert.Null(exception.StartTime);
            Assert.True(exception.IsBlocked);
        }
        finally
        {
            Cleanup(dbPath);
        }
    }

    [Theory]
    [InlineData(0, 10, "America/Chicago")] // too short
    [InlineData(30, -1, "America/Chicago")] // negative buffer
    [InlineData(30, 10, "Not/A_Real_Zone")] // bad timezone
    public async Task UpdateProviderSettingsRejectsInvalidInput(int lengthMinutes, int bufferMinutes, string timeZoneId)
    {
        var (service, dbPath) = NewService();
        try
        {
            var providerId = Guid.NewGuid();
            await Assert.ThrowsAnyAsync<Exception>(() =>
                service.UpdateProviderSettingsAsync(providerId, lengthMinutes, bufferMinutes, timeZoneId));
        }
        finally
        {
            Cleanup(dbPath);
        }
    }
}
