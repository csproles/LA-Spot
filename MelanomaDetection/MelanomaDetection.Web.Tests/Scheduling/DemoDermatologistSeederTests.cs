using MelanomaDetection.Web.Data;
using MelanomaDetection.Web.Services;
using MelanomaDetection.Web.Services.Scheduling;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;

namespace MelanomaDetection.Web.Tests.Scheduling;

public sealed class DemoDermatologistSeederTests : IDisposable
{
    private readonly string _dbPath = Path.Combine(Path.GetTempPath(), $"seeder-{Guid.NewGuid():N}.db");
    private readonly DbContextOptions<AppDbContext> _options;

    public DemoDermatologistSeederTests()
    {
        _options = new DbContextOptionsBuilder<AppDbContext>().UseSqlite($"Data Source={_dbPath}").Options;
        using var db = new AppDbContext(_options);
        db.Database.EnsureCreated();
    }

    private SingleOptionsDbContextFactory Factory => new(_options);

    [Fact]
    public async Task SeedsEveryDoctorWithTheirWeeklyHours()
    {
        await DemoDermatologistSeeder.SeedAsync(Factory);

        await using var db = new AppDbContext(_options);
        Assert.Equal(DemoDermatologistSeeder.Seeds.Length, await db.Providers.CountAsync());
        Assert.Equal(DemoDermatologistSeeder.Seeds.Sum(s => s.Days.Length), await db.AvailabilityRules.CountAsync());

        // Every doctor can actually be booked: there are open times in the next two weeks.
        var availability = new AvailabilityService(Factory);
        var today = DateOnly.FromDateTime(DateTime.UtcNow);
        foreach (var provider in await availability.ListProvidersAsync())
        {
            var slots = await availability.GetOpenSlotsUtcAsync(provider.Id, today, today.AddDays(13));
            Assert.NotEmpty(slots);
        }
    }

    [Fact]
    public void EveryHubCityHasADoctorSoAnyZipCodeHasSomeoneClose()
    {
        var doctorCities = DemoDermatologistSeeder.Seeds.Select(s => s.HubCity).ToHashSet();

        foreach (var hub in LouisianaRegions.AllHubs)
        {
            Assert.Contains(hub.CityLabel, doctorCities);
        }

        // ...and every city named on a doctor is a real hub, so the zip ordering can place them.
        Assert.All(doctorCities, city => Assert.NotNull(LouisianaRegions.FindHub(city)));
    }

    [Fact]
    public async Task EachDoctorIsSavedWithTheirCity()
    {
        await DemoDermatologistSeeder.SeedAsync(Factory);

        await using var db = new AppDbContext(_options);
        var byName = await db.Users.Join(db.Providers, u => u.Id, p => p.Id, (u, p) => new { u.DisplayName, p.HubCity })
            .ToDictionaryAsync(x => x.DisplayName, x => x.HubCity);
        foreach (var seed in DemoDermatologistSeeder.Seeds)
        {
            Assert.Equal(seed.HubCity, byName[seed.Name]);
        }
    }

    [Fact]
    public async Task RunningItAgainAddsNothing()
    {
        await DemoDermatologistSeeder.SeedAsync(Factory);
        await DemoDermatologistSeeder.SeedAsync(Factory);

        await using var db = new AppDbContext(_options);
        Assert.Equal(DemoDermatologistSeeder.Seeds.Length, await db.Providers.CountAsync());
        Assert.Equal(DemoDermatologistSeeder.Seeds.Sum(s => s.Days.Length), await db.AvailabilityRules.CountAsync());
    }

    [Fact]
    public async Task AnOlderDatabaseGetsTheNewDoctorsAndDaysWithoutChangingExistingHours()
    {
        // The state an already-deployed database is in: one seeded doctor, Monday only,
        // with hours a provider has edited themselves.
        var existing = Guid.NewGuid();
        await using (var db = new AppDbContext(_options))
        {
            db.Users.Add(new AppUser
            {
                Id = existing, GoogleSubject = $"seed-derm:{Guid.NewGuid():N}", Email = string.Empty,
                DisplayName = "Dr. Amara Okafor", IsProvider = true,
            });
            db.Providers.Add(new Provider { Id = existing, Specialty = "Dermatology" });
            db.AvailabilityRules.Add(new AvailabilityRule
            {
                ProviderId = existing, Weekday = DayOfWeek.Monday,
                StartTime = new TimeOnly(7, 0), EndTime = new TimeOnly(8, 0),
            });
            await db.SaveChangesAsync();
        }

        await DemoDermatologistSeeder.SeedAsync(Factory);

        await using var check = new AppDbContext(_options);
        Assert.Equal(DemoDermatologistSeeder.Seeds.Length, await check.Providers.CountAsync());
        Assert.Equal(1, await check.Users.CountAsync(u => u.DisplayName == "Dr. Amara Okafor"));

        var rules = await check.AvailabilityRules.Where(r => r.ProviderId == existing).ToListAsync();
        var monday = Assert.Single(rules, r => r.Weekday == DayOfWeek.Monday);
        Assert.Equal(new TimeOnly(7, 0), monday.StartTime);
        Assert.Equal(new TimeOnly(8, 0), monday.EndTime);
        Assert.Equal(5, rules.Count);

        // The doctor had no city yet, so the seeder gave them theirs.
        Assert.Equal("Baton Rouge", (await check.Providers.SingleAsync(p => p.Id == existing)).HubCity);
    }

    [Fact]
    public async Task ACityThatIsAlreadySetIsNotOverwritten()
    {
        var existing = Guid.NewGuid();
        await using (var db = new AppDbContext(_options))
        {
            db.Users.Add(new AppUser
            {
                Id = existing, GoogleSubject = $"seed-derm:{Guid.NewGuid():N}", Email = string.Empty,
                DisplayName = "Dr. Amara Okafor", IsProvider = true,
            });
            db.Providers.Add(new Provider { Id = existing, Specialty = "Dermatology", HubCity = "Shreveport" });
            await db.SaveChangesAsync();
        }

        await DemoDermatologistSeeder.SeedAsync(Factory);

        await using var check = new AppDbContext(_options);
        Assert.Equal("Shreveport", (await check.Providers.SingleAsync(p => p.Id == existing)).HubCity);
    }

    public void Dispose()
    {
        SqliteConnection.ClearAllPools();
        File.Delete(_dbPath);
    }
}
