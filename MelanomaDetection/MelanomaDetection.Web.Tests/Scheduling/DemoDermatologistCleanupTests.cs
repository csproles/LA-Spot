using MelanomaDetection.Web.Data;
using MelanomaDetection.Web.Services.Scheduling;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;

namespace MelanomaDetection.Web.Tests.Scheduling;

public sealed class DemoDermatologistCleanupTests : IDisposable
{
    private readonly SqliteConnection _connection = new("Data Source=:memory:");
    private readonly DbContextOptions<AppDbContext> _options;
    private readonly Guid _realProvider = Guid.NewGuid();
    private readonly Guid _fakeProvider = Guid.NewGuid();
    private readonly Guid _patient = Guid.NewGuid();

    public DemoDermatologistCleanupTests()
    {
        _connection.Open();
        _options = new DbContextOptionsBuilder<AppDbContext>().UseSqlite(_connection).Options;
        using var db = new AppDbContext(_options);
        db.Database.EnsureCreated();

        db.Users.AddRange(
            new AppUser { Id = _realProvider, GoogleSubject = "google-callie", Email = "c.sproles2005@gmail.com", DisplayName = "Dr. Callie Sproles", IsProvider = true },
            new AppUser { Id = _fakeProvider, GoogleSubject = $"{DemoDermatologistCleanup.SeedSubjectPrefix}abc", Email = "", DisplayName = "Dr. Priya Nair", IsProvider = true },
            new AppUser { Id = _patient, GoogleSubject = "google-pat", Email = "pat@example.com", DisplayName = "Pat" });
        db.Providers.AddRange(new Provider { Id = _realProvider }, new Provider { Id = _fakeProvider });
        foreach (var provider in new[] { _realProvider, _fakeProvider })
        {
            db.AvailabilityRules.Add(new AvailabilityRule { ProviderId = provider, Weekday = DayOfWeek.Monday, StartTime = new(9, 0), EndTime = new(12, 0) });
            db.AvailabilityExceptions.Add(new AvailabilityException { ProviderId = provider, Date = new DateOnly(2027, 1, 4), IsBlocked = true });
            var appointment = new Appointment { Id = Guid.NewGuid(), ProviderId = provider, PatientId = _patient, StartUtc = DateTime.UtcNow.AddDays(provider == _realProvider ? 1 : 2) };
            db.Appointments.Add(appointment);
            db.Notifications.Add(new Notification { Id = Guid.NewGuid(), AppointmentId = appointment.Id, RecipientUserId = _patient });
        }

        db.SaveChanges();
    }

    [Fact]
    public async Task OnlyTheSeededDoctorsAndEverythingAttachedToThemAreRemoved()
    {
        var removed = await DemoDermatologistCleanup.RemoveAsync(new SingleOptionsDbContextFactory(_options));

        await using var db = new AppDbContext(_options);
        Assert.Equal(1, removed);
        Assert.Equal(_realProvider, Assert.Single(await db.Providers.ToListAsync()).Id);
        Assert.Equal(2, await db.Users.CountAsync());
        Assert.All(await db.AvailabilityRules.ToListAsync(), r => Assert.Equal(_realProvider, r.ProviderId));
        Assert.All(await db.AvailabilityExceptions.ToListAsync(), e => Assert.Equal(_realProvider, e.ProviderId));
        Assert.Equal(_realProvider, Assert.Single(await db.Appointments.ToListAsync()).ProviderId);
        Assert.Single(await db.Notifications.ToListAsync());
    }

    [Fact]
    public async Task RunningItAgainChangesNothing()
    {
        var factory = new SingleOptionsDbContextFactory(_options);
        await DemoDermatologistCleanup.RemoveAsync(factory);

        Assert.Equal(0, await DemoDermatologistCleanup.RemoveAsync(factory));
    }

    [Fact]
    public async Task TheBookingListOnlyShowsRealProviderAccounts()
    {
        await using (var db = new AppDbContext(_options))
        {
            // An account that lost provider access (email taken off the allow-list) and signed in again.
            var former = Guid.NewGuid();
            db.Users.Add(new AppUser { Id = former, GoogleSubject = "google-former", Email = "former@example.com", DisplayName = "Dr. Former", IsProvider = false });
            db.Providers.Add(new Provider { Id = former });
            await db.SaveChangesAsync();
        }

        var providers = await new AvailabilityService(new SingleOptionsDbContextFactory(_options)).ListProvidersAsync();

        Assert.Equal("Dr. Callie Sproles", Assert.Single(providers).DisplayName);
    }

    public void Dispose() => _connection.Dispose();
}
