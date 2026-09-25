using System.Security.Claims;
using Bunit;
using MelanomaDetection.Web.Components.Scheduling;
using MelanomaDetection.Web.Data;
using MelanomaDetection.Web.Services;
using MelanomaDetection.Web.Services.Account;
using MelanomaDetection.Web.Services.RateLimiting;
using MelanomaDetection.Web.Services.Scheduling;
using Microsoft.AspNetCore.Components.Authorization;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;

namespace MelanomaDetection.Web.Tests.Scheduling;

/// <summary>Drives the shared booking component (used by the Book a visit page and the
/// map's booking box) end to end against a real SQLite database: providers list,
/// open times show, a click books, and a time someone else took can't be double-booked.</summary>
public sealed class VisitBookerTests : BunitContext, IDisposable
{
    private readonly string _dbPath = Path.Combine(Path.GetTempPath(), $"visit-booker-{Guid.NewGuid():N}.db");
    private readonly DbContextOptions<AppDbContext> _options;
    private readonly Guid _patientId = Guid.NewGuid();
    private readonly List<Guid> _providerIds = [];
    private BookingService _booking = null!;

    public VisitBookerTests()
    {
        _options = new DbContextOptionsBuilder<AppDbContext>().UseSqlite($"Data Source={_dbPath}").Options;
    }

    private async Task SeedAsync(int providerCount, string?[]? hubCities = null)
    {
        await using (var db = new AppDbContext(_options))
        {
            await db.Database.EnsureCreatedAsync();
            db.Users.Add(new AppUser { Id = _patientId, GoogleSubject = "patient-sub", Email = "patient@example.com", DisplayName = "Pat Patient" });

            for (var i = 0; i < providerCount; i++)
            {
                var id = Guid.NewGuid();
                _providerIds.Add(id);
                db.Users.Add(new AppUser { Id = id, GoogleSubject = $"provider-{i}", Email = $"dr{i}@example.com", DisplayName = $"Dr. Number {i}" });
                db.Providers.Add(new Provider { Id = id, TimeZoneId = "UTC", AppointmentLengthMinutes = 30, BufferMinutes = 0, HubCity = hubCities?[i] });

                // Open every day so there are always slots inside the next 7 days.
                foreach (var weekday in Enum.GetValues<DayOfWeek>())
                {
                    db.AvailabilityRules.Add(new AvailabilityRule
                    {
                        ProviderId = id, Weekday = weekday,
                        StartTime = TimeOnly.Parse("00:00"), EndTime = TimeOnly.Parse("23:00"),
                    });
                }
            }

            await db.SaveChangesAsync();
        }

        var factory = new SingleOptionsDbContextFactory(_options);
        var availability = new AvailabilityService(factory);
        _booking = new BookingService(
            factory, availability, new NotificationService(factory),
            new JitsiVideoRoomProvider(), NullLogger<BookingService>.Instance);

        Services.AddSingleton<IDbContextFactory<AppDbContext>>(factory);
        Services.AddSingleton(availability);
        Services.AddSingleton(_booking);
        Services.AddSingleton<AuthenticationStateProvider>(new FakeAuthProvider(_patientId));
        Services.AddSingleton<CurrentUser>();
        Services.AddSingleton(sp => new ImageProcessingService(
            new HttpClient { BaseAddress = new Uri("http://localhost:1") },
            sp.GetRequiredService<CurrentUser>(),
            new OperationRateLimiter(),
            new MemoryCache(new MemoryCacheOptions())));

        JSInterop.Mode = JSRuntimeMode.Loose;
        JSInterop.Setup<string>("skinCheckScheduling.getTimeZone").SetResult("UTC");
    }

    [Fact]
    public async Task ASingleProviderIsSelectedAutomaticallyAndABookingGoesThrough()
    {
        await SeedAsync(1);

        var cut = Render<VisitBooker>(p => p.Add(x => x.Embedded, true));

        // Their open times are on screen without picking anyone first.
        cut.WaitForAssertion(() => Assert.NotEmpty(cut.FindAll("button.week-slot-btn")));
        Assert.Contains("2. Choose a time", cut.Markup);

        cut.Find("button.week-slot-btn").Click();
        cut.Find("textarea").Change("a new spot on my arm");
        Assert.Contains("Dr. Number 0", cut.Find("p.book-visit-summary").TextContent);

        cut.Find("button.book-visit-submit").Click();

        cut.WaitForAssertion(() => Assert.NotNull(cut.Find(".book-visit-confirmation")));
        Assert.Contains("Visit booked", cut.Markup);
        Assert.Contains("Join call now", cut.Markup);

        var visits = await _booking.ListForPatientAsync(_patientId);
        var visit = Assert.Single(visits);
        Assert.Equal("a new spot on my arm", visit.Reason);
    }

    [Fact]
    public async Task WithSeveralProvidersNoTimesShowUntilOneIsChosen()
    {
        await SeedAsync(2);

        var cut = Render<VisitBooker>(p => p.Add(x => x.Embedded, true));

        cut.WaitForAssertion(() => Assert.Equal(2, cut.FindAll(".choice-card").Count));
        Assert.Empty(cut.FindAll("button.week-slot-btn"));
        Assert.DoesNotContain("Choose a time", cut.Markup);

        cut.FindAll(".choice-card")[1].Click();

        cut.WaitForAssertion(() => Assert.NotEmpty(cut.FindAll("button.week-slot-btn")));
        Assert.Contains("2. Choose a time", cut.Markup);
    }

    [Fact]
    public async Task WithAZipCodeTheClosestDoctorIsListedFirstAndLabelled()
    {
        // Doctor 0 is in New Orleans, doctor 1 in Shreveport.
        await SeedAsync(2, ["New Orleans", "Shreveport"]);

        var shreveport = Render<VisitBooker>(p => p
            .Add(x => x.Embedded, true).Add(x => x.NearHubCity, "Shreveport").Add(x => x.NearPlaceName, "Bossier City"));
        shreveport.WaitForAssertion(() => Assert.Equal(2, shreveport.FindAll(".choice-card").Count));
        var cards = shreveport.FindAll(".choice-card");
        Assert.Contains("Dr. Number 1", cards[0].TextContent);
        Assert.Contains("Shreveport (closest to you)", cards[0].TextContent);
        Assert.Contains("New Orleans", cards[1].TextContent);
        Assert.DoesNotContain("closest to you", cards[1].TextContent);
        Assert.Contains("Doctors closest to Bossier City, LA are listed first", shreveport.Markup);

        // Baton Rouge is much nearer New Orleans than Shreveport, so that order flips.
        var batonRouge = Render<VisitBooker>(p => p.Add(x => x.Embedded, true).Add(x => x.NearHubCity, "Baton Rouge"));
        batonRouge.WaitForAssertion(() => Assert.Equal(2, batonRouge.FindAll(".choice-card").Count));
        Assert.Contains("Dr. Number 0", batonRouge.FindAll(".choice-card")[0].TextContent);
        Assert.DoesNotContain("closest to you", batonRouge.Markup.Replace("Doctors closest to", ""));
    }

    [Fact]
    public async Task WithoutAZipCodeTheListKeepsItsOrderAndAsksForOne()
    {
        await SeedAsync(2, ["New Orleans", "Shreveport"]);

        var cut = Render<VisitBooker>(p => p.Add(x => x.Embedded, true));

        cut.WaitForAssertion(() => Assert.Equal(2, cut.FindAll(".choice-card").Count));
        Assert.Contains("Enter your zip code above", cut.Markup);
        Assert.DoesNotContain("(closest to you)", cut.Markup);
        Assert.Contains("Dr. Number 0", cut.FindAll(".choice-card")[0].TextContent);
    }

    [Fact]
    public async Task ADoctorWithNoCityIsListedAfterTheLocatedOnes()
    {
        await SeedAsync(2, [null, "Shreveport"]);

        var cut = Render<VisitBooker>(p => p.Add(x => x.Embedded, true).Add(x => x.NearHubCity, "Monroe"));

        cut.WaitForAssertion(() => Assert.Equal(2, cut.FindAll(".choice-card").Count));
        Assert.Contains("Dr. Number 1", cut.FindAll(".choice-card")[0].TextContent);
        Assert.Contains("Dr. Number 0", cut.FindAll(".choice-card")[1].TextContent);
    }

    [Fact]
    public async Task ATimeSomeoneElseTookIsNotDoubleBookedAndDisappears()
    {
        await SeedAsync(1);

        var cut = Render<VisitBooker>(p => p.Add(x => x.Embedded, true));
        cut.WaitForAssertion(() => Assert.NotEmpty(cut.FindAll("button.week-slot-btn")));

        // Tomorrow's first slot (00:00 UTC) is fixed, unlike today's, which moves with the clock.
        var tomorrow = DateOnly.FromDateTime(DateTime.UtcNow).AddDays(1);
        var taken = tomorrow.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc);
        var tomorrowColumn = cut.FindAll(".week-day-column")[1];
        tomorrowColumn.QuerySelector("button.week-slot-btn")!.Click();

        // Someone else books that same time between the list loading and the click.
        var other = await _booking.BookAsync(_providerIds[0], Guid.NewGuid(), taken, null);
        Assert.Equal(BookingOutcome.Booked, other.Outcome);

        cut.Find("button.book-visit-submit").Click();

        cut.WaitForAssertion(() => Assert.Contains("That time was just booked by someone else", cut.Markup));
        Assert.Empty(cut.FindAll(".book-visit-confirmation"));
        Assert.Empty(await _booking.ListForPatientAsync(_patientId));

        // The taken time is gone from the list rather than left clickable.
        cut.WaitForAssertion(() => Assert.DoesNotContain(
            cut.FindAll(".week-day-column")[1].QuerySelectorAll("button.week-slot-btn"),
            b => b.TextContent.Trim() == "12:00 AM"));
    }

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);
        SqliteConnection.ClearAllPools();
        File.Delete(_dbPath);
    }

    private sealed class FakeAuthProvider(Guid userId) : AuthenticationStateProvider
    {
        public override Task<AuthenticationState> GetAuthenticationStateAsync() =>
            Task.FromResult(new AuthenticationState(new ClaimsPrincipal(
                new ClaimsIdentity([new Claim(AppClaimTypes.UserId, userId.ToString())], "test"))));
    }
}
