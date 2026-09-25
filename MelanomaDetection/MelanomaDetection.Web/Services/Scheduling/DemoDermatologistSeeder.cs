using MelanomaDetection.Web.Data;
using Microsoft.EntityFrameworkCore;

namespace MelanomaDetection.Web.Services.Scheduling;

/// <summary>
/// Seeds fake dermatologist accounts, one for each Louisiana hub city on the Find a doctor
/// map, each with weekly availability, so the map's booking box has open times to show for
/// any zip code instead of an empty state. These are demo accounts, not real Google
/// sign-ins, and the open times are placeholders: a weekly pattern, not actual clinic
/// hours. There is no UI path to book them as anything other than the seeded providers.
/// </summary>
public static class DemoDermatologistSeeder
{
    private const string SeedSubjectPrefix = "seed-derm:";

    private static readonly DayOfWeek[] WeekDays =
        [DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Wednesday, DayOfWeek.Thursday, DayOfWeek.Friday];

    /// <summary>HubCity must be a city name from LouisianaRegions (a test checks that every
    /// hub has at least one doctor, so any zip code has someone close by).</summary>
    public static readonly (string Name, string HubCity, string Specialty, string Credentials, string Bio, DayOfWeek[] Days, TimeOnly Start, TimeOnly End)[] Seeds =
    [
        ("Dr. Amara Okafor", "Baton Rouge", "Dermatology", "MD, Board-Certified Dermatologist",
            "General and cosmetic dermatology with a focus on early skin cancer detection.",
            WeekDays, new TimeOnly(9, 0), new TimeOnly(12, 0)),
        ("Dr. Miguel Torres", "New Orleans", "Dermatology", "MD, PhD, Dermato-Oncology",
            "Specializes in pigmented lesions and melanoma risk assessment.",
            WeekDays, new TimeOnly(13, 0), new TimeOnly(17, 0)),
        ("Dr. Priya Nair", "Lafayette", "Dermatology", "MD, Mohs Surgery",
            "Skin cancer surgery and general dermatology, telehealth follow-ups welcome.",
            WeekDays, new TimeOnly(15, 0), new TimeOnly(18, 0)),
        ("Dr. Samuel Whitfield", "Shreveport", "Dermatology", "MD, Pediatric Dermatology",
            "Skin checks for children and teens, and general dermatology for the whole family.",
            [DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Wednesday, DayOfWeek.Thursday, DayOfWeek.Saturday],
            new TimeOnly(8, 0), new TimeOnly(11, 0)),
        ("Dr. Lena Broussard", "Lake Charles", "Dermatology", "MD, Dermatology",
            "Mole checks and follow-up visits for spots you are already tracking.",
            [DayOfWeek.Tuesday, DayOfWeek.Wednesday, DayOfWeek.Thursday], new TimeOnly(10, 0), new TimeOnly(15, 0)),
        ("Dr. Kevin Nguyen", "Monroe", "Dermatology", "MD, Dermatology",
            "Evening and weekend telehealth visits for people who cannot come in during the day.",
            [DayOfWeek.Monday, DayOfWeek.Wednesday, DayOfWeek.Friday, DayOfWeek.Sunday],
            new TimeOnly(17, 0), new TimeOnly(20, 0)),
        ("Dr. Renee Guidry", "Houma", "Dermatology", "MD, Dermatology",
            "General skin checks and follow-up visits for spots you are watching.",
            [DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Thursday, DayOfWeek.Friday], new TimeOnly(9, 0), new TimeOnly(13, 0)),
        ("Dr. Marcus Hebert", "Alexandria", "Dermatology", "MD, Dermatology",
            "Skin cancer screening and mole checks, with afternoon and Saturday times.",
            [DayOfWeek.Tuesday, DayOfWeek.Wednesday, DayOfWeek.Friday, DayOfWeek.Saturday], new TimeOnly(12, 0), new TimeOnly(16, 0)),
        ("Dr. Tasha Landry", "Hammond", "Dermatology", "MD, Dermatology",
            "Routine skin exams and questions about changing moles.",
            WeekDays, new TimeOnly(10, 0), new TimeOnly(14, 0)),
    ];

    public static async Task SeedDemoDermatologistsAsync(this WebApplication app) =>
        await SeedAsync(app.Services.GetRequiredService<IDbContextFactory<AppDbContext>>());

    /// <summary>
    /// Idempotent top-up, safe to run on every start: adds any demo doctor that is missing,
    /// sets a doctor's city only while it is empty, and adds a weekday's open hours only where
    /// that doctor has none for that weekday. Existing hours and cities are never changed, so
    /// an edit made from a provider dashboard survives a restart. Doctors are matched by name
    /// among the seeded accounts, so databases seeded before this list grew get the new
    /// doctors, cities and days too.
    /// </summary>
    public static async Task SeedAsync(IDbContextFactory<AppDbContext> factory, CancellationToken cancellationToken = default)
    {
        await using var db = await factory.CreateDbContextAsync(cancellationToken);

        var seeded = await db.Users
            .Where(u => u.GoogleSubject.StartsWith(SeedSubjectPrefix))
            .ToListAsync(cancellationToken);
        var seededIds = seeded.Select(u => u.Id).ToList();
        var providers = await db.Providers
            .Where(p => seededIds.Contains(p.Id))
            .ToDictionaryAsync(p => p.Id, cancellationToken);
        var haveDays = (await db.AvailabilityRules
                .Where(r => seededIds.Contains(r.ProviderId))
                .Select(r => new { r.ProviderId, r.Weekday })
                .ToListAsync(cancellationToken))
            .Select(r => (r.ProviderId, r.Weekday))
            .ToHashSet();

        var now = DateTime.UtcNow;
        foreach (var seed in Seeds)
        {
            var id = seeded.FirstOrDefault(u => u.DisplayName == seed.Name)?.Id ?? Guid.Empty;
            if (id == Guid.Empty)
            {
                id = Guid.NewGuid();
                db.Users.Add(new AppUser
                {
                    Id = id,
                    GoogleSubject = $"{SeedSubjectPrefix}{Guid.NewGuid():N}",
                    Email = string.Empty,
                    DisplayName = seed.Name,
                    CreatedAtUtc = now,
                    LastSignInAtUtc = now,
                    IsProvider = true,
                });
                db.Providers.Add(new Provider
                {
                    Id = id,
                    Specialty = seed.Specialty,
                    Credentials = seed.Credentials,
                    Bio = seed.Bio,
                    HubCity = seed.HubCity,
                });
            }
            else if (providers.TryGetValue(id, out var existing) && string.IsNullOrWhiteSpace(existing.HubCity))
            {
                existing.HubCity = seed.HubCity;
            }

            foreach (var day in seed.Days)
            {
                if (haveDays.Contains((id, day)))
                {
                    continue;
                }

                db.AvailabilityRules.Add(new AvailabilityRule
                {
                    ProviderId = id,
                    Weekday = day,
                    StartTime = seed.Start,
                    EndTime = seed.End,
                });
            }
        }

        await db.SaveChangesAsync(cancellationToken);
    }
}
