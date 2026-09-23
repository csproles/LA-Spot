using MelanomaDetection.Web.Data;
using Microsoft.EntityFrameworkCore;

namespace MelanomaDetection.Web.Services.Scheduling;

/// <summary>
/// Seeds 2-3 fake dermatologist accounts with a couple of weeks' worth of
/// availability, so "Find a Dermatologist" has something to show on a fresh
/// database instead of an empty state -- these are demo accounts, not real
/// Google sign-ins, and there is no UI path to book them as anything other
/// than the seeded providers.
/// </summary>
public static class DemoDermatologistSeeder
{
    private const string SeedSubjectPrefix = "seed-derm:";

    private static readonly (string Name, string Specialty, string Credentials, string Bio, DayOfWeek[] Days, TimeOnly Start, TimeOnly End)[] Seeds =
    [
        ("Dr. Amara Okafor", "Dermatology", "MD, Board-Certified Dermatologist",
            "General and cosmetic dermatology with a focus on early skin cancer detection.",
            [DayOfWeek.Monday, DayOfWeek.Wednesday, DayOfWeek.Friday], new TimeOnly(9, 0), new TimeOnly(12, 0)),
        ("Dr. Miguel Torres", "Dermatology", "MD, PhD, Dermato-Oncology",
            "Specializes in pigmented lesions and melanoma risk assessment.",
            [DayOfWeek.Tuesday, DayOfWeek.Thursday], new TimeOnly(13, 0), new TimeOnly(17, 0)),
        ("Dr. Priya Nair", "Dermatology", "MD, Mohs Surgery",
            "Skin cancer surgery and general dermatology, telehealth follow-ups welcome.",
            [DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Thursday], new TimeOnly(15, 0), new TimeOnly(18, 0)),
    ];

    public static async Task SeedDemoDermatologistsAsync(this WebApplication app)
    {
        var factory = app.Services.GetRequiredService<IDbContextFactory<AppDbContext>>();
        await using var db = await factory.CreateDbContextAsync();

        // Idempotent: only seed once, on whichever deploy first finds none of these rows.
        if (await db.Users.AnyAsync(u => u.GoogleSubject.StartsWith(SeedSubjectPrefix)))
        {
            return;
        }

        var now = DateTime.UtcNow;
        foreach (var seed in Seeds)
        {
            var id = Guid.NewGuid();
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
            });
            foreach (var day in seed.Days)
            {
                db.AvailabilityRules.Add(new AvailabilityRule
                {
                    ProviderId = id,
                    Weekday = day,
                    StartTime = seed.Start,
                    EndTime = seed.End,
                });
            }
        }

        await db.SaveChangesAsync();
    }
}
