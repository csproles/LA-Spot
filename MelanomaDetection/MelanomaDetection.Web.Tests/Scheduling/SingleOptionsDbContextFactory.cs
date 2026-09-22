using MelanomaDetection.Web.Data;
using Microsoft.EntityFrameworkCore;

namespace MelanomaDetection.Web.Tests.Scheduling;

/// <summary>Minimal IDbContextFactory over a fixed DbContextOptions -- the DI container isn't spun up in these tests.</summary>
public sealed class SingleOptionsDbContextFactory(DbContextOptions<AppDbContext> options) : IDbContextFactory<AppDbContext>
{
    public AppDbContext CreateDbContext() => new(options);

    public Task<AppDbContext> CreateDbContextAsync(CancellationToken cancellationToken = default) =>
        Task.FromResult(CreateDbContext());
}
