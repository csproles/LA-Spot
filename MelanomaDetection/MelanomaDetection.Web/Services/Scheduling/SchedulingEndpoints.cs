using Microsoft.AspNetCore.Http.HttpResults;

namespace MelanomaDetection.Web.Services.Scheduling;

public sealed record OpenSlotsResponse(IReadOnlyList<DateTime> Slots);

/// <summary>
/// Read-only availability endpoints for the booking flow. Signed-in users only
/// (patient or provider) -- there's nothing patient-specific to scope here since
/// a slot list and the provider directory aren't anyone's private data.
/// </summary>
public static class SchedulingEndpoints
{
    public static IEndpointRouteBuilder MapSchedulingEndpoints(this IEndpointRouteBuilder endpoints)
    {
        var providers = endpoints.MapGroup("/api/providers").RequireAuthorization();

        providers.MapGet("/", ListProvidersAsync);
        providers.MapGet("/{providerId:guid}/slots", GetSlotsAsync);

        return endpoints;
    }

    private static async Task<Ok<IReadOnlyList<ProviderSummary>>> ListProvidersAsync(
        AvailabilityService availability, CancellationToken cancellationToken) =>
        TypedResults.Ok(await availability.ListProvidersAsync(cancellationToken));

    private static async Task<Results<Ok<OpenSlotsResponse>, BadRequest<string>, NotFound>> GetSlotsAsync(
        Guid providerId, DateOnly from, DateOnly to, AvailabilityService availability, CancellationToken cancellationToken)
    {
        if (to < from || to > from.AddDays(31))
        {
            return TypedResults.BadRequest("'to' must be on or after 'from', and at most 31 days out.");
        }

        try
        {
            var slots = await availability.GetOpenSlotsUtcAsync(providerId, from, to, cancellationToken);
            return TypedResults.Ok(new OpenSlotsResponse(slots));
        }
        catch (KeyNotFoundException)
        {
            return TypedResults.NotFound();
        }
    }
}
