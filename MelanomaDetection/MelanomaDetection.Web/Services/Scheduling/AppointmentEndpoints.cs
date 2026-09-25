using System.Security.Claims;
using MelanomaDetection.Web.Data;
using MelanomaDetection.Web.Services.Account;
using Microsoft.AspNetCore.Http.HttpResults;

namespace MelanomaDetection.Web.Services.Scheduling;

public sealed record BookAppointmentRequest(Guid ProviderId, DateTime StartUtc, string? Reason, AttachedScan? Scan = null);

public sealed record AppointmentSummary(
    Guid Id, Guid ProviderId, Guid PatientId, DateTime StartUtc, DateTime EndUtc, string Status, string? Reason);

/// <summary>
/// Booking and cancellation. Every handler scopes to the caller's own id from
/// the claims principal -- a patient can only ever book/see/cancel as
/// themselves, and <c>/schedule</c> refuses anyone without the provider claim.
/// </summary>
public static class AppointmentEndpoints
{
    public static IEndpointRouteBuilder MapAppointmentEndpoints(this IEndpointRouteBuilder endpoints)
    {
        var appointments = endpoints.MapGroup("/api/appointments").RequireAuthorization();

        appointments.MapPost("/", BookAsync);
        appointments.MapPost("/{id:guid}/cancel", CancelAsync);
        appointments.MapGet("/mine", MyAppointmentsAsync);
        appointments.MapGet("/schedule", ScheduleAsync);
        appointments.MapGet("/{id:guid}/report", SharedReportAsync)
            .RequireRateLimiting(RateLimiting.RateLimitPolicies.AccountData);

        return endpoints;
    }

    /// <summary>
    /// The patient's PDF report, for the provider on the visit -- only if the patient chose to
    /// share their data when booking and the visit is still booked. Anyone else gets a 404, so
    /// the endpoint doesn't reveal which visits exist.
    /// </summary>
    private static async Task<Results<FileContentHttpResult, ProblemHttpResult, NotFound>> SharedReportAsync(
        Guid id,
        ClaimsPrincipal principal,
        BookingService booking,
        ImageProcessingService imageProcessing,
        CancellationToken cancellationToken)
    {
        if (!CurrentUser.IsProvider(principal))
        {
            return TypedResults.NotFound();
        }

        var providerId = CurrentUser.ReadUserId(principal)!.Value;
        AppointmentView? appointment;
        try
        {
            appointment = await booking.FindForUserAsync(id, providerId, cancellationToken);
        }
        catch (UnauthorizedAccessException)
        {
            return TypedResults.NotFound();
        }

        if (appointment is null)
        {
            return TypedResults.NotFound();
        }

        try
        {
            var pdf = await imageProcessing.GetSharedReportPdfAsync(appointment, providerId);
            var fileName = $"{Slug(appointment.PatientName)}-skin-check-report.pdf";
            return TypedResults.File(pdf, "application/pdf", fileName);
        }
        catch (UnauthorizedAccessException)
        {
            return TypedResults.NotFound();
        }
        catch (ImageProcessingApiException ex)
        {
            return TypedResults.Problem(ex.Message, statusCode: StatusCodes.Status502BadGateway);
        }
    }

    private static string Slug(string name)
    {
        var slug = new string([.. name.ToLowerInvariant().Select(c => char.IsAsciiLetterOrDigit(c) ? c : '-')]).Trim('-');
        while (slug.Contains("--", StringComparison.Ordinal))
        {
            slug = slug.Replace("--", "-", StringComparison.Ordinal);
        }

        return slug.Length == 0 ? "patient" : slug;
    }

    private static async Task<Results<Created<AppointmentSummary>, ProblemHttpResult, NotFound>> BookAsync(
        BookAppointmentRequest request,
        ClaimsPrincipal principal,
        BookingService booking,
        CancellationToken cancellationToken)
    {
        var patientId = CurrentUser.ReadUserId(principal)!.Value;

        BookingResult result;
        try
        {
            result = await booking.BookAsync(request.ProviderId, patientId, request.StartUtc, request.Reason, request.Scan, cancellationToken: cancellationToken);
        }
        catch (KeyNotFoundException)
        {
            return TypedResults.NotFound();
        }

        return result.Outcome switch
        {
            BookingOutcome.Booked => TypedResults.Created(
                $"/api/appointments/{result.Appointment!.Id}", ToSummary(result.Appointment)),
            BookingOutcome.SlotNotAvailable => TypedResults.Problem(
                "That slot is no longer available. Refresh and pick another time.", statusCode: StatusCodes.Status409Conflict),
            BookingOutcome.Conflict => TypedResults.Problem(
                "That slot was just booked by someone else. Refresh and pick another time.", statusCode: StatusCodes.Status409Conflict),
            _ => throw new InvalidOperationException($"Unhandled booking outcome {result.Outcome}."),
        };
    }

    private static async Task<Results<Ok, ProblemHttpResult, NotFound>> CancelAsync(
        Guid id,
        ClaimsPrincipal principal,
        BookingService booking,
        CancellationToken cancellationToken)
    {
        var userId = CurrentUser.ReadUserId(principal)!.Value;
        try
        {
            var found = await booking.CancelAsync(id, userId, cancellationToken);
            return found ? TypedResults.Ok() : TypedResults.NotFound();
        }
        catch (UnauthorizedAccessException)
        {
            return TypedResults.Problem("Not your appointment.", statusCode: StatusCodes.Status403Forbidden);
        }
    }

    private static async Task<Ok<IReadOnlyList<AppointmentView>>> MyAppointmentsAsync(
        ClaimsPrincipal principal, BookingService booking, CancellationToken cancellationToken)
    {
        var patientId = CurrentUser.ReadUserId(principal)!.Value;
        var appointments = await booking.ListForPatientAsync(patientId, cancellationToken);
        return TypedResults.Ok(appointments);
    }

    private static async Task<Results<Ok<IReadOnlyList<AppointmentView>>, ProblemHttpResult>> ScheduleAsync(
        ClaimsPrincipal principal, BookingService booking, CancellationToken cancellationToken)
    {
        if (!CurrentUser.IsProvider(principal))
        {
            return TypedResults.Problem("Not a provider account.", statusCode: StatusCodes.Status403Forbidden);
        }

        var providerId = CurrentUser.ReadUserId(principal)!.Value;
        var appointments = await booking.ListForProviderAsync(providerId, cancellationToken);
        return TypedResults.Ok(appointments);
    }

    private static AppointmentSummary ToSummary(Appointment a) =>
        new(a.Id, a.ProviderId, a.PatientId, a.StartUtc, a.EndUtc, a.Status.ToString(), a.Reason);
}
