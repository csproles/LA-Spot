using System.Security.Claims;
using System.Text;
using MelanomaDetection.Web.Services.Account;
using Microsoft.AspNetCore.Http.HttpResults;

namespace MelanomaDetection.Web.Services.Scheduling;

public static class NotificationEndpoints
{
    public static IEndpointRouteBuilder MapNotificationEndpoints(this IEndpointRouteBuilder endpoints)
    {
        endpoints.MapGet("/api/notifications/{id:guid}/ics", DownloadIcsAsync).RequireAuthorization();
        return endpoints;
    }

    private static async Task<Results<FileContentHttpResult, NotFound>> DownloadIcsAsync(
        Guid id, ClaimsPrincipal principal, NotificationService notifications, CancellationToken cancellationToken)
    {
        var userId = CurrentUser.ReadUserId(principal)!.Value;
        var ics = await notifications.FindIcsForUserAsync(id, userId, cancellationToken);
        if (ics is null)
        {
            return TypedResults.NotFound();
        }

        return TypedResults.File(Encoding.UTF8.GetBytes(ics), "text/calendar", "visit.ics");
    }
}
