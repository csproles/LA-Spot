using System.Net;
using System.Security.Claims;
using System.Text;
using MelanomaDetection.Web.Data;
using MelanomaDetection.Web.Services;
using MelanomaDetection.Web.Services.Account;
using MelanomaDetection.Web.Services.RateLimiting;
using MelanomaDetection.Web.Services.Scheduling;
using Microsoft.AspNetCore.Components.Authorization;
using Microsoft.Extensions.Caching.Memory;

namespace MelanomaDetection.Web.Tests.Scheduling;

public class SharedHealthDataTests
{
    private static readonly Guid ProviderId = Guid.NewGuid();
    private static readonly Guid PatientId = Guid.NewGuid();

    private readonly RecordingHandler _handler = new();
    private readonly HttpClient _sessionClient;
    private readonly ImageProcessingService _service;

    public SharedHealthDataTests()
    {
        _sessionClient = new HttpClient(_handler) { BaseAddress = new Uri("http://flask") };
        _service = new ImageProcessingService(
            _sessionClient,
            new CurrentUser(new SignedInAs(ProviderId)),
            new OperationRateLimiter(),
            new MemoryCache(new MemoryCacheOptions()),
            new HandlerClientFactory(_handler));
    }

    private static AppointmentView Visit(bool shared = true, Guid? providerId = null, AppointmentStatus status = AppointmentStatus.Booked) =>
        new(Guid.NewGuid(), providerId ?? ProviderId, "Dr. Test", PatientId, "Pat", DateTime.UtcNow, DateTime.UtcNow.AddMinutes(30),
            status, null, null, null, null, null, null, shared);

    [Fact]
    public async Task ReadsThePatientsExportAsThePatientWithoutTouchingTheProvidersOwnClient()
    {
        var data = await _service.GetSharedHealthDataAsync(Visit(), ProviderId);

        Assert.Equal("mole", Assert.Single(data.Spots).Label);
        Assert.Equal(PatientId.ToString(), Assert.Single(_handler.UserIds));
        Assert.False(_sessionClient.DefaultRequestHeaders.Contains(ImageProcessingService.UserIdHeader));
    }

    [Theory]
    [InlineData(false, false, AppointmentStatus.Booked)]
    [InlineData(true, true, AppointmentStatus.Booked)]
    [InlineData(true, false, AppointmentStatus.Cancelled)]
    [InlineData(true, false, AppointmentStatus.Completed)]
    public async Task RefusesWithoutConsentOnABookedVisitOfThisProvider(bool shared, bool otherProvider, AppointmentStatus status)
    {
        var visit = Visit(shared, otherProvider ? Guid.NewGuid() : null, status);

        await Assert.ThrowsAsync<UnauthorizedAccessException>(() => _service.GetSharedHealthDataAsync(visit, ProviderId));
        Assert.Empty(_handler.UserIds);
    }

    private sealed class RecordingHandler : HttpMessageHandler
    {
        public List<string> UserIds { get; } = [];

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            UserIds.AddRange(request.Headers.GetValues(ImageProcessingService.UserIdHeader));
            const string body = """{"profile":null,"spots":[{"id":"s1","label":"mole","bodyRegion":"arm"}],"checks":[]}""";
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(body, Encoding.UTF8, "application/json"),
            });
        }
    }

    private sealed class HandlerClientFactory(HttpMessageHandler handler) : IHttpClientFactory
    {
        public HttpClient CreateClient(string name) => new(handler, disposeHandler: false) { BaseAddress = new Uri("http://flask") };
    }

    private sealed class SignedInAs(Guid userId) : AuthenticationStateProvider
    {
        public override Task<AuthenticationState> GetAuthenticationStateAsync() =>
            Task.FromResult(new AuthenticationState(new ClaimsPrincipal(
                new ClaimsIdentity([new Claim(AppClaimTypes.UserId, userId.ToString())], "test"))));
    }
}
