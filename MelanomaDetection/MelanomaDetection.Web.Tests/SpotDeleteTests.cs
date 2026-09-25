using System.Net;
using System.Security.Claims;
using System.Text;
using Bunit;
using MelanomaDetection.Web.Pages;
using MelanomaDetection.Web.Services;
using MelanomaDetection.Web.Services.Account;
using MelanomaDetection.Web.Services.Chat;
using MelanomaDetection.Web.Services.RateLimiting;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Authorization;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.DependencyInjection;

namespace MelanomaDetection.Web.Tests;

public sealed class SpotDeleteTests : BunitContext
{
    private readonly RecordingHandler _handler = new();

    public SpotDeleteTests()
    {
        JSInterop.Mode = JSRuntimeMode.Loose;
        var client = new HttpClient(_handler) { BaseAddress = new Uri("http://flask") };
        Services.AddSingleton(new ImageProcessingService(
            client,
            new CurrentUser(new SignedIn()),
            new OperationRateLimiter(),
            new MemoryCache(new MemoryCacheOptions()),
            new SingleClientFactory(client)));
        Services.AddSingleton(new ChatPageContext());
    }

    private IRenderedComponent<SpotPage> RenderPage()
    {
        var cut = Render<SpotPage>(p => p.Add(x => x.SpotId, "spot_abc"));
        cut.WaitForElement(".spot-delete-button");
        return cut;
    }

    [Fact]
    public void DeletingAsksFirstAndDoesNothingUntilConfirmed()
    {
        var cut = RenderPage();

        cut.Find(".spot-delete-button").Click();

        var prompt = cut.Find(".spot-delete-confirm");
        Assert.Contains("Delete “Forearm mole”?", prompt.TextContent);
        Assert.Contains("can't be undone", prompt.TextContent);
        Assert.DoesNotContain(_handler.Requests, r => r.StartsWith("DELETE", StringComparison.Ordinal));
    }

    [Fact]
    public void CancelLeavesTheSpotAlone()
    {
        var cut = RenderPage();
        cut.Find(".spot-delete-button").Click();

        cut.FindAll(".spot-delete-actions button").Single(b => b.TextContent.Trim() == "Cancel").Click();

        Assert.Empty(cut.FindAll(".spot-delete-confirm"));
        Assert.DoesNotContain(_handler.Requests, r => r.StartsWith("DELETE", StringComparison.Ordinal));
    }

    [Fact]
    public void ConfirmingDeletesTheSpotAndReturnsToTheListWithAMessage()
    {
        var cut = RenderPage();
        cut.Find(".spot-delete-button").Click();

        cut.Find(".spot-delete-confirm-button").Click();

        cut.WaitForAssertion(() => Assert.Contains("DELETE /api/spots/spot_abc", _handler.Requests));
        var nav = Services.GetRequiredService<NavigationManager>();
        cut.WaitForAssertion(() => Assert.EndsWith("/spots?deleted=Forearm%20mole", nav.Uri));
    }

    private sealed class RecordingHandler : HttpMessageHandler
    {
        public List<string> Requests { get; } = [];

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            lock (Requests)
            {
                Requests.Add($"{request.Method} {request.RequestUri!.AbsolutePath}");
            }

            var body = request.Method == HttpMethod.Delete
                ? """{"deleted":true,"checks":2}"""
                : """{"id":"spot_abc","label":"Forearm mole","bodyRegion":"Left arm","checkCount":0,"checks":[]}""";
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(body, Encoding.UTF8, "application/json"),
            });
        }
    }

    private sealed class SingleClientFactory(HttpClient client) : IHttpClientFactory
    {
        public HttpClient CreateClient(string name) => client;
    }

    private sealed class SignedIn : AuthenticationStateProvider
    {
        public override Task<AuthenticationState> GetAuthenticationStateAsync() =>
            Task.FromResult(new AuthenticationState(new ClaimsPrincipal(
                new ClaimsIdentity([new Claim(AppClaimTypes.UserId, Guid.NewGuid().ToString())], "test"))));
    }
}
