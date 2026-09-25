using System.Net;
using System.Security.Claims;
using System.Text;
using Bunit;
using MelanomaDetection.Web.Services.Account;
using MelanomaDetection.Web.Services.RateLimiting;
using Microsoft.AspNetCore.Components.Authorization;
using Microsoft.Extensions.Caching.Memory;
using MelanomaDetection.Web.Components.CheckFlow;
using MelanomaDetection.Web.Models;
using MelanomaDetection.Web.Services;

namespace MelanomaDetection.Web.Tests;

public sealed class RiskModelInputsTests : BunitContext
{
    private static readonly DateOnly Today = new(2026, 9, 25);

    [Theory]
    [InlineData(1990, 36)]
    [InlineData(2026, 0)]
    [InlineData(2027, null)] // in the future
    [InlineData(1890, null)] // implausible
    [InlineData(null, null)]
    public void AgeComesFromTheProfilesYearOfBirth(int? birthYear, int? expected)
    {
        Assert.Equal(expected, RiskModelInputs.Age(new RiskProfile { BirthYear = birthYear }, Today));
    }

    [Fact]
    public void NoProfileMeansNoAgeOrSex()
    {
        Assert.Null(RiskModelInputs.Age(null, Today));
        Assert.Null(RiskModelInputs.Sex(null));
        Assert.Null(RiskModelInputs.Sex(new RiskProfile { Sex = "" }));
        Assert.Equal("female", RiskModelInputs.Sex(new RiskProfile { Sex = "female" }));
    }

    [Theory]
    [InlineData("Head/Neck", null, "head/neck")]
    [InlineData("Left Arm", null, "upper extremity")]
    [InlineData("Right Arm", "back", "upper extremity")]
    [InlineData("Left Leg", null, "lower extremity")]
    [InlineData("Right Leg", "front", "lower extremity")]
    [InlineData("Chest/Upper Back", "front", "anterior torso")]
    [InlineData("Chest/Upper Back", "back", "posterior torso")]
    [InlineData("Abdomen/Lower Back", "front", "anterior torso")]
    [InlineData("Abdomen/Lower Back", "back", "posterior torso")]
    [InlineData("Chest/Upper Back", null, null)] // older spot: side never recorded
    [InlineData("Somewhere else", "front", null)]
    public void BodySiteComesFromTheSpotsRegionAndSide(string region, string? side, string? expected)
    {
        Assert.Equal(expected, RiskModelInputs.BodySite(region, side));
    }

    [Fact]
    public void EveryBodyMapRegionMapsToAModelSiteOnceTheSideIsKnown()
    {
        foreach (var region in Components.SpotTracking.BodyMap.Regions)
        {
            Assert.NotNull(RiskModelInputs.BodySite(region.Name, "front"));
            Assert.NotNull(RiskModelInputs.BodySite(region.Name, "back"));
        }
    }

    [Fact]
    public async Task ATorsoSpotAsksFrontOrBackAndCreatesTheSpotWithIt()
    {
        JSInterop.Mode = JSRuntimeMode.Loose;
        (string Label, string Region, string? Side)? created = null;
        var cut = Render<SpotSelectionStep>(p => p
            .Add(x => x.Spots, new List<Spot>())
            .Add(x => x.CreateSpot, (label, region, side) =>
            {
                created = (label, region, side);
                return Task.FromResult<Spot?>(new Spot { Id = "s1", Label = label, BodyRegion = region, BodySide = side });
            })
            .Add(x => x.OnSpotChosen, (Spot _) => { }));

        cut.FindAll("button").First(b => b.TextContent.Contains("Left Arm")).Click();
        Assert.DoesNotContain("Front or back?", cut.Markup);

        cut.FindAll("button").First(b => b.TextContent.Contains("Chest/Upper Back")).Click();
        Assert.Contains("Front or back?", cut.Markup);
        cut.FindAll("button").First(b => b.TextContent.Trim() == "Back").Click();

        cut.Find("#spot-label").Input("mole between shoulders");
        await cut.InvokeAsync(() => cut.FindAll("button.btn-primary").Last().Click());

        Assert.Equal(("mole between shoulders", "Chest/Upper Back", "back"), created);
    }

    [Fact]
    public async Task SavingTheProfileSendsYearOfBirthAndSex()
    {
        var handler = new CapturingHandler();
        var client = new HttpClient(handler) { BaseAddress = new Uri("http://flask") };
        var service = new ImageProcessingService(client, new CurrentUser(new SignedIn()), new OperationRateLimiter(),
            new MemoryCache(new MemoryCacheOptions()), new SingleClientFactory(client));

        await service.SaveProfileAsync(new RiskProfile { FullName = "Pat", BirthYear = 1990, Sex = "female" });

        Assert.Contains("\"birthYear\":1990", handler.Body);
        Assert.Contains("\"sex\":\"female\"", handler.Body);
    }

    private sealed class CapturingHandler : HttpMessageHandler
    {
        public string Body { get; private set; } = string.Empty;

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Body = await request.Content!.ReadAsStringAsync(cancellationToken);
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("""{"configured":true,"birthYear":1990,"sex":"female"}""", Encoding.UTF8, "application/json"),
            };
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
