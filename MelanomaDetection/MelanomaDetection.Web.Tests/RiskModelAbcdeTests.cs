using System.Net;
using System.Security.Claims;
using System.Text;
using System.Text.Json;
using MelanomaDetection.Web.Models;
using MelanomaDetection.Web.Services;
using MelanomaDetection.Web.Services.Account;
using MelanomaDetection.Web.Services.RateLimiting;
using Microsoft.AspNetCore.Components.Authorization;
using Microsoft.Extensions.Caching.Memory;

namespace MelanomaDetection.Web.Tests;

/// <summary>
/// The A, B and C shown on the results page come from the risk model's own measurements, the
/// same ones its score was made from, so the bars never disagree with the score beside them.
/// The rules for the numbers live in the analysis service (tests/test_abcd_scores.py); this
/// covers reading them, applying them, and asking for them.
/// </summary>
public class RiskModelAbcdeTests
{
    private const string PredictJson = """
        {
          "risk_score": 0.61, "risk_level": "medium", "yolo_found_lesion": true,
          "abcd_features": {}, "fold_scores": [0.5, 0.7], "processingId": "pred_abc",
          "abcde_adopted": true,
          "abcde_scores": {
            "asymmetry": {"score": 7.5, "details": {"raw_asymmetry_ratio": 0.30, "concern": true}},
            "border": {"score": 2.5, "details": {"raw_border_irregularity": 0.25, "concern": false}},
            "color": {"score": null, "details": {"color_cv": null, "concern": false}},
            "diameter": {"score": null, "details": {"reason": "no calibration", "diameter_px": 300.0, "concern": false}},
            "evolving": {"score": null, "details": {"reason": "no prior check to compare against"}}
          }
        }
        """;

    private static PredictResponse Parse(string json) => JsonSerializer.Deserialize<PredictResponse>(json)!;

    private static ImageProcessingResults V5Results() => new()
    {
        AbcdeScores = new AbcdeScores
        {
            Asymmetry = new AbcdeScore { Score = 1.0 },
            Border = new AbcdeScore { Score = 1.5 },
            Color = new AbcdeScore { Score = 2.0 },
            Diameter = new AbcdeScore { Score = null },
            Evolving = new AbcdeScore { Score = 4.0 },
        },
    };

    [Fact]
    public void TheResponseCarriesTheRiskModelsScoresIncludingAnUnmeasuredOne()
    {
        var response = Parse(PredictJson);

        Assert.True(response.AbcdeAdopted);
        Assert.Equal(7.5, response.AbcdeScores!.Asymmetry.Score);
        Assert.Equal(2.5, response.AbcdeScores.Border.Score);
        Assert.Null(response.AbcdeScores.Color.Score); // shown as N/A, not as a measured zero
    }

    [Fact]
    public void AnOlderServiceWithoutTheFieldsChangesNothing()
    {
        var response = Parse("""{"risk_score": 0.4, "risk_level": "medium", "processingId": "pred_x"}""");
        var results = V5Results();

        response.ApplyTo(results);

        Assert.False(response.AbcdeAdopted);
        Assert.Null(response.AbcdeScores);
        Assert.Equal(1.0, results.AbcdeScores.Asymmetry.Score);
    }

    [Fact]
    public void WhenAdoptedTheResultsShowTheRiskModelsAbcAndKeepDiameterAndEvolving()
    {
        var results = V5Results();

        Parse(PredictJson).ApplyTo(results);

        Assert.Equal(7.5, results.AbcdeScores.Asymmetry.Score);
        Assert.Equal(2.5, results.AbcdeScores.Border.Score);
        Assert.Null(results.AbcdeScores.Color.Score);
        Assert.Equal(4.0, results.AbcdeScores.Evolving.Score);
        Assert.Null(results.AbcdeScores.Diameter.Score);
    }

    [Fact]
    public void ScoresTheServiceDidNotAdoptAreNotApplied()
    {
        // e.g. the risk model could not find the spot: the check keeps the numbers it had.
        var response = Parse(PredictJson.Replace("\"abcde_adopted\": true", "\"abcde_adopted\": false"));
        var results = V5Results();

        response.ApplyTo(results);

        Assert.Equal(1.0, results.AbcdeScores.Asymmetry.Score);
        Assert.Equal(1.5, results.AbcdeScores.Border.Score);
        Assert.Equal(2.0, results.AbcdeScores.Color.Score);
    }

    [Fact]
    public async Task ThePhotoIsScoredAgainstTheCheckItBelongsTo()
    {
        var handler = new CapturingHandler(PredictJson);
        var service = NewService(handler);

        var response = await service.PredictRiskAsync(Jpeg, "spot.jpg", 40, "female", "head/neck", "proc_123abc");

        Assert.Equal("pred_abc", response.ProcessingId);
        Assert.Contains("name=linked_processing_id", handler.Body);
        Assert.Contains("proc_123abc", handler.Body);
    }

    [Fact]
    public async Task WithoutACheckNoLinkIsSent()
    {
        var handler = new CapturingHandler(PredictJson);
        var service = NewService(handler);

        await service.PredictRiskAsync(Jpeg, "spot.jpg", null, null, null);

        Assert.DoesNotContain("linked_processing_id", handler.Body);
    }

    private static byte[] Jpeg => [0xFF, 0xD8, 0xFF, 0xE0, 0, 0x10, (byte)'J', (byte)'F', (byte)'I', (byte)'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0xFF, 0xD9];

    private static ImageProcessingService NewService(HttpMessageHandler handler)
    {
        var user = new CurrentUser(new FakeAuth(Guid.NewGuid()));
        return new ImageProcessingService(
            new HttpClient(handler) { BaseAddress = new Uri("http://localhost:5002") },
            user, new OperationRateLimiter(), new MemoryCache(new MemoryCacheOptions()));
    }

    private sealed class CapturingHandler(string responseJson) : HttpMessageHandler
    {
        public string Body { get; private set; } = string.Empty;

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Body = request.Content is null ? string.Empty : await request.Content.ReadAsStringAsync(cancellationToken);
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(responseJson, Encoding.UTF8, "application/json"),
            };
        }
    }

    private sealed class FakeAuth(Guid userId) : AuthenticationStateProvider
    {
        public override Task<AuthenticationState> GetAuthenticationStateAsync() =>
            Task.FromResult(new AuthenticationState(new ClaimsPrincipal(
                new ClaimsIdentity([new Claim(AppClaimTypes.UserId, userId.ToString())], "test"))));
    }
}
