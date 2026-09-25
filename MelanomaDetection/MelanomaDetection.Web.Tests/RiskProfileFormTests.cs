using Bunit;
using MelanomaDetection.Web.Components.Settings;
using MelanomaDetection.Web.Models;

namespace MelanomaDetection.Web.Tests;

public sealed class RiskProfileFormTests : BunitContext
{
    public RiskProfileFormTests() => JSInterop.Mode = JSRuntimeMode.Loose;

    private (IRenderedComponent<RiskProfileForm> Cut, RiskProfile Profile) RenderForm(int? birthYear)
    {
        var profile = new RiskProfile { BirthYear = birthYear };
        var cut = Render<RiskProfileForm>(p => p.Add(x => x.Profile, profile));
        return (cut, profile);
    }

    [Fact]
    public void AValidYearAndSexAreStoredOnTheProfile()
    {
        var (cut, profile) = RenderForm(null);

        cut.Find("input[type=number]").Change("1990");
        cut.Find(".about-you-field select").Change("female");

        Assert.Equal(1990, profile.BirthYear);
        Assert.Equal("female", profile.Sex);
        Assert.Empty(cut.FindAll(".about-you-problem"));
    }

    [Fact]
    public void ATypoNeverErasesTheSavedYear()
    {
        var (cut, profile) = RenderForm(1990);

        cut.Find("input[type=number]").Change("19900");

        Assert.Equal(1990, profile.BirthYear);
        Assert.Contains("Your saved year (1990) is kept", cut.Find(".about-you-problem").TextContent);
    }

    [Fact]
    public void ClearingTheBoxClearsTheYear()
    {
        var (cut, profile) = RenderForm(1990);

        cut.Find("input[type=number]").Change("");

        Assert.Null(profile.BirthYear);
        Assert.Empty(cut.FindAll(".about-you-problem"));
    }
}
