using Bunit;
using MelanomaDetection.Web.Components.SpotTracking;

namespace MelanomaDetection.Web.Tests;

public sealed class BodyMapTests : BunitContext
{
    public BodyMapTests() => JSInterop.Mode = JSRuntimeMode.Loose;

    [Fact]
    public void TheFrontAndBackExplainerHasAVisibleLabelAndOpensAPopUp()
    {
        var cut = Render<BodyMap>();

        // The trigger says what it is, instead of being a bare icon.
        var trigger = cut.Find("button.info-popover-trigger");
        Assert.Contains("What is this?", trigger.TextContent);
        Assert.Empty(cut.FindAll("[role=dialog]"));

        trigger.Click();

        var dialog = cut.Find("[role=dialog]");
        Assert.Contains("What front view and back view mean", dialog.GetAttribute("aria-label"));
        Assert.Contains("Front view", dialog.TextContent);
        Assert.Contains("Back view", dialog.TextContent);
        Assert.Contains("facing a mirror", dialog.TextContent);
        Assert.Contains("standing behind you", dialog.TextContent);

        cut.Find("button.info-popover-close").Click();
        Assert.Empty(cut.FindAll("[role=dialog]"));
    }

    [Fact]
    public void TheExplainerIsANativeDialogThatClosesOnEscapeOrABackdropClick()
    {
        var cut = Render<BodyMap>();

        cut.Find("button.info-popover-trigger").Click();
        var dialog = cut.Find("dialog.info-popover-dialog");
        JSInterop.VerifyInvoke("import");

        // A click on the backdrop lands on the dialog itself and closes it (a click inside the
        // card is stopped before it gets there -- checked in a real browser, bUnit can't dispatch it).
        dialog.Click();
        Assert.Empty(cut.FindAll("dialog.info-popover-dialog"));

        cut.Find("button.info-popover-trigger").Click();
        cut.Find("dialog.info-popover-dialog").KeyDown("Escape");
        Assert.Empty(cut.FindAll("dialog.info-popover-dialog"));
    }

    [Fact]
    public void TheToggleFlipsBetweenFrontAndBack()
    {
        var cut = Render<BodyMap>();
        Assert.Contains("Front view", cut.Find("button.body-toggle").TextContent);
        Assert.NotEmpty(cut.FindAll(".body-face"));

        cut.Find("button.body-toggle").Click();

        Assert.Contains("Back view", cut.Find("button.body-toggle").TextContent);
        Assert.Empty(cut.FindAll(".body-face"));
    }
}
