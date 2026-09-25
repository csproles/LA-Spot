using Bunit;
using MelanomaDetection.Web.Components.Shared;

namespace MelanomaDetection.Web.Tests;

public sealed class DownloadButtonTests : BunitContext
{
    private IRenderedComponent<DownloadButton> RenderButton() =>
        Render<DownloadButton>(p => p
            .Add(x => x.Href, "/account/report")
            .Add(x => x.Label, "Export my data (PDF)")
            .Add(x => x.FileName, "skin-check-report.pdf")
            .Add(x => x.ContentType, "application/pdf"));

    private BunitJSModuleInterop Module() => JSInterop.SetupModule("./js/download.js");

    [Fact]
    public void ASuccessfulDownloadSaysWhatWasSaved()
    {
        Module().Setup<DownloadButton.DownloadResult>("download", "/account/report", "skin-check-report.pdf", "application/pdf")
            .SetResult(new DownloadButton.DownloadResult(true, "skin-check-report.pdf", 400_000, null));
        var cut = RenderButton();

        cut.Find("a").Click();

        var feedback = cut.WaitForElement(".form-feedback");
        Assert.Contains("is-success", feedback.ClassName);
        Assert.Contains("Downloaded skin-check-report.pdf (391 KB)", feedback.TextContent);
        Assert.Equal("status", feedback.GetAttribute("role"));
    }

    [Fact]
    public void AFailedDownloadShowsTheServersReasonInsteadOfTheBrowsersError()
    {
        Module().Setup<DownloadButton.DownloadResult>("download", _ => true)
            .SetResult(new DownloadButton.DownloadResult(false, null, 0, "The report service may still be starting up."));
        var cut = RenderButton();

        cut.Find("a").Click();

        var feedback = cut.WaitForElement(".form-feedback");
        Assert.Contains("is-error", feedback.ClassName);
        Assert.Contains("still be starting up", feedback.TextContent);
        Assert.Equal("alert", feedback.GetAttribute("role"));
    }

    [Fact]
    public void WhilePreparingTheButtonSaysSoAndIgnoresMoreClicks()
    {
        var pending = Module().Setup<DownloadButton.DownloadResult>("download", _ => true);
        var cut = RenderButton();

        cut.Find("a").Click();
        cut.Find("a").Click();

        Assert.Contains("Preparing", cut.Find("a").TextContent);
        Assert.Equal("true", cut.Find("a").GetAttribute("aria-disabled"));
        Assert.Single(pending.Invocations);

        pending.SetResult(new DownloadButton.DownloadResult(true, "skin-check-report.pdf", 10, null));
        cut.WaitForAssertion(() => Assert.Contains("Export my data (PDF)", cut.Find("a").TextContent));
    }
}
