using Bunit;
using MelanomaDetection.Web.Components.CheckFlow;

namespace MelanomaDetection.Web.Tests;

public sealed class PhotoCropTests : BunitContext
{
    private const string Photo = "data:image/jpeg;base64,AAAA";

    private IRenderedComponent<PhotoConfirmStep> RenderStep(Action<byte[]> onCropped, bool isCropped = false, Action? onUndo = null) =>
        Render<PhotoConfirmStep>(p => p
            .Add(x => x.PreviewDataUrl, Photo)
            .Add(x => x.FileName, "camera-capture.jpg")
            .Add(x => x.FileSize, 2048)
            .Add(x => x.OnConfirm, () => { })
            .Add(x => x.OnRetake, () => { })
            .Add(x => x.OnCropped, onCropped)
            .Add(x => x.IsCropped, isCropped)
            .Add(x => x.OnUndoCrop, onUndo ?? (() => { })));

    private BunitJSModuleInterop CropModule()
    {
        var module = JSInterop.SetupModule("./js/crop.js");
        module.SetupVoid("init", _ => true).SetVoidResult();
        module.SetupVoid("dispose", _ => true).SetVoidResult();
        return module;
    }

    [Fact]
    public void CropOpensTheCropperAndCancelGoesBack()
    {
        CropModule();
        var cut = RenderStep(_ => { });

        cut.FindAll("button").Single(b => b.TextContent.Trim() == "Crop").Click();
        Assert.NotNull(cut.Find(".crop-box"));
        cut.WaitForAssertion(() => Assert.False(cut.Find(".cropper .btn-primary").HasAttribute("disabled")));

        cut.FindAll("button").Single(b => b.TextContent.Trim() == "Cancel").Click();
        Assert.Empty(cut.FindAll(".crop-box"));
        Assert.Contains("Use this photo?", cut.Markup);
    }

    [Fact]
    public void ATooSmallCropSaysWhyAndKeepsTheCropperOpen()
    {
        var module = CropModule();
        module.Setup<PhotoCropper.CropOutcome>("crop", _ => true)
            .SetResult(new PhotoCropper.CropOutcome(false, "That area is too small to analyze. Make the box bigger."));
        var cropped = false;
        var cut = RenderStep(_ => cropped = true);
        cut.FindAll("button").Single(b => b.TextContent.Trim() == "Crop").Click();
        cut.WaitForAssertion(() => Assert.False(cut.Find(".cropper .btn-primary").HasAttribute("disabled")));

        cut.Find(".cropper .btn-primary").Click();

        cut.WaitForAssertion(() => Assert.Contains("too small to analyze", cut.Find(".form-feedback").TextContent));
        Assert.False(cropped);
        Assert.NotNull(cut.Find(".crop-box"));
    }

    [Fact]
    public void UndoCropIsOfferedOnlyAfterACrop()
    {
        var undone = false;
        var plain = RenderStep(_ => { });
        Assert.DoesNotContain("Undo crop", plain.Markup);

        var cropped = RenderStep(_ => { }, isCropped: true, onUndo: () => undone = true);
        cropped.FindAll("button").Single(b => b.TextContent.Trim() == "Undo crop").Click();

        Assert.True(undone);
        Assert.Contains("Back to the original photo.", cropped.Markup);
    }
}
