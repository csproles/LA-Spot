using Bunit;
using MelanomaDetection.Web.Components.CheckFlow;
using MelanomaDetection.Web.Services;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Forms;
using Microsoft.JSInterop;

namespace MelanomaDetection.Web.Tests;

/// <summary>
/// The camera tile's behaviour around the photo hand-off. The photo must come back from the
/// browser as a stream: a Blazor Server circuit drops the whole connection if the browser sends
/// it more than 32 KB in one message, which is what a base64 string of a real photo did (see
/// wwwroot/cameraCapture.js). The browser side was checked in real Edge, Chrome and Firefox.
/// </summary>
public sealed class CameraCaptureTests : BunitContext
{
    private byte[]? _captured;

    private IRenderedComponent<CameraCapture> RenderWithCamera(IJSStreamReference? photo, bool supported = true)
    {
        JSInterop.Mode = JSRuntimeMode.Loose;
        JSInterop.Setup<bool>("skinCheckCamera.isSupported").SetResult(supported);
        JSInterop.Setup<string?>("skinCheckCamera.start", _ => true).SetResult(null);
        JSInterop.SetupVoid("skinCheckCamera.stop", _ => true).SetVoidResult();
        JSInterop.Setup<IJSStreamReference?>("skinCheckCamera.capture", _ => true).SetResult(photo);

        return Render<CameraCapture>(p => p
            .Add(x => x.OnFileSelected, EventCallback.Factory.Create<InputFileChangeEventArgs>(this, _ => { }))
            .Add(x => x.OnCaptured, EventCallback.Factory.Create<byte[]>(this, bytes => _captured = bytes)));
    }

    private static void OpenAndPressShutter(IRenderedComponent<CameraCapture> cut)
    {
        cut.WaitForAssertion(() => Assert.NotEmpty(cut.FindAll("button.capture-tile")));
        cut.Find("button.capture-tile").Click();
        cut.WaitForAssertion(() => Assert.NotEmpty(cut.FindAll(".camera-shutter:not([disabled])")));
        cut.Find(".camera-shutter").Click();
    }

    [Fact]
    public void APhotoLargerThanTheOldMessageLimitIsDeliveredAndTheViewfinderCloses()
    {
        var photo = new byte[600 * 1024];
        photo[0] = 0xFF;
        photo[1] = 0xD8;
        var cut = RenderWithCamera(new FakeStreamReference(photo));

        OpenAndPressShutter(cut);

        cut.WaitForAssertion(() => Assert.NotNull(_captured));
        Assert.Equal(photo, _captured);
        Assert.Empty(cut.FindAll(".camera-overlay"));
    }

    [Fact]
    public void ACaptureWithNoFrameShowsAHintAndKeepsTheCameraOpenForARetry()
    {
        var cut = RenderWithCamera(photo: null);

        OpenAndPressShutter(cut);

        cut.WaitForAssertion(() => Assert.Contains("try again", cut.Find(".camera-hint").TextContent));
        Assert.Null(_captured);
        Assert.NotEmpty(cut.FindAll(".camera-overlay"));
        Assert.NotEmpty(cut.FindAll(".camera-shutter:not([disabled])"));
    }

    [Fact]
    public void APhotoOverTheUploadLimitShowsAHintInsteadOfBreakingTheConnection()
    {
        var cut = RenderWithCamera(new FakeStreamReference(new byte[(int)InputLimits.ImageMaxBytes + 1]));

        OpenAndPressShutter(cut);

        cut.WaitForAssertion(() => Assert.Contains("could not be read", cut.Find(".camera-hint").TextContent));
        Assert.Null(_captured);
        Assert.NotEmpty(cut.FindAll(".camera-overlay"));
    }

    [Fact]
    public void WithoutALiveCameraApiThePhoneCameraFileInputIsOffered()
    {
        var cut = RenderWithCamera(photo: null, supported: false);

        cut.WaitForAssertion(() => Assert.NotEmpty(cut.FindAll("input[type=file][capture]")));
        Assert.Empty(cut.FindAll("button.capture-tile"));
    }

    // Like the real reference, refuses to open a stream longer than the limit it is given,
    // and does so with the same exception type.
    private sealed class FakeStreamReference(byte[] data) : IJSStreamReference
    {
        public long Length => data.Length;

        public ValueTask<Stream> OpenReadStreamAsync(long maxAllowedSize = 512000, CancellationToken cancellationToken = default) =>
            data.Length > maxAllowedSize
                ? throw new InvalidOperationException("The stream is larger than the allowed size.")
                : ValueTask.FromResult<Stream>(new MemoryStream(data));

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
