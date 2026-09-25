using MelanomaDetection.Web.Services;

namespace MelanomaDetection.Web.Tests;

/// <summary>
/// Which photos get re-encoded in the browser before the app's own checks. The conversion
/// itself runs in the person's browser and was checked in real Edge, Chrome, Firefox and
/// WebKit (Safari's engine); this covers the decision about when to do it.
/// </summary>
public class PhotoIntakeTests
{
    private const long Limit = InputLimits.ImageMaxBytes;

    [Theory]
    [InlineData("image/heic", "IMG_0001.HEIC")]
    [InlineData("image/heif", "IMG_0001.heif")]
    [InlineData("IMAGE/HEIC", "photo")]
    [InlineData("", "IMG_0001.HEIC")]           // some browsers give a HEIC file no type at all
    [InlineData(null, "IMG_0001.heic")]
    public void AnAppleHeicPhotoIsConvertedWhateverItsSize(string? contentType, string fileName)
    {
        Assert.True(PhotoIntake.NeedsConversion(contentType, 300_000, fileName));
    }

    [Theory]
    [InlineData("image/jpeg")]
    [InlineData("image/jpg")]
    [InlineData("image/png")]
    [InlineData("image/webp")]
    [InlineData("image/bmp")]
    public void ASupportedPhotoOverTheSizeLimitIsShrunk(string contentType)
    {
        Assert.True(PhotoIntake.NeedsConversion(contentType, Limit + 1, "photo.jpg"));
    }

    [Theory]
    [InlineData("image/jpeg", 1L)]
    [InlineData("image/png", 2_000_000L)]
    [InlineData("image/jpeg", Limit)]           // exactly at the limit is still allowed as it is
    public void APhotoWithinTheLimitIsLeftAlone(string contentType, long size)
    {
        Assert.False(PhotoIntake.NeedsConversion(contentType, size, "photo.jpg"));
    }

    [Theory]
    [InlineData("application/pdf")]
    [InlineData("text/plain")]
    [InlineData("image/svg+xml")]
    [InlineData("")]
    [InlineData(null)]
    public void ThingsThatAreNotSupportedPhotosAreNeverConverted(string? contentType)
    {
        // Left alone so the app's own file check rejects them with its usual message.
        Assert.False(PhotoIntake.NeedsConversion(contentType, Limit * 4, "document.pdf"));
    }

    [Fact]
    public void AConversionThatNeverAnswersIsGivenAShortDeadline()
    {
        // A browser that can't decode the image (HEIC in Chrome, say) never replies, so the wait
        // has to be capped or the page would sit there forever.
        Assert.InRange(PhotoIntake.ConversionTimeout, TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(30));
    }
}
