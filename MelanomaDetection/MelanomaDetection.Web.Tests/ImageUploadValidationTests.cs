using MelanomaDetection.Web.Services;

namespace MelanomaDetection.Web.Tests;

/// <summary>
/// The check every photo passes before it is sent for analysis, whether it was picked, taken
/// with the camera or cropped. It mirrors validation.py's check_image_upload in the analysis
/// service, which only decodes JPEG, PNG and BMP.
/// </summary>
public class ImageUploadValidationTests
{
    private static readonly byte[] Jpeg = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
    private static readonly byte[] Png = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00];
    private static readonly byte[] Bmp = [.. "BM"u8, .. new byte[24]];
    private static readonly byte[] WebP = [.. "RIFF"u8, 0, 0, 0, 0, .. "WEBP"u8];

    public static TheoryData<byte[], string> RealImages => new()
    {
        { Jpeg, ".jpg" },
        { Png, ".png" },
        { Bmp, ".bmp" },
    };

    [Theory]
    [MemberData(nameof(RealImages))]
    public void APhotoIsRecognisedByItsBytes(byte[] data, string extension)
    {
        Assert.Equal(extension, InputLimits.ImageExtension(data));
        Assert.Null(InputLimits.ImageProblem(data));
    }

    [Fact]
    public void AnEmptyPhotoIsRefused()
    {
        Assert.Contains("empty", InputLimits.ImageProblem([]));
    }

    [Fact]
    public void APhotoOverTheLimitIsRefused()
    {
        var data = new byte[InputLimits.ImageMaxBytes + 1];
        Jpeg.CopyTo(data, 0);
        Assert.Contains("5 MB", InputLimits.ImageProblem(data));
    }

    [Fact]
    public void APhotoExactlyAtTheLimitIsAccepted()
    {
        var data = new byte[InputLimits.ImageMaxBytes];
        Jpeg.CopyTo(data, 0);
        Assert.Null(InputLimits.ImageProblem(data));
    }

    public static TheoryData<byte[]> NotPhotos => new()
    {
        "MZ\u0090 an executable"u8.ToArray(),
        "<html><script>alert(1)</script>"u8.ToArray(),
        "%PDF-1.7"u8.ToArray(),
        "BM"u8.ToArray(),   // two letters, not a BMP header
        WebP,               // the analysis service can't decode it; PhotoIntake converts it first
    };

    [Theory]
    [MemberData(nameof(NotPhotos))]
    public void AnythingElseIsRefusedWhateverItClaimsToBe(byte[] data)
    {
        Assert.Null(InputLimits.ImageExtension(data));
        Assert.Contains("doesn't look like a photo", InputLimits.ImageProblem(data));
    }

    [Theory]
    [InlineData("IMG_0001.HEIC", "IMG_0001.jpg")]   // converted in the browser, name unchanged
    [InlineData("mole.png", "mole.jpg")]
    [InlineData("mole", "mole.jpg")]
    [InlineData("", "photo.jpg")]
    [InlineData(null, "photo.jpg")]
    [InlineData("../../etc/passwd", "passwd.jpg")]
    public void TheNameSentGetsTheExtensionTheBytesReallyHave(string? original, string expected)
    {
        Assert.Equal(expected, InputLimits.ImageFileName(Jpeg, original));
    }

    [Theory]
    [InlineData(".jpg", "image/jpeg")]
    [InlineData(".png", "image/png")]
    [InlineData(".bmp", "image/bmp")]
    public void TheMediaTypeFollowsTheRealFormat(string extension, string contentType)
    {
        Assert.Equal(contentType, InputLimits.ImageContentType(extension));
    }
}
