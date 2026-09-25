using Microsoft.AspNetCore.Components.Forms;

namespace MelanomaDetection.Web.Services;

/// <summary>
/// Gets a picked or camera-taken photo into a form the app accepts, before its own checks run.
/// iPhone photos are the reason it exists: they are often over the 5 MB limit, and a photo in
/// Apple's HEIC format has a type the app doesn't take. Both are re-encoded as a JPEG by the
/// person's own browser (Blazor's RequestImageFileAsync), so nothing large is uploaded first.
/// Anything already within the limits is left exactly as it was.
/// </summary>
public static class PhotoIntake
{
    /// <summary>Longest side of a re-encoded photo, in pixels. Matches wwwroot/cameraCapture.js.</summary>
    public const int MaxEdgePixels = 2560;

    private static readonly HashSet<string> ResizableTypes = new(StringComparer.OrdinalIgnoreCase)
    {
        "image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp",
    };

    private static readonly HashSet<string> HeifTypes = new(StringComparer.OrdinalIgnoreCase)
    {
        "image/heic", "image/heif", "image/heic-sequence", "image/heif-sequence",
    };

    /// <summary>
    /// True for an Apple HEIC/HEIF photo (some browsers give it no type at all, so the file
    /// extension counts too), or for a photo of a supported type that is over the size limit.
    /// </summary>
    public static bool NeedsConversion(string? contentType, long size, string? fileName)
    {
        if (IsHeif(contentType, fileName))
        {
            return true;
        }

        return size > InputLimits.ImageMaxBytes
            && !string.IsNullOrEmpty(contentType)
            && ResizableTypes.Contains(contentType);
    }

    /// <summary>How long a conversion may take. A normal 12 megapixel photo takes a second or two.</summary>
    public static readonly TimeSpan ConversionTimeout = TimeSpan.FromSeconds(15);

    /// <summary>
    /// The file itself when it needs no work, otherwise a JPEG re-encoded in the browser and
    /// scaled down to <see cref="MaxEdgePixels"/>. Throws if the browser can't do that, for
    /// example a HEIC photo in a browser without HEIC support. Such a browser never answers
    /// (the request just hangs), so the wait is capped and a <see cref="TimeoutException"/>
    /// thrown instead of leaving the person looking at a page that never responds.
    /// </summary>
    public static async Task<IBrowserFile> PrepareAsync(IBrowserFile file) =>
        NeedsConversion(file.ContentType, file.Size, file.Name)
            ? await file.RequestImageFileAsync("image/jpeg", MaxEdgePixels, MaxEdgePixels)
                .AsTask()
                .WaitAsync(ConversionTimeout)
            : file;

    private static bool IsHeif(string? contentType, string? fileName) =>
        (!string.IsNullOrEmpty(contentType) && HeifTypes.Contains(contentType))
        || (!string.IsNullOrEmpty(fileName)
            && (fileName.EndsWith(".heic", StringComparison.OrdinalIgnoreCase)
                || fileName.EndsWith(".heif", StringComparison.OrdinalIgnoreCase)));
}
