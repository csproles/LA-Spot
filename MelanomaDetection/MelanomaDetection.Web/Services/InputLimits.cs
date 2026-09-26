namespace MelanomaDetection.Web.Services;

/// <summary>
/// What the app accepts from people, in one place. The analysis service checks
/// the same limits (MelanomaDetection.Python/validation.py) as its own last line
/// of defence; checking here first means a person gets a specific message
/// instead of a round trip and a generic one, and inputs use these as maxlength.
/// Keep the two in step.
/// </summary>
public static class InputLimits
{
    public const int SpotLabelMax = 60;
    public const int BodyRegionMax = 60;
    public const int LocationMax = 120;
    public const int FullNameMax = 120;
    public const int NotesMax = 2000;
    public const int SymptomMax = 60;
    public const int SymptomsMaxCount = 20;
    public const int ChatQuestionMax = 500;

    public const long ImageMaxBytes = 5 * 1024 * 1024;

    /// <summary>
    /// The file extension that matches what the bytes really are: ".jpg", ".png" or ".bmp",
    /// the formats the analysis service decodes (validation.py's check_image_upload). Null for
    /// anything else, WebP and HEIC included (PhotoIntake turns those into JPEGs first). A
    /// file's declared type and name are just claims from the browser; this is what it is.
    /// </summary>
    public static string? ImageExtension(ReadOnlySpan<byte> data)
    {
        if (data.StartsWith(JpegSignature))
        {
            return ".jpg";
        }

        if (data.StartsWith(PngSignature))
        {
            return ".png";
        }

        // "BM" alone is two ordinary letters, so also require room for the BMP header.
        return data.Length >= 26 && data.StartsWith("BM"u8) ? ".bmp" : null;
    }

    /// <summary>True when the bytes are a JPEG, PNG or BMP (see <see cref="ImageExtension"/>).</summary>
    public static bool LooksLikeImage(ReadOnlySpan<byte> data) => ImageExtension(data) is not null;

    /// <summary>
    /// Why these bytes can't be sent for analysis, as a message to show the person, or null
    /// when they can. Every photo passes this, whether picked, taken with the camera or cropped.
    /// </summary>
    public static string? ImageProblem(ReadOnlySpan<byte> data)
    {
        if (data.IsEmpty)
        {
            return "The photo is empty. Please choose another one.";
        }

        if (data.Length > ImageMaxBytes)
        {
            return $"The photo is too large ({data.Length / (1024.0 * 1024.0):F2} MB). Maximum allowed size is 5 MB.";
        }

        return LooksLikeImage(data)
            ? null
            : "That file doesn't look like a photo. Please choose a JPEG, PNG, BMP, WebP or HEIC image.";
    }

    /// <summary>
    /// <paramref name="fileName"/> with its extension replaced by the one the bytes really
    /// have. A HEIC photo converted to JPEG in the browser keeps its "IMG_0001.HEIC" name, and
    /// the analysis service refuses a file whose extension isn't .jpg, .png or .bmp.
    /// </summary>
    public static string ImageFileName(ReadOnlySpan<byte> data, string? fileName)
    {
        var stem = Path.GetFileNameWithoutExtension(fileName ?? string.Empty);
        return (string.IsNullOrWhiteSpace(stem) ? "photo" : stem) + (ImageExtension(data) ?? string.Empty);
    }

    /// <summary>The media type for an extension from <see cref="ImageExtension"/>.</summary>
    public static string ImageContentType(string extension) => extension switch
    {
        ".png" => "image/png",
        ".bmp" => "image/bmp",
        _ => "image/jpeg",
    };

    // Raw bytes, not UTF-8 literals: 0x89 and 0xFF aren't valid single-byte UTF-8.
    private static ReadOnlySpan<byte> PngSignature => [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    private static ReadOnlySpan<byte> JpegSignature => [0xFF, 0xD8, 0xFF];
}
