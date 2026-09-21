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
    /// True when the bytes start like a PNG, JPEG, BMP or WebP. A file's declared type
    /// and name are just claims from the browser; this is what it actually is.
    /// </summary>
    public static bool LooksLikeImage(ReadOnlySpan<byte> data)
    {
        return data.StartsWith(PngSignature)
            || data.StartsWith(JpegSignature)
            || data.StartsWith("BM"u8)
            || (data.Length >= 12 && data[..4].SequenceEqual("RIFF"u8) && data.Slice(8, 4).SequenceEqual("WEBP"u8));
    }

    // Raw bytes, not UTF-8 literals: 0x89 and 0xFF aren't valid single-byte UTF-8.
    private static ReadOnlySpan<byte> PngSignature => [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    private static ReadOnlySpan<byte> JpegSignature => [0xFF, 0xD8, 0xFF];
}
