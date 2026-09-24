namespace MelanomaDetection.Web.Services;

/// <summary>
/// The Flask API returns every image as a bare base64 PNG string. This wraps
/// the one conversion the UI needs so the data-URL prefix isn't spelled out in
/// every component that shows a thumbnail.
/// </summary>
public static class ImageData
{
    public static string PngUrl(string base64) => $"data:image/png;base64,{base64}";

    /// <summary>Some pipeline stages (original/filtered/hair-removed) are JPEG-encoded
    /// server-side to keep the analysis payload smaller -- see main.py's _JPEG_STAGES.</summary>
    public static string JpegUrl(string base64) => $"data:image/jpeg;base64,{base64}";

    public static string Url(string base64, bool isJpeg) => isJpeg ? JpegUrl(base64) : PngUrl(base64);

    public static bool HasImage(string? base64) => !string.IsNullOrEmpty(base64);
}
