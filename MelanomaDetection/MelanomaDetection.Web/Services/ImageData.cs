namespace MelanomaDetection.Web.Services;

/// <summary>
/// The Flask API returns every image as a bare base64 PNG string. This wraps
/// the one conversion the UI needs so the data-URL prefix isn't spelled out in
/// every component that shows a thumbnail.
/// </summary>
public static class ImageData
{
    public static string PngUrl(string base64) => $"data:image/png;base64,{base64}";

    public static bool HasImage(string? base64) => !string.IsNullOrEmpty(base64);
}
