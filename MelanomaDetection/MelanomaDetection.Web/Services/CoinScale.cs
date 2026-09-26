using System.Globalization;

namespace MelanomaDetection.Web.Services;

/// <summary>A coin someone can line the on-screen circle up with, to give a photo a real size.</summary>
public sealed record Coin(string Id, string Name, double DiameterMm)
{
    /// <summary>"24.26 mm", shown under the coin's name in the picker.</summary>
    public string SizeLabel => $"{DiameterMm.ToString("0.##", CultureInfo.InvariantCulture)} mm";
}

/// <summary>
/// Which coin is in the photo and how many of the photo's own pixels it spans. The analysis
/// service turns that into millimetres per pixel from its own coin table (validation.py's
/// COIN_DIAMETERS_MM), so the spot's diameter and any growth between checks come out in mm.
/// </summary>
public sealed record CoinScale(string CoinId, double DiameterPx)
{
    /// <summary>Same coins and sizes as validation.py's COIN_DIAMETERS_MM (official US Mint diameters).</summary>
    public static IReadOnlyList<Coin> Coins { get; } =
    [
        new("penny", "Penny", 19.05),
        new("nickel", "Nickel", 21.21),
        new("dime", "Dime", 17.91),
        new("quarter", "Quarter", 24.26),
    ];

    /// <summary>A circle narrower than this is too small to line up accurately. Matches validation.py.</summary>
    public const double MinDiameterPx = 20;

    public static Coin? Find(string? id) => Coins.FirstOrDefault(coin => coin.Id == id);

    /// <summary>What goes in the "coin_diameter_px" form field: invariant, so never "121,3".</summary>
    public string DiameterPxText => Math.Round(DiameterPx, 1).ToString(CultureInfo.InvariantCulture);

    /// <summary>True when this can be sent: a known coin and a circle big enough to trust.</summary>
    public bool IsUsable => Find(CoinId) is not null && double.IsFinite(DiameterPx) && DiameterPx >= MinDiameterPx;
}
