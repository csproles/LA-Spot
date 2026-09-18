namespace MelanomaDetection.Web.Services;

/// <summary>
/// Louisiana's 64 parishes grouped by nearest metro hub. The hub decides both
/// where the map recentres and which city the NPI Registry is queried for --
/// most parishes have no dermatologist of their own, so searching the parish
/// itself would return nothing.
/// </summary>
public static class LouisianaRegions
{
    public record Region(string CityLabel, string NpiCityQuery, double Lat, double Lng, int Zoom);

    private static readonly Dictionary<string, Region> Hubs = new()
    {
        ["shreveport"] = new("Shreveport", "Shreveport", 32.5252, -93.7502, 9),
        ["monroe"] = new("Monroe", "Monroe", 32.5093, -92.1193, 9),
        ["alexandria"] = new("Alexandria", "Alexandria", 31.3113, -92.4451, 9),
        ["lakecharles"] = new("Lake Charles", "Lake Charles", 30.2266, -93.2174, 9),
        ["lafayette"] = new("Lafayette", "Lafayette", 30.2241, -92.0198, 9),
        ["batonrouge"] = new("Baton Rouge", "Baton Rouge", 30.4515, -91.1871, 9),
        ["hammond"] = new("Hammond", "Hammond", 30.5044, -90.4615, 9),
        ["houma"] = new("Houma", "Houma", 29.5958, -90.7195, 9),
        ["neworleans"] = new("New Orleans", "New Orleans", 29.9511, -90.0715, 9),
    };

    private static readonly Dictionary<string, string> ParishToHub = new()
    {
        ["Acadia"] = "lafayette", ["Allen"] = "lakecharles", ["Ascension"] = "batonrouge",
        ["Assumption"] = "houma", ["Avoyelles"] = "alexandria", ["Beauregard"] = "lakecharles",
        ["Bienville"] = "shreveport", ["Bossier"] = "shreveport", ["Caddo"] = "shreveport",
        ["Calcasieu"] = "lakecharles", ["Caldwell"] = "monroe", ["Cameron"] = "lakecharles",
        ["Catahoula"] = "alexandria", ["Claiborne"] = "shreveport", ["Concordia"] = "alexandria",
        ["De Soto"] = "shreveport", ["East Baton Rouge"] = "batonrouge", ["East Carroll"] = "monroe",
        ["East Feliciana"] = "batonrouge", ["Evangeline"] = "lafayette", ["Franklin"] = "monroe",
        ["Grant"] = "alexandria", ["Iberia"] = "lafayette", ["Iberville"] = "batonrouge",
        ["Jackson"] = "monroe", ["Jefferson"] = "neworleans", ["Jefferson Davis"] = "lakecharles",
        ["Lafayette"] = "lafayette", ["Lafourche"] = "houma", ["LaSalle"] = "alexandria",
        ["Lincoln"] = "monroe", ["Livingston"] = "batonrouge", ["Madison"] = "monroe",
        ["Morehouse"] = "monroe", ["Natchitoches"] = "alexandria", ["Orleans"] = "neworleans",
        ["Ouachita"] = "monroe", ["Plaquemines"] = "neworleans", ["Pointe Coupee"] = "batonrouge",
        ["Rapides"] = "alexandria", ["Red River"] = "shreveport", ["Richland"] = "monroe",
        ["Sabine"] = "shreveport", ["St. Bernard"] = "neworleans", ["St. Charles"] = "neworleans",
        ["St. Helena"] = "batonrouge", ["St. James"] = "neworleans", ["St. John the Baptist"] = "neworleans",
        ["St. Landry"] = "lafayette", ["St. Martin"] = "lafayette", ["St. Mary"] = "houma",
        ["St. Tammany"] = "hammond", ["Tangipahoa"] = "hammond", ["Tensas"] = "monroe",
        ["Terrebonne"] = "houma", ["Union"] = "monroe", ["Vermilion"] = "lafayette",
        ["Vernon"] = "alexandria", ["Washington"] = "hammond", ["Webster"] = "shreveport",
        ["West Baton Rouge"] = "batonrouge", ["West Carroll"] = "monroe", ["West Feliciana"] = "batonrouge",
        ["Winn"] = "alexandria",
    };

    public static readonly IReadOnlyList<string> Parishes = [.. ParishToHub.Keys.Order()];

    public static Region? ForParish(string? parish) =>
        parish is not null
        && ParishToHub.TryGetValue(parish, out var hub)
        && Hubs.TryGetValue(hub, out var region)
            ? region
            : null;
}
