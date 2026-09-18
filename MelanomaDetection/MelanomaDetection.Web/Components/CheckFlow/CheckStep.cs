namespace MelanomaDetection.Web.Components.CheckFlow;

/// <summary>
/// The stages of a single skin check, in the order they happen. Confirm and
/// Outline are separate stages but both still count as "getting a usable
/// photo", which is why the stepper collapses them into Capture.
/// </summary>
public enum CheckStep
{
    Spot,
    Capture,
    Confirm,
    Outline,
    Details,
    Results,
}
