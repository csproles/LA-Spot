namespace MelanomaDetection.Web.Services.Account;

/// <summary>
/// Whether the sign-in page offers "Try the demo". Bound from the "Demo"
/// configuration section; when the section is absent it is on in Development
/// and off everywhere else, so a deployment has to opt in to anonymous
/// account creation.
/// </summary>
public sealed class DemoSettings
{
    public const string SectionName = "Demo";

    public bool Enabled { get; set; }

    /// <summary>How long an abandoned demo session lives before the sweeper erases it. Sign-out erases it immediately.</summary>
    public TimeSpan SessionLifetime { get; set; } = TimeSpan.FromHours(8);
}
