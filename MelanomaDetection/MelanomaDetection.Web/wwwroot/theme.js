// Theme selection. Runs before first paint (it is loaded synchronously in
// <head>) so a dark-mode viewer never gets a flash of the light palette.
//
// "system" stores nothing and lets the prefers-color-scheme block in app.css
// decide; "light"/"dark" write data-theme on <html>, which that stylesheet
// treats as an override.
(function () {
    const KEY = 'skincheck-theme';
    const root = document.documentElement;
    const themeColorMeta = document.getElementById('theme-color-meta');
    // Mirrors --color-nav-bg's light/dark values in app.css - keep these two in sync if
    // that token ever changes, since a <meta name="theme-color"> can't read CSS variables.
    const THEME_COLOR = { light: '#2A3B31', dark: '#171D22' };
    const darkMedia = window.matchMedia('(prefers-color-scheme: dark)');

    let stored = null;
    try {
        stored = localStorage.getItem(KEY);
    } catch {
        // Private mode or blocked storage: fall back to the OS preference.
    }

    function effectiveTheme() {
        return stored === 'light' || stored === 'dark' ? stored : (darkMedia.matches ? 'dark' : 'light');
    }

    function apply() {
        const want = stored === 'light' || stored === 'dark' ? stored : null;
        if (root.getAttribute('data-theme') !== want) {
            if (want) {
                root.setAttribute('data-theme', want);
            } else {
                root.removeAttribute('data-theme');
            }
        }
        if (themeColorMeta) {
            themeColorMeta.setAttribute('content', THEME_COLOR[effectiveTheme()]);
        }
    }

    apply();

    // The browser's OS-level dark-mode toggle repaints the page on its own via app.css's
    // prefers-color-scheme block, but the theme-color meta tag has no CSS equivalent and
    // only matters when no manual override is stored (an override already fixed the color).
    darkMedia.addEventListener('change', () => {
        if (stored === null) {
            apply();
        }
    });

    // Blazor's enhanced navigation patches the DOM against the server's
    // response, and the server never renders data-theme -- it only exists
    // because this script put it there. Without this, moving between pages
    // silently strips the attribute and the app snaps back to the OS theme
    // while the stored preference still says otherwise. Re-applying on
    // mutation is cheap and idempotent: the guard above stops our own write
    // from re-triggering the observer.
    new MutationObserver(apply).observe(root, {
        attributes: true,
        attributeFilter: ['data-theme'],
    });

    window.skinCheckTheme = {
        get() {
            return stored ?? 'system';
        },
        set(theme) {
            stored = theme === 'system' ? null : theme;
            apply();
            try {
                if (stored) {
                    localStorage.setItem(KEY, stored);
                } else {
                    localStorage.removeItem(KEY);
                }
            } catch {
                // Preference still applies for this page view.
            }
        },
    };
})();
