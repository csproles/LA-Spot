// Theme selection. Runs before first paint (it is loaded synchronously in
// <head>) so a dark-mode viewer never gets a flash of the light palette.
//
// "system" stores nothing and lets the prefers-color-scheme block in app.css
// decide; "light"/"dark" write data-theme on <html>, which that stylesheet
// treats as an override.
(function () {
    const KEY = 'skincheck-theme';
    const root = document.documentElement;

    let stored = null;
    try {
        stored = localStorage.getItem(KEY);
    } catch {
        // Private mode or blocked storage: fall back to the OS preference.
    }

    function apply() {
        const want = stored === 'light' || stored === 'dark' ? stored : null;
        if (root.getAttribute('data-theme') === want) {
            return;
        }
        if (want) {
            root.setAttribute('data-theme', want);
        } else {
            root.removeAttribute('data-theme');
        }
    }

    apply();

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
