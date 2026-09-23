// Positions the sliding indicator under the active tab. The animation itself
// is a CSS transition on .tabs-indicator (see Tabs.razor.css) -- this just
// sets the target transform/width, so no animation library loads from a CDN
// on every page that uses <Tabs>.
//
// Position moves via `transform: translateX()`, not the `left` property:
// `left` forces a layout recalculation on every animation frame, which
// renders choppy or barely visible on plenty of real hardware -- transform
// is GPU-composited. This is also what the original Motion-library version
// of this component animated under the hood (its `x` option is a
// transform, not the CSS `left` property).
export function move(indicator, tab, instant) {
    const x = tab.offsetLeft;
    const width = tab.offsetWidth;

    if (instant) {
        // Skip the transition for the very first paint (nothing to animate
        // from yet), then re-enable it for every later tab switch.
        indicator.style.transitionProperty = 'none';
        indicator.style.transform = `translateX(${x}px)`;
        indicator.style.width = `${width}px`;
        void indicator.offsetWidth; // flush the instant position before re-enabling the transition
        indicator.style.transitionProperty = '';
        return;
    }

    // Deferred to the next frame on purpose: this runs from OnAfterRenderAsync,
    // in the same tick Blazor just patched the DOM for the click that changed
    // Active. Setting the new transform/width in that same tick can get
    // coalesced with Blazor's own patch into a single paint -- the browser
    // never registers a "previous painted value" to transition from, so it
    // jumps straight to the end state instead of animating. One frame of
    // separation gives it an unambiguous before/after to interpolate between.
    requestAnimationFrame(() => {
        indicator.style.transform = `translateX(${x}px)`;
        indicator.style.width = `${width}px`;
    });
}
