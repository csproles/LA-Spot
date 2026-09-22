// "Skip to main content".
//
// The document has <base href="/">, so a plain "#main-content" link resolves to "/#main-content" and would send the
// person to the home page. Blazor's own link handler also treats every <a> click as a navigation. So this listens on
// the window in the capture phase, ahead of Blazor, and moves focus to <main> in place instead.
window.addEventListener(
    "click",
    (event) => {
        const link = event.target.closest("a.skip-link");
        if (!link) {
            return;
        }

        event.preventDefault();
        event.stopImmediatePropagation();

        const main = document.getElementById("main-content");
        if (main) {
            main.focus({ preventScroll: true });
            main.scrollIntoView({ block: "start" });
        }
    },
    true
);
