// Tiny browser-timezone helper for the telehealth booking pages: the server
// generates slot times in UTC, this hands back the visitor's IANA zone name
// so they can be shown in whichever timezone the visitor's device is set to.
window.skinCheckScheduling = {
    getTimeZone: () => Intl.DateTimeFormat().resolvedOptions().timeZone,
};
