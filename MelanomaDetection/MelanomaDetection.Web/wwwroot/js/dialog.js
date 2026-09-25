// Opens a <dialog> as a real modal: focus is kept inside and the page behind becomes inert.
export function showModal(dialog) {
    if (dialog && !dialog.open) {
        dialog.showModal();
    }
}
