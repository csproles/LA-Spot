// Fetches a file and saves it, reporting what actually happened -- unlike a plain
// link, where a failed download only shows up as the browser's own vague error.
export async function download(url, fallbackName, expectedType) {
    let response;
    try {
        response = await fetch(url, {
            credentials: "same-origin",
            headers: { "X-Requested-With": "fetch" },
        });
    } catch {
        return { ok: false, message: "Couldn't reach LA Spot. Check your connection and try again." };
    }

    const type = response.headers.get("Content-Type") || "";
    if (!response.ok || (expectedType && !type.startsWith(expectedType))) {
        return { ok: false, message: await errorMessage(response) };
    }

    const blob = await response.blob();
    const fileName = fileNameFrom(response.headers.get("Content-Disposition")) || fallbackName;
    const href = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = href;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(href), 10000);
    return { ok: true, fileName, size: blob.size };
}

async function errorMessage(response) {
    if (response.status === 429) {
        return "Too many downloads in a row. Wait a minute and try again.";
    }
    if (response.status === 401) {
        return "Your session has ended. Sign in again to download.";
    }
    try {
        const body = await response.json();
        const detail = body.detail || body.error || body.title;
        if (detail) {
            return detail;
        }
    } catch {
        // Not JSON -- fall through to the generic message.
    }
    return "The download couldn't be prepared just now. Try again in a minute.";
}

function fileNameFrom(disposition) {
    if (!disposition) {
        return null;
    }
    const encoded = /filename\*=UTF-8''([^;]+)/i.exec(disposition);
    if (encoded) {
        return decodeURIComponent(encoded[1]);
    }
    const plain = /filename="?([^";]+)"?/i.exec(disposition);
    return plain ? plain[1] : null;
}
