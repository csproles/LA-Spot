// Crop box for PhotoCropper.razor. The box lives entirely in the browser -- dragging
// and resizing never round-trip to the server -- and only the finished crop goes back
// to .NET, as a Uint8Array (received as an IJSStreamReference: a Blazor Server circuit
// takes at most 32 KB per message from the browser, far less than a photo).
//
// The box is kept as fractions (0-1) of the displayed image, so it stays put when the
// layout reflows, and is mapped to the image's natural pixels only when cropping.

const MIN_FRACTION = 0.08;      // smallest box, as a share of each side
const MIN_OUTPUT_PX = 128;      // smallest crop the analysis can do anything useful with
const JPEG_QUALITIES = [0.92, 0.85, 0.75, 0.6];
const KEY_STEP = 0.02;

const states = new WeakMap();

function clamp(value, low, high) {
    return Math.min(Math.max(value, low), high);
}

function render(state) {
    const { box, rect } = state;
    box.style.left = `${rect.x * 100}%`;
    box.style.top = `${rect.y * 100}%`;
    box.style.width = `${rect.w * 100}%`;
    box.style.height = `${rect.h * 100}%`;
}

function normalise(rect) {
    const w = clamp(rect.w, MIN_FRACTION, 1);
    const h = clamp(rect.h, MIN_FRACTION, 1);
    return { x: clamp(rect.x, 0, 1 - w), y: clamp(rect.y, 0, 1 - h), w, h };
}

function onPointerDown(state, event) {
    if (event.button !== undefined && event.button !== 0) {
        return;
    }
    const handle = event.target.closest('[data-handle]')?.dataset.handle ?? 'move';
    const bounds = state.frame.getBoundingClientRect();
    state.drag = { handle, startX: event.clientX, startY: event.clientY, start: { ...state.rect }, bounds };
    state.box.setPointerCapture(event.pointerId);
    event.preventDefault();
}

function onPointerMove(state, event) {
    const drag = state.drag;
    if (!drag) {
        return;
    }
    const dx = (event.clientX - drag.startX) / drag.bounds.width;
    const dy = (event.clientY - drag.startY) / drag.bounds.height;
    const s = drag.start;
    let { x, y, w, h } = s;

    if (drag.handle === 'move') {
        x = clamp(s.x + dx, 0, 1 - s.w);
        y = clamp(s.y + dy, 0, 1 - s.h);
    } else {
        if (drag.handle.includes('w')) {
            x = clamp(s.x + dx, 0, s.x + s.w - MIN_FRACTION);
            w = s.x + s.w - x;
        }
        if (drag.handle.includes('e')) {
            w = clamp(s.w + dx, MIN_FRACTION, 1 - s.x);
        }
        if (drag.handle.includes('n')) {
            y = clamp(s.y + dy, 0, s.y + s.h - MIN_FRACTION);
            h = s.y + s.h - y;
        }
        if (drag.handle.includes('s')) {
            h = clamp(s.h + dy, MIN_FRACTION, 1 - s.y);
        }
    }

    state.rect = { x, y, w, h };
    render(state);
}

function onPointerUp(state, event) {
    if (state.drag && state.box.hasPointerCapture(event.pointerId)) {
        state.box.releasePointerCapture(event.pointerId);
    }
    state.drag = null;
}

// Arrow keys move the box; Shift + arrows resize it from the bottom-right corner.
function onKeyDown(state, event) {
    const moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    const move = moves[event.key];
    if (!move) {
        return;
    }
    event.preventDefault();
    const [mx, my] = move;
    const r = state.rect;
    state.rect = event.shiftKey
        ? normalise({ x: r.x, y: r.y, w: Math.min(r.w + mx * KEY_STEP, 1 - r.x), h: Math.min(r.h + my * KEY_STEP, 1 - r.y) })
        : normalise({ ...r, x: r.x + mx * KEY_STEP, y: r.y + my * KEY_STEP });
    render(state);
}

function waitForImage(image) {
    if (image.complete && image.naturalWidth) {
        return Promise.resolve();
    }
    return new Promise((resolve) => {
        image.addEventListener('load', resolve, { once: true });
        image.addEventListener('error', resolve, { once: true });
    });
}

// Sets up the crop box inside frame (which holds the <img> and the .crop-box).
// Starts at the middle 80% of the photo.
export async function init(frame) {
    const image = frame.querySelector('img');
    const box = frame.querySelector('.crop-box');
    await waitForImage(image);

    const state = { frame, image, box, rect: { x: 0.1, y: 0.1, w: 0.8, h: 0.8 }, drag: null };
    state.handlers = {
        down: (e) => onPointerDown(state, e),
        move: (e) => onPointerMove(state, e),
        up: (e) => onPointerUp(state, e),
        key: (e) => onKeyDown(state, e),
    };
    box.addEventListener('pointerdown', state.handlers.down);
    box.addEventListener('pointermove', state.handlers.move);
    box.addEventListener('pointerup', state.handlers.up);
    box.addEventListener('pointercancel', state.handlers.up);
    box.addEventListener('keydown', state.handlers.key);
    states.set(frame, state);
    render(state);
}

export function dispose(frame) {
    const state = states.get(frame);
    if (!state) {
        return;
    }
    const { box, handlers } = state;
    box.removeEventListener('pointerdown', handlers.down);
    box.removeEventListener('pointermove', handlers.move);
    box.removeEventListener('pointerup', handlers.up);
    box.removeEventListener('pointercancel', handlers.up);
    box.removeEventListener('keydown', handlers.key);
    states.delete(frame);
}

function toJpeg(canvas, quality) {
    return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality));
}

// Crops the selected area at the photo's full resolution to JPEG bytes that fit maxBytes.
// Returns { ok, message }; on success the bytes wait for takeResult (a separate call, so
// they travel as a stream rather than inside this small JSON reply).
export async function crop(frame, maxBytes) {
    const state = states.get(frame);
    if (!state) {
        return { ok: false, message: "The crop tool isn't ready yet. Try again in a moment." };
    }

    const { image, rect } = state;
    const sx = Math.round(rect.x * image.naturalWidth);
    const sy = Math.round(rect.y * image.naturalHeight);
    const sw = Math.round(rect.w * image.naturalWidth);
    const sh = Math.round(rect.h * image.naturalHeight);
    if (sw < MIN_OUTPUT_PX || sh < MIN_OUTPUT_PX) {
        return { ok: false, message: 'That area is too small to analyze. Make the box bigger.' };
    }

    const canvas = document.createElement('canvas');
    canvas.width = sw;
    canvas.height = sh;
    canvas.getContext('2d').drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh);

    for (const quality of JPEG_QUALITIES) {
        const blob = await toJpeg(canvas, quality);
        if (!blob) {
            break;
        }
        if (!maxBytes || blob.size <= maxBytes) {
            state.result = new Uint8Array(await blob.arrayBuffer());
            return { ok: true };
        }
    }
    return { ok: false, message: "The cropped photo couldn't be saved. Try a smaller area." };
}

// The bytes the last successful crop produced (then forgotten), or an empty array --
// never null, which .NET can't read as a stream reference.
export function takeResult(frame) {
    const state = states.get(frame);
    const result = state?.result ?? new Uint8Array(0);
    if (state) {
        state.result = null;
    }
    return result;
}
