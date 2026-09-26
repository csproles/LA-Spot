// The coin circle for CoinScaleMarker.razor. Dragging happens entirely in the browser; .NET
// only asks for the finished measurement. The circle is kept in the photo's natural pixels
// (the same pixels the analysis measures the spot in) and drawn as percentages of the image,
// so it stays on the coin when the layout reflows.

const MIN_DIAMETER_PX = 20;   // matches CoinScale.MinDiameterPx and validation.py
const START_FRACTION = 0.24;  // starting diameter, as a share of the photo's short side
const KEY_STEP = 0.01;        // arrow-key step, as a share of the photo's short side

const states = new WeakMap();

function clamp(value, low, high) {
    return Math.min(Math.max(value, low), high);
}

function limits(state) {
    const short = Math.min(state.width, state.height);
    return { minR: Math.min(MIN_DIAMETER_PX, short) / 2, maxR: short / 2 };
}

// Keep the circle whole and inside the photo.
function settle(state, cx, cy, r) {
    const { minR, maxR } = limits(state);
    r = clamp(r, minR, maxR);
    state.circle = {
        cx: clamp(cx, r, state.width - r),
        cy: clamp(cy, r, state.height - r),
        r,
    };
}

function render(state) {
    const { circle, width, height, ring } = state;
    const diameter = Math.round(circle.r * 2);
    ring.style.left = `${((circle.cx - circle.r) / width) * 100}%`;
    ring.style.top = `${((circle.cy - circle.r) / height) * 100}%`;
    ring.style.width = `${((circle.r * 2) / width) * 100}%`;
    ring.style.height = `${((circle.r * 2) / height) * 100}%`;
    ring.setAttribute('aria-valuenow', String(diameter));
    ring.setAttribute('aria-valuetext', `${diameter} pixels across`);
}

// Screen pixels to the photo's natural pixels.
function scaleOf(state) {
    const bounds = state.image.getBoundingClientRect();
    return { bounds, sx: state.width / bounds.width, sy: state.height / bounds.height };
}

function onPointerDown(state, event) {
    if (event.button !== undefined && event.button !== 0) {
        return;
    }
    const resizing = Boolean(event.target.closest('[data-handle]'));
    state.drag = { resizing, startX: event.clientX, startY: event.clientY, start: { ...state.circle }, ...scaleOf(state) };
    state.ring.setPointerCapture(event.pointerId);
    state.ring.focus({ preventScroll: true });
    event.preventDefault();
}

function onPointerMove(state, event) {
    const drag = state.drag;
    if (!drag) {
        return;
    }
    const s = drag.start;
    if (drag.resizing) {
        // The edge follows the pointer: the radius is its distance from the centre.
        const px = (event.clientX - drag.bounds.left) * drag.sx;
        const py = (event.clientY - drag.bounds.top) * drag.sy;
        settle(state, s.cx, s.cy, Math.hypot(px - s.cx, py - s.cy));
    } else {
        settle(state, s.cx + (event.clientX - drag.startX) * drag.sx, s.cy + (event.clientY - drag.startY) * drag.sy, s.r);
    }
    render(state);
}

function onPointerUp(state, event) {
    if (state.drag && state.ring.hasPointerCapture(event.pointerId)) {
        state.ring.releasePointerCapture(event.pointerId);
    }
    state.drag = null;
}

// Arrow keys move the circle. + and - (or Shift + Up / Shift + Down) resize it.
function onKeyDown(state, event) {
    const step = Math.min(state.width, state.height) * KEY_STEP;
    const { cx, cy, r } = state.circle;
    const moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    let next = null;

    if (event.key === '+' || event.key === '=' || (event.shiftKey && event.key === 'ArrowUp')) {
        next = [cx, cy, r + step];
    } else if (event.key === '-' || event.key === '_' || (event.shiftKey && event.key === 'ArrowDown')) {
        next = [cx, cy, r - step];
    } else if (moves[event.key]) {
        const [mx, my] = moves[event.key];
        next = [cx + mx * step * 2, cy + my * step * 2, r];
    }

    if (next) {
        event.preventDefault();
        settle(state, ...next);
        render(state);
    }
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

// Puts the circle in the middle of the photo inside frame (which holds the <img> and the
// .coin-ring). Safe to call again when the photo changes, after a crop or an undo.
// Returns false when the photo could not be loaded.
export async function init(frame) {
    dispose(frame);
    const image = frame.querySelector('img');
    const ring = frame.querySelector('.coin-ring');
    await waitForImage(image);
    if (!image.naturalWidth || !image.naturalHeight) {
        return false;
    }

    const state = { frame, image, ring, width: image.naturalWidth, height: image.naturalHeight, drag: null };
    const short = Math.min(state.width, state.height);
    settle(state, state.width / 2, state.height / 2, (short * START_FRACTION) / 2);

    state.handlers = {
        down: (e) => onPointerDown(state, e),
        move: (e) => onPointerMove(state, e),
        up: (e) => onPointerUp(state, e),
        key: (e) => onKeyDown(state, e),
    };
    ring.addEventListener('pointerdown', state.handlers.down);
    ring.addEventListener('pointermove', state.handlers.move);
    ring.addEventListener('pointerup', state.handlers.up);
    ring.addEventListener('pointercancel', state.handlers.up);
    ring.addEventListener('keydown', state.handlers.key);
    ring.setAttribute('aria-valuemin', String(Math.round(limits(state).minR * 2)));
    ring.setAttribute('aria-valuemax', String(Math.round(short)));
    states.set(frame, state);
    render(state);
    return true;
}

export function dispose(frame) {
    const state = states.get(frame);
    if (!state) {
        return;
    }
    const { ring, handlers } = state;
    ring.removeEventListener('pointerdown', handlers.down);
    ring.removeEventListener('pointermove', handlers.move);
    ring.removeEventListener('pointerup', handlers.up);
    ring.removeEventListener('pointercancel', handlers.up);
    ring.removeEventListener('keydown', handlers.key);
    states.delete(frame);
}

// The circle's diameter in the photo's own pixels, or null when it is not set up.
export function measure(frame) {
    const state = states.get(frame);
    return state ? { diameterPx: state.circle.r * 2 } : null;
}
