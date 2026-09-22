import cv2
import numpy as np
import matplotlib.pyplot as plt
from MelanomaDeterminingStuff.color import pixelate


# ─────────────────────────────────────────────
# ABCD VISUALIZATIONS
# ─────────────────────────────────────────────

def build_abcd_visuals(original, mask, abcde):
    h, w = original.shape[:2]
    visuals = {}

    # ── A: Asymmetry ──────────────────────────────────────────────────────────
    a_img  = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).copy()
    M      = cv2.moments(mask)
    cx     = int(M["m10"] / M["m00"]) if M["m00"] > 0 else w // 2
    cy     = int(M["m01"] / M["m00"]) if M["m00"] > 0 else h // 2

    coords = np.argwhere(mask > 0)
    if len(coords) > 0:
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        half = max(r_max - r_min, c_max - c_min) // 2 + 10
        r0 = max(cy - half, 0); r1 = min(cy + half, h)
        c0 = max(cx - half, 0); c1 = min(cx + half, w)

        crop      = (mask[r0:r1, c0:c1] // 255).astype(np.uint8)
        flip_h    = np.fliplr(crop)
        overlap_h = crop & flip_h
        only_orig = crop & ~flip_h
        only_flip = flip_h & ~crop

        overlay = a_img[r0:r1, c0:c1].copy()
        overlay[overlap_h > 0] = (overlay[overlap_h > 0] * 0.5 + np.array([0,   200, 0  ]) * 0.5).clip(0, 255).astype(np.uint8)
        overlay[only_orig > 0] = (overlay[only_orig > 0] * 0.5 + np.array([220, 50,  50 ]) * 0.5).clip(0, 255).astype(np.uint8)
        overlay[only_flip > 0] = (overlay[only_flip > 0] * 0.5 + np.array([50,  50,  220]) * 0.5).clip(0, 255).astype(np.uint8)
        a_img[r0:r1, c0:c1] = overlay

    a_vis = cv2.cvtColor(a_img, cv2.COLOR_RGB2BGR)
    cv2.circle(a_vis, (cx, cy), max(5, w // 100), (0, 255, 255), -1)
    visuals["A"] = cv2.cvtColor(a_vis, cv2.COLOR_BGR2RGB)

    # ── B: Border ─────────────────────────────────────────────────────────────
    b_vis     = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        color_b = (255, 80, 80) if abcde["B_border"]["concern"] else (80, 255, 80)
        cv2.drawContours(b_vis, contours, -1, color_b, max(2, w // 300))
    visuals["B"] = b_vis

    # ── C: Color heatmap ──────────────────────────────────────────────────────
    lab_full = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    cs       = min(h, w) // 8
    corners  = np.vstack([
        lab_full[:cs,  :cs ].reshape(-1, 3),
        lab_full[:cs,  -cs:].reshape(-1, 3),
        lab_full[-cs:, :cs ].reshape(-1, 3),
        lab_full[-cs:, -cs:].reshape(-1, 3),
    ])
    skin_lab  = np.median(corners, axis=0)
    pix_lab   = cv2.cvtColor(pixelate(original, block_size=12), cv2.COLOR_BGR2LAB).astype(np.float32)
    dist_map  = np.sqrt(np.sum((pix_lab - skin_lab) ** 2, axis=2))
    dist_norm = np.zeros_like(dist_map)
    if np.sum(mask > 0) > 0:
        d_min = dist_map[mask > 0].min()
        d_max = dist_map[mask > 0].max()
        dist_norm[mask > 0] = (dist_map[mask > 0] - d_min) / (d_max - d_min + 1e-6) * 255
    heatmap  = cv2.applyColorMap(dist_norm.astype(np.uint8), cv2.COLORMAP_JET)
    c_vis    = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).copy()
    c_vis[mask > 0] = (
        c_vis[mask > 0] * 0.3 +
        cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)[mask > 0] * 0.7
    ).clip(0, 255).astype(np.uint8)
    visuals["C"] = c_vis

    # ── D: Diameter ───────────────────────────────────────────────────────────
    d_vis = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).copy()
    if contours:
        contour          = max(contours, key=cv2.contourArea)
        (ccx, ccy), rad  = cv2.minEnclosingCircle(contour)
        color_d          = (255, 80, 80) if abcde["D_diameter"]["concern"] else (80, 255, 80)
        cv2.circle(d_vis, (int(ccx), int(ccy)), int(rad),        color_d,        max(2, w // 300))
        cv2.circle(d_vis, (int(ccx), int(ccy)), max(4, w // 150),(255, 255, 0),  -1)
        d_val = abcde["D_diameter"]["value"]
        cv2.putText(d_vis, f"{d_val:.1f}mm" if d_val else "N/A",
                    (int(ccx) - 30, int(ccy) - int(rad) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w / 2000), color_d, max(1, w // 500))
    visuals["D"] = d_vis

    return visuals


# ─────────────────────────────────────────────
# FULL PIPELINE VISUALIZATION
# ─────────────────────────────────────────────

def visualize_pipeline(
    original,
    denoised,
    bilateral,
    no_hair,
    mask,
    edges,
    abcde,
    save_path=None,
):
    def bgr2rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    abcd_visuals = build_abcd_visuals(original, mask, abcde)

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#12121f")

    # Section labels
    fig.text(0.02, 0.985, "── PIPELINE STAGES ──────────────────────────────────────────────────",
             color="#888888", fontsize=8, fontfamily="monospace", ha="left", va="bottom")
    fig.text(0.02, 0.345, "── ABCD ANALYSIS ────────────────────────────────────────────────────",
             color="#888888", fontsize=8, fontfamily="monospace", ha="left", va="bottom")

    # Pipeline stage panels
    pipeline_stages = [
        (original,  "1  Original"),
        (denoised,  "2  Median Filter"),
        (bilateral, "3  Bilateral Filter"),
        (no_hair,   "4  Hair Removal"),
        (mask,      "5  Segmentation"),
        (edges,     "6  Canny Edges"),
    ]
    for i, (img, title) in enumerate(pipeline_stages):
        ax = fig.add_subplot(3, 4, i + 1)
        ax.imshow(img if len(img.shape) == 2 else bgr2rgb(img),
                  cmap="gray" if len(img.shape) == 2 else None)
        ax.set_title(title, color="#cccccc", fontsize=9,
                     fontfamily="monospace", pad=6, loc="left")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")
            spine.set_linewidth(1)

    # Report panel
    ax_r = fig.add_subplot(3, 4, (7, 8))
    ax_r.set_facecolor("#0d1b2a")
    ax_r.axis("off")
    for spine in ax_r.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#1e3a5f")
        spine.set_linewidth(1.5)

    risk    = abcde["_summary"]["risk_level"]
    rc      = {"LOW": "#00e676", "HIGH": "#ff1744"}.get(risk, "white")
    risk_bg = {"LOW": "#003300", "HIGH": "#330000"}.get(risk, "#111")

    ax_r.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, color=risk_bg,
                   transform=ax_r.transAxes, clip_on=False))
    ax_r.text(0.5, 0.94,
              "⚠  MELANOMA RISK: HIGH" if risk == "HIGH" else "✔  LOW RISK",
              color=rc, fontsize=12, fontweight="bold",
              transform=ax_r.transAxes, va="center", ha="center",
              fontfamily="monospace")

    y = 0.80
    for key, letter in [("A_asymmetry","A"),("B_border","B"),("C_color","C"),("D_diameter","D")]:
        item    = abcde[key]
        concern = item["concern"]
        val_col = "#ff4444" if concern else ("#888888" if item["value"] is None else "#44ff88")
        flag    = "⚠" if concern else ("—" if item["value"] is None else "✔")
        flag_col= "#ff4444" if concern else ("#666" if item["value"] is None else "#44ff88")

        ax_r.text(0.05, y, letter,     color="#aaaaaa", fontsize=10, fontweight="bold",
                  transform=ax_r.transAxes, va="top", fontfamily="monospace")
        ax_r.text(0.15, y, item["label"], color=val_col, fontsize=8,
                  transform=ax_r.transAxes, va="top", fontfamily="monospace")
        ax_r.text(0.92, y, flag,       color=flag_col, fontsize=11, fontweight="bold",
                  transform=ax_r.transAxes, va="top", ha="right", fontfamily="monospace")
        ax_r.plot([0.03, 0.97], [y - 0.01, y - 0.01], color="#1e2a3a",
                  linewidth=0.5, transform=ax_r.transAxes)
        y -= 0.155

    ax_r.text(0.5, 0.06, "⚠  Not a medical diagnosis", color="#ffcc44",
              fontsize=7, ha="center", transform=ax_r.transAxes, fontfamily="monospace")
    ax_r.text(0.5, 0.02, "Consult a dermatologist",    color="#ffcc44",
              fontsize=7, ha="center", transform=ax_r.transAxes, fontfamily="monospace")

    # ABCD panels
    abcd_meta = {
        "A": ("A  Asymmetry",  "Green=overlap  Red=original only  Blue=flip only"),
        "B": ("B  Border",     "⚠ Irregular" if abcde["B_border"]["concern"] else "✔ Regular"),
        "C": ("C  Color",      "Heatmap: blue=skin-like  red=different"),
        "D": ("D  Diameter",   abcde["D_diameter"]["label"]),
    }
    for i, key in enumerate(["A", "B", "C", "D"]):
        title, subtitle = abcd_meta[key]
        ax = fig.add_subplot(3, 4, 9 + i)
        ax.imshow(abcd_visuals[key])
        ax.set_title(title,    color="#cccccc", fontsize=9, fontfamily="monospace",
                     pad=4, loc="left", fontweight="bold")
        ax.set_xlabel(subtitle, color="#888888", fontsize=7,
                      fontfamily="monospace", labelpad=3)
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")
            spine.set_linewidth(1)

    plt.suptitle("Melanoma Detection Pipeline", color="white", fontsize=15,
                 fontweight="bold", fontfamily="monospace", y=1.005)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[✔] Result saved to: {save_path}")

    plt.show()
    plt.close(fig)