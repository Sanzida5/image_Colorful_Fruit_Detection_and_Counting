import cv2
import numpy as np
from typing import Dict, Tuple, List
import gradio as gr
import matplotlib.pyplot as plt

COLOR_RANGES_HSV: Dict[str, Tuple[np.ndarray, np.ndarray] | Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = {
    "red": ((np.array([0, 120, 90]), np.array([10, 255, 255])),
            (np.array([170, 120, 90]), np.array([180, 255, 255]))),
    "green": (np.array([35, 60, 60]), np.array([85, 255, 255])),
    "yellow": (np.array([18, 100, 120]), np.array([30, 255, 255])),
    "blue": (np.array([95, 80, 80]), np.array([130, 255, 255])),
}

COLOR_TO_BGR: Dict[str, Tuple[int, int, int]] = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "blue": (255, 0, 0),
}

def preprocess_mask(mask: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    h, w = binary_mask.shape[:2]
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    mask_copy = binary_mask.copy()
    cv2.floodFill(mask_copy, flood, (0, 0), 255)
    inv = cv2.bitwise_not(mask_copy)
    filled = cv2.bitwise_or(binary_mask, inv)
    return filled

def mask_for_color(hsv: np.ndarray, color: str) -> np.ndarray:
    ranges = COLOR_RANGES_HSV[color]
    if color == "red":
        (l1, u1), (l2, u2) = ranges
        return cv2.bitwise_or(cv2.inRange(hsv, l1, u1), cv2.inRange(hsv, l2, u2))
    lower, upper = ranges
    return cv2.inRange(hsv, lower, upper)

def filter_yellow_components(mask_y: np.ndarray, hsv: np.ndarray, mask_g: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(mask_y)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1800:
            continue
        comp = np.zeros_like(mask_y)
        cv2.drawContours(comp, [cnt], -1, 255, -1)
        ys, xs = np.where(comp == 255)
        if ys.size == 0:
            continue
        h_vals = hsv[ys, xs, 0]
        s_vals = hsv[ys, xs, 1]
        v_vals = hsv[ys, xs, 2]
        mean_h = float(np.mean(h_vals))
        mean_s = float(np.mean(s_vals))
        mean_v = float(np.mean(v_vals))
        overlap = cv2.bitwise_and(comp, mask_g)
        overlap_ratio = float(np.count_nonzero(overlap)) / float(area)
        if 18 <= mean_h <= 30 and mean_s >= 110 and mean_v >= 120 and overlap_ratio < 0.4:
            filtered = cv2.bitwise_or(filtered, comp)
    return filtered

def find_and_label(image: np.ndarray, mask: np.ndarray, color_name: str, color_bgr: Tuple[int, int, int], min_area: int = 300) -> int:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    item_index = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        item_index += 1
        cv2.drawContours(image, [contour], -1, color_bgr, 2)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(image, (cx, cy), 6, color_bgr, -1)
        cv2.putText(image, f"{color_name} {item_index}", (cx - 40, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
    return item_index

def split_green_components(image: np.ndarray, mask_g: np.ndarray) -> List[np.ndarray]:
    result_masks: List[np.ndarray] = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_g, connectivity=8)
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area < 3000:
            continue
        comp = np.uint8(labels == lab) * 255
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        max_val = dist.max()
        if max_val <= 0:
            continue
        norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        peaks = cv2.goodFeaturesToTrack(norm, maxCorners=3, qualityLevel=0.06, minDistance=80)
        if peaks is not None and len(peaks) >= 2:
            p1 = tuple(peaks[0][0]); p2 = tuple(peaks[1][0])
            sep = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
        else:
            sep = 0.0
        if sep >= 60 and max_val >= 15:
            _, sure_fg = cv2.threshold(dist, 0.5 * max_val, 255, 0)
            sure_fg = np.uint8(sure_fg)
            sure_bg = cv2.dilate(comp, np.ones((3, 3), np.uint8), iterations=2)
            unknown = cv2.subtract(sure_bg, sure_fg)
            markers = cv2.connectedComponents(sure_fg)[1] + 1
            markers[unknown == 255] = 0
            img3 = cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR)
            cv2.watershed(img3, markers)
            unique = [u for u in np.unique(markers) if u > 1]
            for u in unique:
                region = np.uint8(markers == u) * 255
                if cv2.countNonZero(region) >= 4000:
                    result_masks.append(region)
            if not unique:
                result_masks.append(comp)
        else:
            ys, xs = np.where(comp == 255)
            if len(xs) < 1000:
                result_masks.append(comp)
                continue
            coords = np.float32(np.column_stack((xs, ys)))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            attempts = 3
            ret, labels_k, centers = cv2.kmeans(coords, 2, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
            if ret and centers is not None:
                mask1 = np.zeros_like(comp); mask2 = np.zeros_like(comp)
                cluster1 = coords[labels_k.ravel() == 0].astype(int)
                cluster2 = coords[labels_k.ravel() == 1].astype(int)
                if cluster1.shape[0] > 0:
                    mask1[cluster1[:, 1], cluster1[:, 0]] = 255
                    mask1 = preprocess_mask(mask1)
                if cluster2.shape[0] > 0:
                    mask2[cluster2[:, 1], cluster2[:, 0]] = 255
                    mask2 = preprocess_mask(mask2)
                area1 = cv2.countNonZero(mask1); area2 = cv2.countNonZero(mask2)
                center_sep = float(np.hypot(*(centers[0] - centers[1]))) if centers.shape[0] == 2 else 0.0
                if area1 >= 4000 and area2 >= 4000 and center_sep >= 120:
                    result_masks.extend([mask1, mask2])
                else:
                    result_masks.append(comp)
            else:
                result_masks.append(comp)
    return result_masks

def detect_fruits(rgb_img: np.ndarray):
    image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    dbg: Dict[str, np.ndarray] = {}
    dbg["image_bgr"] = image.copy()

    mask_y_raw = mask_for_color(hsv, "yellow"); dbg["mask_y_raw"] = mask_y_raw
    mask_y = preprocess_mask(mask_y_raw); dbg["mask_y_clean"] = mask_y

    mask_g_raw = mask_for_color(hsv, "green"); dbg["mask_g_raw"] = mask_g_raw
    mask_g = preprocess_mask(mask_g_raw); dbg["mask_g_clean"] = mask_g
    mask_g_filled = fill_holes(mask_g); dbg["mask_g_filled"] = mask_g_filled

    mask_r_raw = mask_for_color(hsv, "red"); dbg["mask_r_raw"] = mask_r_raw
    mask_r = preprocess_mask(mask_r_raw); dbg["mask_r_clean"] = mask_r

    mask_b_raw = mask_for_color(hsv, "blue"); dbg["mask_b_raw"] = mask_b_raw
    mask_b = preprocess_mask(mask_b_raw); dbg["mask_b_clean"] = mask_b

    mask_y_filt = filter_yellow_components(mask_y, hsv, mask_g_filled); dbg["mask_y_filtered"] = mask_y_filt
    yellow_buffer = cv2.dilate(mask_y_filt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1); dbg["yellow_buffer"] = yellow_buffer

    mask_g_final = cv2.bitwise_and(mask_g_filled, cv2.bitwise_not(yellow_buffer)); dbg["mask_g_final"] = mask_g_final
    mask_r_final = cv2.bitwise_and(mask_r, cv2.bitwise_not(cv2.bitwise_or(mask_g_final, yellow_buffer))); dbg["mask_r_final"] = mask_r_final

    dist = cv2.distanceTransform(mask_g_final, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dbg["green_distance"] = dist_norm

    result_img = image.copy()
    counts: Dict[str, int] = {}
    counts["yellow"] = find_and_label(result_img, mask_y_filt, "Yellow", COLOR_TO_BGR["yellow"])

    green_regions = split_green_components(result_img, mask_g_final)
    if not green_regions:
        counts["green"] = find_and_label(result_img, mask_g_final, "Green", COLOR_TO_BGR["green"])
    else:
        counts["green"] = 0
        canvas = np.zeros_like(mask_g_final)
        for reg in green_regions:
            counts["green"] += find_and_label(result_img, reg, "Green", COLOR_TO_BGR["green"])
            canvas = cv2.bitwise_or(canvas, reg)
        dbg["green_regions_merged"] = canvas

    counts["red"] = find_and_label(result_img, mask_r_final, "Red", COLOR_TO_BGR["red"])
    counts["blue"] = find_and_label(result_img, mask_b, "Blue", COLOR_TO_BGR["blue"])

    return result_img, counts, dbg

def to_rgb(img_bgr_or_gray: np.ndarray) -> np.ndarray:
    if img_bgr_or_gray.ndim == 2:
        return cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)

def display_detected_fruits(dbg: Dict[str, np.ndarray], result_bgr: np.ndarray) -> np.ndarray:
    tiles = [
        ("Original", to_rgb(dbg["image_bgr"])),
        ("Yellow raw", to_rgb(dbg["mask_y_raw"])),
        ("Yellow clean", to_rgb(dbg["mask_y_clean"])),
        ("Yellow filtered", to_rgb(dbg["mask_y_filtered"])),
        ("Yellow buffer", to_rgb(dbg["yellow_buffer"])),
        ("Green raw", to_rgb(dbg["mask_g_raw"])),
        ("Green clean", to_rgb(dbg["mask_g_clean"])),
        ("Green filled", to_rgb(dbg["mask_g_filled"])),
        ("Green final", to_rgb(dbg["mask_g_final"])),
        ("Green distance", to_rgb(dbg["green_distance"])),
        ("Red raw", to_rgb(dbg["mask_r_raw"])),
        ("Red final", to_rgb(dbg["mask_r_final"])),
    ]
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    fig = plt.figure(figsize=(16, 5 * (rows + 1)))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(rows + 1, cols, height_ratios=[1] * rows + [1.5])

    for i, (title, img) in enumerate(tiles):
        r = i // cols; c = i % cols
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(img); ax.set_title(title); ax.axis("off")

    ax_big = fig.add_subplot(gs[rows, :])
    ax_big.imshow(to_rgb(result_bgr)); ax_big.set_title("Annotated Result"); ax_big.axis("off")

    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())  # HxWx4 RGBA
    grid_rgb = cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return grid_rgb

def gradio_predict(image: np.ndarray):
    if image is None:
        return {"blue": 0, "green": 0, "red": 0, "yellow": 0}, None
    result_bgr, counts, dbg = detect_fruits(image)
    grid_rgb = display_detected_fruits(dbg, result_bgr)
    return {k: int(v) for k, v in counts.items()}, grid_rgb

def build_ui():
    with gr.Blocks(title="Colorful Fruit Count") as demo:
        gr.Markdown("### Upload an image and click Detect Fruits")
        inp = gr.Image(type="numpy", label="Upload", sources=["upload"])
        btn = gr.Button("Detect Fruits", variant="primary")
        counts = gr.JSON(label="Counts")
        combined = gr.Image(label="Intermediates + Annotated Result", type="numpy")
        btn.click(fn=gradio_predict, inputs=inp, outputs=[counts, combined])
    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(inbrowser=True, server_port=7860)
