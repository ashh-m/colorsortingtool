import argparse
import json
import re
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw


@dataclass
class ExtractedColor:
    code: str
    rgb: tuple[int, int, int]
    pixel_count: int
    bbox: tuple[int, int, int, int]


def rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert an RGB color (0-255) to CIELAB using D65 white point."""
    r, g, b = [channel / 255.0 for channel in rgb]

    def gamma_expand(c: float) -> float:
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92

    r = gamma_expand(r)
    g = gamma_expand(g)
    b = gamma_expand(b)

    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    x, y, z = x / 0.95047, y / 1.00000, z / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t) + (16 / 116)

    fx = f(x)
    fy = f(y)
    fz = f(z)
    l_val = (116 * fy) - 16
    a_val = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    return l_val, a_val, b_val


def _rgb_to_hsv_array(rgb_arr: np.ndarray) -> np.ndarray:
    """Convert Nx3 RGB array in 0-255 to HSV array with H in [0, 1]."""
    rgb = rgb_arr.astype(np.float32) / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    max_c = np.max(rgb, axis=1)
    min_c = np.min(rgb, axis=1)
    delta = max_c - min_c

    h = np.zeros_like(max_c)
    nonzero = delta > 1e-8

    idx = (max_c == r) & nonzero
    h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6
    idx = (max_c == g) & nonzero
    h[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2
    idx = (max_c == b) & nonzero
    h[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4
    h = h / 6.0

    s = np.zeros_like(max_c)
    valid = max_c > 1e-8
    s[valid] = delta[valid] / max_c[valid]
    v = max_c
    return np.stack([h, s, v], axis=1)


def _kmeans(data: np.ndarray, k: int, max_iter: int = 40, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Simple k-means implementation to avoid external dependencies."""
    rng = np.random.default_rng(seed)
    k = min(k, len(data))

    indices = rng.choice(len(data), size=k, replace=False)
    centers = data[indices].copy()

    for _ in range(max_iter):
        dist_sq = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dist_sq, axis=1)

        new_centers = centers.copy()
        for ci in range(k):
            members = data[labels == ci]
            if len(members) == 0:
                new_centers[ci] = data[rng.integers(0, len(data))]
            else:
                new_centers[ci] = members.mean(axis=0)

        if np.allclose(new_centers, centers, atol=1e-3):
            centers = new_centers
            break
        centers = new_centers

    return centers, labels


def _ocr_three_digit_code(label_region_bgr: np.ndarray, ocr_scale: int) -> str | None:
    if label_region_bgr.size == 0:
        return None

    gray = cv2.cvtColor(label_region_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), dtype=np.uint8)
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel, iterations=1)

    if ocr_scale > 1:
        binary_inv = cv2.resize(
            binary_inv,
            (binary_inv.shape[1] * ocr_scale, binary_inv.shape[0] * ocr_scale),
            interpolation=cv2.INTER_CUBIC,
        )

    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(binary_inv, config=config)
    digits = re.findall(r"\d{3}", text)
    if digits:
        return digits[0]

    fallback = re.findall(r"\d+", text)
    if fallback:
        return fallback[0][-3:].zfill(3)
    return None


def extract_colors(
    image_path: str,
    swatch_count: int,
    saturation_threshold: float,
    value_threshold: float,
    sample_limit: int,
    min_area_ratio: float,
    label_band_ratio: float,
    ocr_scale: int,
) -> list[ExtractedColor]:
    """Extract swatch colors and read 3-digit labels under each swatch via OCR."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    h_img, w_img = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    sat_min = int(np.clip(round(saturation_threshold * 255), 0, 255))
    val_min = int(np.clip(round(value_threshold * 255), 0, 255))
    mask = cv2.inRange(hsv, (0, sat_min, val_min), (179, 255, 255))

    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(20.0, min_area_ratio * w_img * h_img)

    candidates: list[ExtractedColor] = []
    used_codes: set[str] = set()
    rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 8 or h < 8:
            continue

        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        points = rgb_img[contour_mask == 255]
        if len(points) == 0:
            continue

        median_rgb = tuple(int(v) for v in np.median(points, axis=0))

        y0 = min(h_img, y + h)
        y1 = min(h_img, int(y + h + h * label_band_ratio))
        x_pad = max(2, int(w * 0.12))
        x0 = max(0, x - x_pad)
        x1 = min(w_img, x + w + x_pad)
        label_crop = bgr[y0:y1, x0:x1]

        detected = _ocr_three_digit_code(label_crop, ocr_scale=ocr_scale)
        if detected is None:
            detected = f"UNK{len(candidates) + 1:03d}"
        while detected in used_codes:
            detected = f"{detected}_DUP"
        used_codes.add(detected)

        candidates.append(
            ExtractedColor(
                code=detected,
                rgb=median_rgb,
                pixel_count=int(area),
                bbox=(x, y, w, h),
            )
        )

    if len(candidates) < 3:
        # Fallback to color clustering when contour extraction fails.
        flat_pixels = rgb_img.reshape(-1, 3)
        hsv_flat = _rgb_to_hsv_array(flat_pixels)
        colorful = flat_pixels[
            (hsv_flat[:, 1] >= saturation_threshold) & (hsv_flat[:, 2] >= value_threshold)
        ]
        if len(colorful) < max(10, swatch_count):
            raise ValueError(
                "Could not detect swatches. Use a clearer image or lower thresholds."
            )
        if len(colorful) > sample_limit:
            rng = np.random.default_rng(11)
            colorful = colorful[rng.choice(len(colorful), size=sample_limit, replace=False)]

        centers, labels = _kmeans(colorful.astype(np.float32), swatch_count)
        counts = np.bincount(labels, minlength=len(centers))
        candidates = []
        for i, (center, count) in enumerate(zip(centers, counts), start=1):
            if count == 0:
                continue
            rgb = tuple(int(np.clip(round(channel), 0, 255)) for channel in center)
            candidates.append(
                ExtractedColor(code=f"UNK{i:03d}", rgb=rgb, pixel_count=int(count), bbox=(0, 0, 0, 0))
            )

    candidates.sort(key=lambda c: c.pixel_count, reverse=True)
    if swatch_count > 0:
        candidates = candidates[:swatch_count]
    return candidates


def _path_length(order: list[int], labs: list[tuple[float, float, float]]) -> float:
    total = 0.0
    for i in range(1, len(order)):
        a = np.array(labs[order[i - 1]])
        b = np.array(labs[order[i]])
        total += float(np.linalg.norm(a - b))
    return total


def _nearest_neighbor_order(labs: list[tuple[float, float, float]]) -> list[int]:
    start_idx = int(np.argmax([lab[1] for lab in labs]))  # most red/magenta as anchor
    unvisited = set(range(len(labs)))
    order = [start_idx]
    unvisited.remove(start_idx)

    while unvisited:
        last = order[-1]
        nearest = min(
            unvisited,
            key=lambda i: np.linalg.norm(np.array(labs[last]) - np.array(labs[i])),
        )
        order.append(nearest)
        unvisited.remove(nearest)

    return order


def _two_opt(order: list[int], labs: list[tuple[float, float, float]], max_passes: int = 5) -> list[int]:
    if len(order) < 4:
        return order

    best = order[:]
    best_len = _path_length(best, labs)

    for _ in range(max_passes):
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                candidate = best[:i] + list(reversed(best[i : j + 1])) + best[j + 1 :]
                candidate_len = _path_length(candidate, labs)
                if candidate_len + 1e-6 < best_len:
                    best = candidate
                    best_len = candidate_len
                    improved = True
        if not improved:
            break
    return best


def sort_rainbow(colors: list[ExtractedColor]) -> list[ExtractedColor]:
    """Sort extracted colors into a smooth perceptual rainbow path."""
    labs = [rgb_to_lab(c.rgb) for c in colors]
    order = _nearest_neighbor_order(labs)
    order = _two_opt(order, labs)
    return [colors[i] for i in order]


def save_json(colors: list[ExtractedColor], output_json: str) -> None:
    payload = []
    for idx, color in enumerate(colors, start=1):
        payload.append(
            {
                "order": idx,
                "code": color.code,
                "rgb": list(color.rgb),
                "hex": "#{:02X}{:02X}{:02X}".format(*color.rgb),
                "pixels": color.pixel_count,
                "bbox": list(color.bbox),
            }
        )
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_preview(colors: list[ExtractedColor], output_image: str, block_w: int = 110, block_h: int = 140) -> None:
    width = len(colors) * block_w
    height = block_h
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for idx, color in enumerate(colors):
        x0 = idx * block_w
        x1 = x0 + block_w
        draw.rectangle([x0, 0, x1, height], fill=color.rgb)

    canvas.save(output_image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract hand-swatch colors from an image and sort them into a "
            "smooth rainbow order while keeping OCR label codes."
        )
    )
    parser.add_argument("image", help="Path to the swatch image file")
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Max number of swatches to keep (0 means keep all detected)",
    )
    parser.add_argument("--out-json", default="sorted_markers.json", help="Output JSON path")
    parser.add_argument("--out-preview", default="sorted_markers_preview.png", help="Output preview PNG path")
    parser.add_argument(
        "--sat-threshold",
        type=float,
        default=0.18,
        help="Minimum saturation for considering a pixel part of a swatch (0-1)",
    )
    parser.add_argument(
        "--value-threshold",
        type=float,
        default=0.15,
        help="Minimum value/brightness for considering a pixel part of a swatch (0-1)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=80000,
        help="Max colorful pixels used for clustering (faster on very large images)",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.0006,
        help="Minimum contour area ratio to treat as a swatch",
    )
    parser.add_argument(
        "--label-band-ratio",
        type=float,
        default=0.75,
        help="Label OCR region height below each swatch as a ratio of swatch height",
    )
    parser.add_argument(
        "--ocr-scale",
        type=int,
        default=4,
        help="Upscale factor used before OCR",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.count < 0:
        raise ValueError("--count must be >= 0")
    if not (0.0 <= args.sat_threshold <= 1.0):
        raise ValueError("--sat-threshold must be between 0 and 1")
    if not (0.0 <= args.value_threshold <= 1.0):
        raise ValueError("--value-threshold must be between 0 and 1")
    if args.sample_limit < max(args.count, 2):
        raise ValueError("--sample-limit must be >= --count")
    if args.ocr_scale < 1:
        raise ValueError("--ocr-scale must be >= 1")

    extracted = extract_colors(
        image_path=args.image,
        swatch_count=args.count if args.count > 0 else 10_000,
        saturation_threshold=args.sat_threshold,
        value_threshold=args.value_threshold,
        sample_limit=args.sample_limit,
        min_area_ratio=args.min_area_ratio,
        label_band_ratio=args.label_band_ratio,
        ocr_scale=args.ocr_scale,
    )
    ordered = sort_rainbow(extracted)

    save_json(ordered, args.out_json)
    save_preview(ordered, args.out_preview)

    print(f"Extracted {len(ordered)} colors from {args.image}")
    print(f"Saved sorted palette JSON: {args.out_json}")
    print(f"Saved sorted palette preview: {args.out_preview}")


if __name__ == "__main__":
    main()
