"""
utils/image_processing.py
Pipeline pengolahan citra untuk Smart Rice Quality Analyzer.

PERBAIKAN v5 — adaptive background removal + data-driven labeling:

MASALAH SEBELUMNYA:
- Pipeline memakai Otsu threshold → tidak bisa membedakan background hijau vs objek
- Threshold color_mean statis tidak cocok untuk berbagai background
- Solidity beras di atas background hijau bervariasi (0.41–0.98) karena cluster

SOLUSI v5:
1. Auto-detect background (gelap atau berwarna) via HSV
2. Jika background berwarna: gunakan color-based masking (inRange HSV)
3. Jika background gelap/netral: gunakan Otsu seperti biasa
4. Labeling berbasis data nyata:
   - BATU/KOTORAN GELAP: gray_mean < 130 → selalu impurity
   - SEKAM/GABAH      : aspect_ratio > 4.0 AND solidity < 0.80
   - BERAS UTUH       : gray_mean > 160, solidity > 0.88, AR 1.0–3.5
   - BERAS PATAH      : gray_mean > 160, tidak memenuhi whole
   - SISANYA          : impurity
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ─────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────
@dataclass
class GrainFeatures:
    contour: np.ndarray
    area: float
    perimeter: float
    aspect_ratio: float
    extent: float
    solidity: float
    circularity: float = 0.0
    label: str = "unknown"
    color_mean: float = 0.0
    color_hsv: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0))


# ─────────────────────────────────────────────
# STEP 1 — Preprocessing
# ─────────────────────────────────────────────
def preprocess(image_bgr: np.ndarray, target_size: int = 800) -> dict:
    h, w = image_bgr.shape[:2]
    scale = target_size / max(h, w)
    resized = cv2.resize(
        image_bgr, (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(blurred)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    return {
        "original_bgr": resized,
        "gray": gray,
        "blurred": blurred,
        "enhanced": enhanced,
        "hsv": hsv,
    }


# ─────────────────────────────────────────────
# STEP 2 — Background Detection & Segmentasi
# ─────────────────────────────────────────────
def _detect_background_type(hsv: np.ndarray) -> Tuple[str, Optional[np.ndarray]]:
    """
    Deteksi tipe background:
    - 'dark'  : background gelap/hitam → pakai Otsu
    - 'color' : background berwarna (hijau, biru, dll) → pakai HSV masking

    Return: (type, bg_mask) — bg_mask = area background (putih = background)
    """
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    # Cek apakah ada dominansi warna tertentu (saturation tinggi)
    # Background gelap: value rendah (<50) dan saturation rendah
    dark_mask = (v_ch < 50) & (s_ch < 60)
    dark_ratio = np.sum(dark_mask) / dark_mask.size

    if dark_ratio > 0.20:
        return "dark", None

    # Cek background berwarna dominan
    # Hitung histogram hue untuk area dengan saturation tinggi
    colored_mask = s_ch > 60
    colored_ratio = np.sum(colored_mask) / colored_mask.size

    if colored_ratio > 0.20:
        # Ada background berwarna — cari hue dominan
        hue_vals = h_ch[colored_mask]
        if len(hue_vals) == 0:
            return "dark", None

        # Histogram hue 0-180
        hist, bins = np.histogram(hue_vals, bins=18, range=(0, 180))
        dominant_bin = np.argmax(hist)
        dominant_hue = dominant_bin * 10  # tengah bin

        # Buat mask background berdasarkan hue dominan ± toleransi
        tol = 20
        h_low = max(0, dominant_hue - tol)
        h_high = min(180, dominant_hue + tol)
        lower = np.array([h_low, 60, 30])
        upper = np.array([h_high, 255, 255])
        bg_mask = cv2.inRange(hsv, lower, upper)
        return "color", bg_mask

    return "dark", None


def segment(enhanced: np.ndarray, hsv: np.ndarray = None) -> dict:
    """
    Segmentasi adaptive:
    - Background berwarna → HSV color masking (lebih akurat)
    - Background gelap    → Otsu thresholding
    """
    method = "otsu_std"
    bg_mask = None

    if hsv is not None:
        bg_type, bg_mask = _detect_background_type(hsv)
    else:
        bg_type = "dark"

    if bg_type == "color" and bg_mask is not None:
        # Foreground = bukan background
        binary = cv2.bitwise_not(bg_mask)
        method = "hsv_color_removal"
    else:
        # Otsu
        _, binary_inv = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        _, binary_std = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        ratio_inv = np.sum(binary_inv == 255) / binary_inv.size
        ratio_std = np.sum(binary_std == 255) / binary_std.size

        if 0.05 <= ratio_std <= 0.65:
            binary, method = binary_std, "otsu_std"
        elif 0.05 <= ratio_inv <= 0.65:
            binary, method = binary_inv, "otsu_inv"
        else:
            binary = cv2.adaptiveThreshold(
                enhanced, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            method = "adaptive"

    return {"binary": binary, "segment_method": method, "bg_mask": bg_mask}


# ─────────────────────────────────────────────
# STEP 3 — Morfologi
# ─────────────────────────────────────────────
def morphology(binary: np.ndarray) -> dict:
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel_open,  iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    return {"morphed": closed}


# ─────────────────────────────────────────────
# STEP 4 — Feature Extraction
# ─────────────────────────────────────────────
def _extract_single_contour(
    cnt: np.ndarray,
    gray: np.ndarray,
    hsv: np.ndarray,
    bg_mask: Optional[np.ndarray] = None,
) -> Optional[GrainFeatures]:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0 or area == 0:
        return None

    x, y, bw, bh = cv2.boundingRect(cnt)
    aspect_ratio = float(max(bw, bh)) / float(min(bw, bh) + 1e-6)
    extent       = float(area) / (bw * bh + 1e-6)
    hull         = cv2.convexHull(cnt)
    hull_area    = cv2.contourArea(hull)
    solidity     = float(area) / (hull_area + 1e-6)
    circularity  = min(float(4 * np.pi * area) / (perimeter ** 2 + 1e-6), 1.0)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    # Jika ada bg_mask, exclude pixel background dari perhitungan warna
    # agar warna kontur yang tumpang tindih background tidak bias
    if bg_mask is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(bg_mask))
        if np.sum(mask) < 10:
            # Hampir semua pixel adalah background → skip
            return None

    color_mean = float(cv2.mean(gray, mask=mask)[0])
    h_m = float(cv2.mean(hsv[:, :, 0], mask=mask)[0])
    s_m = float(cv2.mean(hsv[:, :, 1], mask=mask)[0])
    v_m = float(cv2.mean(hsv[:, :, 2], mask=mask)[0])

    return GrainFeatures(
        contour=cnt, area=area, perimeter=perimeter,
        aspect_ratio=aspect_ratio, extent=extent,
        solidity=solidity, circularity=circularity,
        color_mean=color_mean, color_hsv=(h_m, s_m, v_m),
    )


def _watershed_split_contours(
    cluster_mask: np.ndarray,
    original_bgr: np.ndarray,
    min_area: int,
) -> list:
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dist      = cv2.distanceTransform(cluster_mask, cv2.DIST_L2, 5)
    dist_smooth = cv2.GaussianBlur(dist, (5, 5), 0)
    dist_max  = dist_smooth.max()
    if dist_max < 2.0:
        return []

    best_contours = []
    for tf in [0.4, 0.5, 0.6]:
        thresh    = max(1.5, dist_max * tf)
        dk        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        local_max = cv2.dilate(dist_smooth, dk)
        sure_fg   = np.uint8((dist_smooth == local_max) & (dist > thresh)) * 255
        sure_fg   = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel, iterations=1)
        sure_bg   = cv2.dilate(cluster_mask, kernel, iterations=3)
        unknown   = cv2.subtract(sure_bg, sure_fg)

        n_labels, markers = cv2.connectedComponents(sure_fg)
        if n_labels <= 1:
            continue

        markers = np.int32(markers + 1)
        markers[unknown == 255] = 0
        markers = cv2.watershed(original_bgr, markers)

        found = []
        for lid in range(2, n_labels + 1):
            sm = np.uint8(markers == lid) * 255
            cs, _ = cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                if cv2.contourArea(c) >= min_area:
                    found.append(c)

        if len(found) > len(best_contours):
            best_contours = found

    return best_contours


def _split_large_cluster(
    cnt, morphed, original_bgr, gray, hsv, bg_mask, min_area, max_area, depth=0
) -> list:
    if depth > 5:
        return []
    cm = np.zeros(morphed.shape, dtype=np.uint8)
    cv2.drawContours(cm, [cnt], -1, 255, -1)
    subs = _watershed_split_contours(cm, original_bgr, min_area)

    grains = []
    for sc in subs:
        a = cv2.contourArea(sc)
        if a < min_area:
            continue
        if a > max_area:
            grains.extend(_split_large_cluster(
                sc, morphed, original_bgr, gray, hsv, bg_mask, min_area, max_area, depth+1
            ))
        else:
            f = _extract_single_contour(sc, gray, hsv, bg_mask)
            if f is not None:
                grains.append(f)
    return grains


def extract_features(
    morphed: np.ndarray,
    original_bgr: np.ndarray,
    gray: np.ndarray,
    hsv: np.ndarray,
    bg_mask: Optional[np.ndarray] = None,
    min_area: int = None,
    max_area: int = 80000,
) -> list:
    if min_area is None:
        image_area = gray.shape[0] * gray.shape[1]
        min_area = max(100, int(image_area * 0.0001))

    contours, _ = cv2.findContours(
        morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    grains = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if area > max_area:
            grains.extend(_split_large_cluster(
                cnt, morphed, original_bgr, gray, hsv, bg_mask, min_area, max_area
            ))
            continue
        f = _extract_single_contour(cnt, gray, hsv, bg_mask)
        if f is not None:
            grains.append(f)
    return grains


# ─────────────────────────────────────────────
# STEP 5 — Labeling (data-driven v5)
# ─────────────────────────────────────────────
def label_grains_rule_based(grains: list) -> list:
    """
    Labeling berbasis data nyata dari dua kondisi foto:

    FOTO BACKGROUND GELAP (gray bg < 50):
    ─────────────────────────────────────
      Beras      : gray 179–186, S 3–8,  solidity 0.96–0.98
      Batu/kotoran: gray 128–167, S 8–101, solidity 0.51–0.97

    FOTO BACKGROUND HIJAU (HSV color removal):
    ─────────────────────────────────────────
      Beras      : gray 187–200, S 26–60, solidity 0.41–0.98*
      Batu/kotoran: gray 35–95,  S 52–212, solidity 0.70–0.94
      (*solidity rendah karena cluster belum terpisah sempurna)

    LOGIKA FINAL (berlaku untuk kedua kondisi):
    ──────────────────────────────────────────
    IMPURITY:
      1. gray_mean < 130                      → gelap pekat = batu/kotoran
      2. gray 130–160 AND saturation > 40     → abu-abu dengan warna = batu abu
      3. aspect_ratio > 4.0 AND solidity<0.78 → panjang tak beraturan = sekam
      4. solidity < 0.60                      → bentuk sangat tidak beraturan

    WHOLE:
      1. gray_mean > 160
      2. solidity > 0.88
      3. aspect_ratio 1.0–3.5
      4. extent > 0.52

    BROKEN:
      • gray_mean > 160 tapi tidak memenuhi syarat whole
      • (beras patah: lebih kecil, solidity sedikit lebih rendah)
    """
    if not grains:
        return grains

    for g in grains:
        _, s, v = g.color_hsv

        # ── IMPURITY ──────────────────────────────────────────────────────
        is_very_dark    = g.color_mean < 130
        is_gray_stone   = (130 <= g.color_mean <= 160) and (s > 40)
        is_husk         = (g.aspect_ratio > 4.0) and (g.solidity < 0.78)
        is_irregular    = g.solidity < 0.60

        if is_very_dark or is_gray_stone or is_husk or is_irregular:
            g.label = "impurity"

        # ── WHOLE ─────────────────────────────────────────────────────────
        elif (
            g.color_mean > 160
            and g.solidity > 0.88
            and 1.0 <= g.aspect_ratio <= 3.5
            and g.extent > 0.52
        ):
            g.label = "whole"

        # ── BROKEN ────────────────────────────────────────────────────────
        else:
            g.label = "broken"

    return grains


# ─────────────────────────────────────────────
# STEP 6 — Visualisasi
# ─────────────────────────────────────────────
COLOR_MAP = {
    "whole":    (0, 200, 0),
    "broken":   (0, 100, 255),
    "impurity": (0, 0, 220),
    "unknown":  (200, 200, 0),
}


def draw_annotated(original_bgr: np.ndarray, grains: list) -> np.ndarray:
    canvas = original_bgr.copy()
    for g in grains:
        color = COLOR_MAP.get(g.label, (200, 200, 0))
        cv2.drawContours(canvas, [g.contour], -1, color, 2)
        M = cv2.moments(g.contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(canvas, g.label[0].upper(), (cx - 5, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return canvas


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────
def grains_to_feature_array(grains: list) -> np.ndarray:
    if not grains:
        return np.zeros((0, 8), dtype=np.float32)
    return np.array([
        [g.area, g.perimeter, g.aspect_ratio, g.extent,
         g.solidity, g.circularity, g.color_hsv[1], g.color_hsv[2]]
        for g in grains
    ], dtype=np.float32)


# ─────────────────────────────────────────────
# Watershed untuk butir menempel
# ─────────────────────────────────────────────
def separate_touching_grains(
    binary: np.ndarray,
    original_bgr: np.ndarray,
    min_dist: float = 3.0,
    dilate_size: int = 15,
) -> np.ndarray:
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    dist    = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_s  = cv2.GaussianBlur(dist, (7, 7), 0)

    dk_size = max(3, dilate_size) | 1
    dk      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk_size, dk_size))
    lmax    = cv2.dilate(dist_s, dk)
    sure_fg = np.uint8((dist_s == lmax) & (dist > min_dist)) * 255
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel, iterations=1)

    unknown  = cv2.subtract(sure_bg, sure_fg)
    _, marks = cv2.connectedComponents(sure_fg)
    marks    = np.int32(marks + 1)
    marks[unknown == 255] = 0
    marks    = cv2.watershed(original_bgr, marks)

    result   = binary.copy()
    bnd      = cv2.dilate(np.uint8(marks == -1) * 255, kernel, iterations=1)
    result[bnd == 255] = 0
    result   = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
    return result


# ─────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────
def run_pipeline(
    image_bgr: np.ndarray,
    use_watershed: bool = False,
    min_dist: float = 3.0,
    dilate_size: int = 15,
    min_area: int = None,
    max_area: int = 80000,
) -> dict:
    pre   = preprocess(image_bgr)
    seg   = segment(pre["enhanced"], pre["hsv"])   # <-- pass HSV untuk auto-detect bg
    morph = morphology(seg["binary"])

    binary_for_contours = morph["morphed"]
    if use_watershed:
        binary_for_contours = separate_touching_grains(
            binary_for_contours,
            pre["original_bgr"],
            min_dist=min_dist,
            dilate_size=dilate_size,
        )

    bg_mask = seg.get("bg_mask")

    grains = extract_features(
        binary_for_contours,
        pre["original_bgr"],
        pre["gray"],
        pre["hsv"],
        bg_mask=bg_mask,
        min_area=min_area,
        max_area=max_area,
    )
    grains = label_grains_rule_based(grains)
    annotated = draw_annotated(pre["original_bgr"], grains)

    label_counts = {"whole": 0, "broken": 0, "impurity": 0, "unknown": 0}
    for g in grains:
        label_counts[g.label] = label_counts.get(g.label, 0) + 1

    return {
        **pre,
        **seg,
        **morph,
        "morphed_separated": binary_for_contours,
        "grains": grains,
        "annotated": annotated,
        "feature_array": grains_to_feature_array(grains),
        "label_counts": label_counts,
        "total_grains": len(grains),
        "bg_type": seg.get("segment_method", "unknown"),
    }