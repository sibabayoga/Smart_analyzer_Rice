"""
utils/image_processing.py
Pipeline pengolahan citra untuk Smart Rice Quality Analyzer.
Tahapan: Preprocessing → Segmentasi → Morfologi → Contour Analysis → Feature Extraction
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Data class hasil analisis per butir
# ─────────────────────────────────────────────
@dataclass
class GrainFeatures:
    contour: np.ndarray
    area: float
    perimeter: float
    aspect_ratio: float
    extent: float
    solidity: float
    label: str = "unknown"   # "whole", "broken", "impurity"
    color_mean: float = 0.0  # mean brightness dari region butir


# ─────────────────────────────────────────────
# STEP 1 — Preprocessing
# ─────────────────────────────────────────────
def preprocess(image_bgr: np.ndarray, target_size: int = 800) -> dict:
    """
    Resize → Grayscale → Gaussian Blur → Contrast Enhancement (CLAHE)
    Returns dict berisi semua intermediate hasil untuk ditampilkan di UI.
    """
    # Resize: jaga aspect ratio, panjang sisi terpanjang = target_size
    h, w = image_bgr.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Noise reduction — Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Contrast enhancement — CLAHE (lebih baik dari global histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    return {
        "original_bgr": resized,
        "gray": gray,
        "blurred": blurred,
        "enhanced": enhanced,
    }


# ─────────────────────────────────────────────
# STEP 2 — Segmentasi (Otsu's Thresholding)
# ─────────────────────────────────────────────
def segment(enhanced: np.ndarray) -> dict:
    """
    Otsu's Thresholding dengan auto-detect arah.
    Deteksi apakah butir lebih terang (beras putih di background gelap)
    atau lebih gelap (background terang) → pilih arah threshold yang tepat.
    Hasil: butir = PUTIH (255), background = HITAM (0).
    """
    # Coba kedua arah, pilih yang memberi foreground < 60% gambar
    # (beras tidak mungkin memenuhi > 60% frame)
    _, binary_inv = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, binary_std = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    ratio_inv = np.sum(binary_inv == 255) / binary_inv.size
    ratio_std = np.sum(binary_std == 255) / binary_std.size

    # Pilih hasil yang foreground-nya antara 5% – 60%
    # (threshold wajar untuk foto beras di atas nampan/kertas)
    if 0.05 <= ratio_std <= 0.60:
        binary = binary_std
    elif 0.05 <= ratio_inv <= 0.60:
        binary = binary_inv
    else:
        # Fallback: pilih yang lebih mendekati 30%
        binary = binary_std if abs(ratio_std - 0.30) < abs(ratio_inv - 0.30) \
                 else binary_inv

    return {"binary": binary}


# ─────────────────────────────────────────────
# STEP 3 — Operasi Morfologi
# ─────────────────────────────────────────────
def morphology(binary: np.ndarray) -> dict:
    """
    Opening → hilangkan noise kecil
    Closing → tutup lubang di dalam butir
    """
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    return {"morphed": closed}


# ─────────────────────────────────────────────
# STEP 4 — Contour Detection & Feature Extraction
# ─────────────────────────────────────────────
def extract_features(
    morphed: np.ndarray,
    original_bgr: np.ndarray,
    gray: np.ndarray,
    min_area: int = 300,
    max_area: int = 50000,
) -> list[GrainFeatures]:
    """
    Deteksi kontur → filter area → ekstrak fitur geometris.
    """
    contours, _ = cv2.findContours(
        morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    grains: list[GrainFeatures] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # Bounding box → aspect ratio
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = float(max(bw, bh)) / float(min(bw, bh) + 1e-6)

        # Extent: area / bounding box area
        bbox_area = bw * bh
        extent = float(area) / (bbox_area + 1e-6)

        # Solidity: area / convex hull area
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / (hull_area + 1e-6)

        # Mean brightness di area butir
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        color_mean = float(cv2.mean(gray, mask=mask)[0])

        grains.append(GrainFeatures(
            contour=cnt,
            area=area,
            perimeter=perimeter,
            aspect_ratio=aspect_ratio,
            extent=extent,
            solidity=solidity,
            color_mean=color_mean,
        ))

    return grains


# ─────────────────────────────────────────────
# STEP 5 — Rule-based labeling (sebagai ground truth / fallback)
# ─────────────────────────────────────────────
def label_grains_rule_based(
    grains: list[GrainFeatures],
) -> list[GrainFeatures]:
    """
    Heuristik sederhana berdasarkan fitur geometris:
    - whole   : aspect_ratio < 3.5, solidity > 0.80, extent > 0.55
    - impurity: area sangat kecil atau color_mean < 80 (gelap)
    - broken  : sisanya
    """
    if not grains:
        return grains

    areas = [g.area for g in grains]
    mean_area = np.mean(areas)

    for g in grains:
        # Kotoran / benda asing: sangat gelap atau area jauh di bawah rata-rata
        if g.color_mean < 80 or g.area < mean_area * 0.25:
            g.label = "impurity"
        # Butir utuh: bentuk memanjang tapi tidak terlalu, solidity & extent tinggi
        elif g.aspect_ratio < 3.5 and g.solidity > 0.78 and g.extent > 0.50:
            g.label = "whole"
        # Butir patah: sisanya
        else:
            g.label = "broken"

    return grains


# ─────────────────────────────────────────────
# STEP 6 — Visualisasi: gambar dengan contour overlay
# ─────────────────────────────────────────────
COLOR_MAP = {
    "whole":    (0, 200, 0),    # hijau
    "broken":   (0, 100, 255),  # oranye
    "impurity": (0, 0, 220),    # merah
    "unknown":  (200, 200, 0),  # kuning
}

def draw_annotated(original_bgr: np.ndarray, grains: list[GrainFeatures]) -> np.ndarray:
    """
    Gambar contour warna-warni berdasarkan label di atas citra asli.
    """
    canvas = original_bgr.copy()
    for g in grains:
        color = COLOR_MAP.get(g.label, (200, 200, 0))
        cv2.drawContours(canvas, [g.contour], -1, color, 2)
        # Titik tengah untuk label teks kecil
        M = cv2.moments(g.contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            initial = g.label[0].upper()  # W / B / I
            cv2.putText(canvas, initial, (cx - 5, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return canvas


# ─────────────────────────────────────────────
# Utility: fitur array untuk ML
# ─────────────────────────────────────────────
def grains_to_feature_array(grains: list[GrainFeatures]) -> np.ndarray:
    """Konversi list GrainFeatures ke numpy array [N x 5] untuk classifier."""
    return np.array([
        [g.area, g.perimeter, g.aspect_ratio, g.extent, g.solidity]
        for g in grains
    ], dtype=np.float32)


# ─────────────────────────────────────────────
# Pemisah butir menempel (Watershed)
# ─────────────────────────────────────────────
def separate_touching_grains(
    binary: np.ndarray,
    original_bgr: np.ndarray,
    min_dist: float = 3.0,
    dilate_size: int = 15,
) -> np.ndarray:
    """
    Memisahkan butir beras yang menempel menggunakan kombinasi Distance Transform,
    Gaussian smoothing, local maxima detection via dilation, dan segmentasi Watershed.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # 1. Tentukan sure background dengan dilasi
    sure_bg = cv2.dilate(binary, kernel, iterations=2)
    
    # 2. Distance Transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # 3. Smooth distance transform untuk meredam noise lokal kecil
    dist_smooth = cv2.GaussianBlur(dist_transform, (5, 5), 0)
    
    # 4. Cari peak lokal (local maxima) dengan dilasi berkelompok sesuai perkiraan lebar butir
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    local_max = cv2.dilate(dist_smooth, dilate_kernel)
    
    # 5. Sure foreground: pixel yang sama dengan nilai lokal maksimum dan di atas batas jarak minimal
    sure_fg = (dist_smooth == local_max) & (dist_transform > min_dist)
    sure_fg = np.uint8(sure_fg) * 255
    
    # 6. Cari area perbatasan (unknown)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 7. Labeling objek/markers
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Naikkan 1 nilai agar background bernilai 1 (bukan 0)
    markers = markers + 1
    # Tandai area perbatasan tidak dikenal dengan 0
    markers[unknown == 255] = 0
    
    # 8. Jalankan Watershed
    markers = np.int32(markers)
    markers = cv2.watershed(original_bgr, markers)
    
    # 9. Gambar garis pembatas (markers == -1) dengan warna hitam (0) di mask
    result = binary.copy()
    result[markers == -1] = 0
    
    # Bersihkan sisa jembatan kecil dengan opening ringan
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return result


# ─────────────────────────────────────────────
# Full pipeline (single entry point)
# ─────────────────────────────────────────────
def run_pipeline(
    image_bgr: np.ndarray,
    use_watershed: bool = False,
    min_dist: float = 3.0,
    dilate_size: int = 15,
) -> dict:
    """
    Jalankan seluruh pipeline dan kembalikan semua hasil.
    """
    pre   = preprocess(image_bgr)
    seg   = segment(pre["enhanced"])
    morph = morphology(seg["binary"])
    
    binary_for_contours = morph["morphed"]
    if use_watershed:
        binary_for_contours = separate_touching_grains(
            binary_for_contours,
            pre["original_bgr"],
            min_dist=min_dist,
            dilate_size=dilate_size,
        )
        
    grains = extract_features(
        binary_for_contours, pre["original_bgr"], pre["gray"]
    )
    grains = label_grains_rule_based(grains)
    annotated = draw_annotated(pre["original_bgr"], grains)
 
    return {
        **pre,
        **seg,
        **morph,
        "morphed_separated": binary_for_contours,
        "grains": grains,
        "annotated": annotated,
        "feature_array": grains_to_feature_array(grains),
    }
