"""
utils/dataset_builder.py
Helper untuk membangun dataset training dari:
1. Folder gambar Kaggle (dengan subfolder whole/broken/impurity)
2. Labeling manual (append baris ke CSV)
"""

import os
import csv
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Callable

from utils.image_processing import (
    preprocess,
    segment,
    morphology,
    extract_features,
    GrainFeatures,
)

# ─────────────────────────────────────────────────────────────
# Kolom standar CSV training data
# ─────────────────────────────────────────────────────────────
CSV_COLUMNS = ["area", "perimeter", "aspect_ratio", "extent", "solidity",
               "color_mean", "label", "source"]

VALID_LABELS = {"whole", "broken", "impurity"}


# ─────────────────────────────────────────────────────────────
# Load / save CSV
# ─────────────────────────────────────────────────────────────
def load_training_csv(csv_path: str) -> pd.DataFrame:
    """
    Load training CSV. Return DataFrame kosong jika file tidak ada atau kosong.
    Hanya return baris yang label-nya valid (whole/broken/impurity).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)

    try:
        df = pd.read_csv(csv_path)
        # Pastikan kolom ada semua
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        # Filter label valid
        df = df[df["label"].isin(VALID_LABELS)].reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=CSV_COLUMNS)


def save_training_csv(df: pd.DataFrame, csv_path: str):
    """Simpan DataFrame ke CSV, buat folder jika belum ada."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df[CSV_COLUMNS].to_csv(csv_path, index=False)


def append_rows_to_csv(rows: list[dict], csv_path: str):
    """
    Append list of dicts ke CSV. Buat file dengan header jika belum ada.
    Setiap dict harus punya key sesuai CSV_COLUMNS.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            # Pastikan semua key ada
            clean_row = {col: row.get(col, "") for col in CSV_COLUMNS}
            writer.writerow(clean_row)


# ─────────────────────────────────────────────────────────────
# Ekstraksi fitur dari satu gambar
# ─────────────────────────────────────────────────────────────
def extract_features_from_image(
    image_bgr: np.ndarray,
    label: str,
    source: str = "manual",
) -> list[dict]:
    """
    Jalankan pipeline OpenCV pada satu gambar, ekstrak fitur tiap butir.
    Gunakan label yang diberikan untuk semua butir pada gambar ini.
    Return list of dicts siap di-append ke CSV.

    Catatan: Pakai target_size=300 (bukan 800) agar gambar individual butir
    (seperti dari Kaggle) tidak melampaui batas max_area pipeline default.
    Area bounds juga dibuat dinamis berdasarkan ukuran gambar setelah resize.
    """
    if label not in VALID_LABELS:
        raise ValueError(f"Label tidak valid: {label}. Pilih dari: {VALID_LABELS}")

    # Resize lebih kecil — gambar Kaggle adalah individual grain (1 butir per foto)
    # Kalau pakai 800px, area butir bisa 200.000+ px² → difilter keluar
    pre  = preprocess(image_bgr, target_size=300)
    seg  = segment(pre["enhanced"])
    morph = morphology(seg["binary"])

    # Dynamic area bounds berdasarkan ukuran gambar setelah preprocessing
    # Sehingga butir yang mengisi sebagian besar frame tetap terdeteksi
    h, w = morph["morphed"].shape
    min_a = max(100, int(h * w * 0.005))   # min 0.5% luas frame
    max_a = int(h * w * 0.95)              # max 95% luas frame

    grains = extract_features(
        morph["morphed"], pre["original_bgr"], pre["gray"],
        min_area=min_a, max_area=max_a,
    )

    rows = []
    for g in grains:
        rows.append({
            "area":         round(g.area, 2),
            "perimeter":    round(g.perimeter, 2),
            "aspect_ratio": round(g.aspect_ratio, 4),
            "extent":       round(g.extent, 4),
            "solidity":     round(g.solidity, 4),
            "color_mean":   round(g.color_mean, 2),
            "label":        label,
            "source":       source,
        })
    return rows



# ─────────────────────────────────────────────────────────────
# Proses folder Kaggle
# ─────────────────────────────────────────────────────────────
def process_kaggle_folder(
    folder_path: str,
    output_csv: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    max_per_class: int = 500,
) -> dict:
    """
    Scan folder Kaggle dengan struktur subfolder label:
        folder_path/
            whole/      ← gambar butir utuh
            broken/     ← gambar butir patah
            impurity/   ← gambar kotoran/benda asing

    Untuk tiap gambar: jalankan pipeline OpenCV → ekstrak fitur butir →
    simpan ke CSV.

    Args:
        folder_path    : Path ke folder root Kaggle
        output_csv     : Path output CSV
        progress_callback: fn(current, total, message) untuk update progress UI
        max_per_class  : Batasi jumlah gambar per kelas (bukan butir)

    Returns:
        dict: {"total_images": int, "total_grains": int, "per_class": dict,
               "errors": list[str]}
    """
    folder_path = Path(folder_path)
    stats = {
        "total_images": 0,
        "total_grains": 0,
        "per_class":    {lbl: 0 for lbl in VALID_LABELS},
        "errors":       [],
    }

    # Ekstensi gambar yang didukung
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    # Kumpulkan semua task (label, path)
    tasks = []
    for label in VALID_LABELS:
        label_dir = folder_path / label
        if not label_dir.is_dir():
            # Coba case-insensitive
            for d in folder_path.iterdir():
                if d.is_dir() and d.name.lower() == label.lower():
                    label_dir = d
                    break
            else:
                stats["errors"].append(
                    f"Subfolder '{label}' tidak ditemukan di {folder_path}"
                )
                continue

        # Scan rekursif — tangani subfolder seperti whole/Arborio/, whole/Basmati/, dst.
        img_files = [
            f for f in label_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in IMG_EXTS
        ][:max_per_class]

        if not img_files:
            stats["errors"].append(
                f"⚠️ Tidak ada gambar di folder '{label}' (termasuk subfolder)."
            )
            continue

        tasks.extend([(label, f) for f in img_files])

    total_tasks = len(tasks)

    # Proses tiap gambar
    all_rows = []
    for idx, (label, img_path) in enumerate(tasks):
        if progress_callback:
            progress_callback(idx, total_tasks, f"[{label}] {img_path.name}")

        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                stats["errors"].append(f"Gagal baca: {img_path}")
                continue

            rows = extract_features_from_image(img_bgr, label, source="kaggle")
            all_rows.extend(rows)
            stats["total_images"]  += 1
            stats["total_grains"]  += len(rows)
            stats["per_class"][label] += len(rows)

        except Exception as e:
            stats["errors"].append(f"Error pada {img_path.name}: {e}")

    if progress_callback:
        progress_callback(total_tasks, total_tasks, "Menyimpan CSV...")

    # Load existing + merge + save
    existing_df = load_training_csv(output_csv)
    # Hapus data kaggle lama supaya tidak duplikat jika diproses ulang
    existing_df = existing_df[existing_df["source"] != "kaggle"]
    new_df = pd.DataFrame(all_rows, columns=CSV_COLUMNS)
    merged = pd.concat([existing_df, new_df], ignore_index=True)
    save_training_csv(merged, output_csv)

    return stats


# ─────────────────────────────────────────────────────────────
# Append hasil labeling manual
# ─────────────────────────────────────────────────────────────
def append_manual_labels(
    grains: list[GrainFeatures],
    labels_override: dict[int, str],
    output_csv: str,
) -> int:
    """
    Append fitur butir yang sudah diberi label manual ke CSV.

    Args:
        grains          : List GrainFeatures dari run_pipeline
        labels_override : Dict {index_butir: label_string}
        output_csv      : Path ke training_data.csv

    Returns:
        Jumlah baris yang berhasil disimpan
    """
    rows = []
    for idx, label in labels_override.items():
        if label not in VALID_LABELS:
            continue
        if idx < 0 or idx >= len(grains):
            continue
        g = grains[idx]
        rows.append({
            "area":         round(g.area, 2),
            "perimeter":    round(g.perimeter, 2),
            "aspect_ratio": round(g.aspect_ratio, 4),
            "extent":       round(g.extent, 4),
            "solidity":     round(g.solidity, 4),
            "color_mean":   round(g.color_mean, 2),
            "label":        label,
            "source":       "manual",
        })

    if rows:
        append_rows_to_csv(rows, output_csv)
    return len(rows)


# ─────────────────────────────────────────────────────────────
# Info statistik CSV
# ─────────────────────────────────────────────────────────────
def get_dataset_stats(csv_path: str) -> dict:
    """Return statistik ringkas dataset."""
    df = load_training_csv(csv_path)
    if df.empty:
        return {"total": 0, "per_label": {}, "per_source": {}, "ready": False}

    per_label  = df["label"].value_counts().to_dict()
    per_source = df["source"].value_counts().to_dict()

    # Minimal dataset: tiap kelas >= 10 baris
    ready = all(per_label.get(lbl, 0) >= 10 for lbl in VALID_LABELS)

    return {
        "total":      len(df),
        "per_label":  per_label,
        "per_source": per_source,
        "ready":      ready,
    }
