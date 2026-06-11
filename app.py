"""
app.py — Smart Rice Quality Analyzer
Streamlit UI: upload / kamera → pipeline OpenCV → KNN/SVM → hasil analisis
+ Tab Training Data: Dataset Builder (Kaggle) & Labeling Manual
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import io
import sys
import os
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from utils.image_processing import run_pipeline, draw_annotated, COLOR_MAP
from utils.dataset_builder import (
    process_kaggle_folder,
    append_manual_labels,
    get_dataset_stats,
    load_training_csv,
    VALID_LABELS,
)
from models.classifier import RiceClassifier, compute_quality

# ── path CSV ─────────────────────────────────────────────────────────────────
CSV_PATH = str(Path(__file__).parent / "data" / "training_data.csv")

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Rice Quality Analyzer",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 { font-size: 2.1rem; margin-bottom: 0.2rem; }
    .main-header p  { color: #666; font-size: 0.95rem; }

    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 12px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-card .label { font-size: 0.82rem; color: #666; margin-bottom: 4px; }
    .metric-card .value { font-size: 1.7rem; font-weight: 700; }

    .grade-box {
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 8px 0;
    }
    .grade-baik     { background: #d4edda; color: #155724; border: 2px solid #28a745; }
    .grade-sedang   { background: #fff3cd; color: #856404; border: 2px solid #ffc107; }
    .grade-rendah   { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
    .grade-unknown  { background: #e2e3e5; color: #383d41; border: 2px solid #aaa; }

    .source-badge-real    { background:#d4edda; color:#155724; padding:4px 10px;
                            border-radius:20px; font-size:0.8rem; font-weight:600; }
    .source-badge-synth   { background:#fff3cd; color:#856404; padding:4px 10px;
                            border-radius:20px; font-size:0.8rem; font-weight:600; }

    div[data-testid="stImage"] img { border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# ── cached model ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model KNN & SVM...")
def load_classifier():
    return RiceClassifier(knn_k=5, svm_C=1.0, svm_kernel="rbf")


# ── helper: PIL → BGR numpy ──────────────────────────────────────────────────
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ── helper: BGR numpy → PIL ──────────────────────────────────────────────────
def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ── helper: plot distribusi fitur ────────────────────────────────────────────
def plot_feature_dist(grains, labels):
    if not grains:
        return None

    color_rgb = {
        "whole":    "#00c800",
        "broken":   "#ff6400",
        "impurity": "#dc0000",
        "unknown":  "#cccc00",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    feat_data = {
        "Aspect Ratio":  [g.aspect_ratio for g in grains],
        "Solidity":      [g.solidity      for g in grains],
        "Extent":        [g.extent        for g in grains],
    }
    feat_names = list(feat_data.keys())
    feat_vals  = list(feat_data.values())

    unique_labels = list(set(labels))
    for ax, name, vals in zip(axes, feat_names, feat_vals):
        for lbl in unique_labels:
            idxs = [i for i, l in enumerate(labels) if l == lbl]
            ax.hist(
                [vals[i] for i in idxs],
                bins=15, alpha=0.6,
                color=color_rgb.get(lbl, "#aaa"),
                label=lbl.capitalize(),
                edgecolor="white", linewidth=0.5
            )
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Nilai", fontsize=8)
        ax.set_ylabel("Frekuensi", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Distribusi Fitur per Kelas", fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ── helper: pie chart kualitas ────────────────────────────────────────────────
def plot_pie(quality: dict) -> plt.Figure:
    sizes  = [quality["whole_pct"], quality["broken_pct"], quality["impurity_pct"]]
    labels = ["Utuh", "Patah", "Kotoran"]
    colors = ["#28a745", "#fd7e14", "#dc3545"]
    explode = (0.04, 0.04, 0.04)

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct=lambda p: f"{p:.1f}%" if p > 0.5 else "",
        startangle=140, wedgeprops={"linewidth": 1.5, "edgecolor": "white"}
    )
    for t in texts:     t.set_fontsize(9)
    for t in autotexts: t.set_fontsize(8); t.set_fontweight("bold")
    ax.set_title("Komposisi Butir Beras", fontsize=10, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


# ── helper: bar chart jumlah butir ───────────────────────────────────────────
def plot_bar(quality: dict) -> plt.Figure:
    categories = ["Utuh", "Patah", "Kotoran"]
    values     = [quality["whole_n"], quality["broken_n"], quality["impurity_n"]]
    colors     = ["#28a745", "#fd7e14", "#dc3545"]

    fig, ax = plt.subplots(figsize=(4, 3.5))
    bars = ax.bar(categories, values, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, str(val),
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Jumlah Butir", fontsize=9)
    ax.set_title("Jumlah per Kategori", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig


# ── helper: generate laporan teks ────────────────────────────────────────────
def build_report(quality: dict, method: str) -> str:
    lines = [
        "=" * 45,
        "    SMART RICE QUALITY ANALYZER",
        "    Laporan Hasil Analisis Kualitas Beras",
        "=" * 45,
        "",
        f"Metode Klasifikasi : {method.upper()}",
        f"Total Butir        : {quality['total']}",
        "",
        "── Komposisi ─────────────────────────────",
        f"  Butir Utuh   : {quality['whole_n']:>4}  ({quality['whole_pct']:>5.1f}%)",
        f"  Butir Patah  : {quality['broken_n']:>4}  ({quality['broken_pct']:>5.1f}%)",
        f"  Kotoran      : {quality['impurity_n']:>4}  ({quality['impurity_pct']:>5.1f}%)",
        "",
        "── Grade Kualitas ────────────────────────",
        f"  {quality['grade']}",
        "",
        "── Kriteria Grading ──────────────────────",
        "  Baik   : Utuh ≥ 80%, Kotoran ≤ 2%",
        "  Sedang : Utuh ≥ 60%, Kotoran ≤ 5%",
        "  Rendah : Di bawah kriteria Sedang",
        "=" * 45,
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: TRAINING DATA
# ════════════════════════════════════════════════════════════════════════════
def page_training_data(clf: RiceClassifier):
    st.markdown("""
    <div class="main-header">
        <h1>🏷️ Training Data Manager</h1>
        <p>Bangun dataset real dari gambar Kaggle atau labeling manual</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Info dataset saat ini ────────────────────────────────────────────────
    stats = get_dataset_stats(CSV_PATH)
    info  = clf.get_training_info()

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        src_label = "✅ Real Data (CSV)" if info["data_source"] == "csv_real" else "⚠️ Synthetic Data"
        st.metric("Sumber Model Aktif", src_label)
    with col_info2:
        st.metric("Total Sampel di CSV", stats.get("total", 0))
    with col_info3:
        ready_txt = "✅ Siap dipakai" if stats.get("ready") else "❌ Belum cukup"
        st.metric("Status Dataset", ready_txt)

    if stats.get("total", 0) > 0:
        per_lbl = stats.get("per_label", {})
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Butir Utuh (whole)",    per_lbl.get("whole", 0),    help="min. 10")
        with c2: st.metric("Butir Patah (broken)",  per_lbl.get("broken", 0),   help="min. 10")
        with c3: st.metric("Kotoran (impurity)",    per_lbl.get("impurity", 0), help="min. 10")

        per_src = stats.get("per_source", {})
        src_parts = [f"**{k}**: {v}" for k, v in per_src.items()]
        st.caption("Sumber data: " + " · ".join(src_parts))

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_kaggle, tab_manual, tab_manage = st.tabs([
        "📦 Opsi 1 — Import Kaggle",
        "✍️ Opsi 2 — Labeling Manual",
        "🗂️ Kelola Dataset",
    ])

    # ════════════════ TAB 1: KAGGLE ════════════════
    with tab_kaggle:
        st.subheader("Import Fitur dari Dataset Kaggle")
        st.info("""
        **Cara pakai:**
        1. Download salah satu dataset di bawah dari Kaggle
        2. Ekstrak ke folder lokal dengan struktur subfolder: `whole/`, `broken/`, `impurity/`
        3. Masukkan path folder di bawah → klik **Proses Folder**

        📎 Dataset rekomendasi:
        - [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset) — 75k gambar, 5 varietas
        - [Rice Grain Quality](https://www.kaggle.com/datasets/sobhanmoosavi/rice-quality) — ada label whole/broken

        > **Catatan:** Folder Kaggle tidak perlu di-commit ke GitHub (sudah di `.gitignore`).
        > Yang di-commit hanya `data/training_data.csv` (fitur, bukan gambar).
        """)

        st.markdown("**Struktur folder yang diharapkan:**")
        st.code("""
data/kaggle/          ← isi path ini di bawah
├── whole/
│   ├── img_001.jpg
│   └── ...
├── broken/
│   ├── img_101.jpg
│   └── ...
└── impurity/
    ├── img_201.jpg
    └── ...
        """, language="text")

        kaggle_path = st.text_input(
            "📁 Path folder Kaggle",
            placeholder=r"Contoh: D:\Datasets\rice_kaggle",
            help="Folder yang berisi subfolder whole/, broken/, impurity/"
        )

        max_per = st.slider(
            "Maks. gambar per kelas", 50, 1000, 300, 50,
            help="Lebih banyak = lebih akurat tapi lebih lama diproses"
        )

        if st.button("🚀 Proses Folder Kaggle", type="primary", disabled=not kaggle_path):
            folder = Path(kaggle_path.strip())
            if not folder.exists():
                st.error(f"❌ Folder tidak ditemukan: `{folder}`")
            elif not folder.is_dir():
                st.error("❌ Path bukan folder/direktori.")
            else:
                prog_bar  = st.progress(0, text="Memulai ekstraksi fitur...")
                prog_text = st.empty()

                def cb(current, total, msg):
                    pct = int(current / total * 100) if total > 0 else 0
                    prog_bar.progress(pct, text=f"[{current}/{total}] {msg}")
                    prog_text.caption(msg)

                with st.spinner("Mengekstrak fitur dari gambar..."):
                    result = process_kaggle_folder(
                        folder_path=str(folder),
                        output_csv=CSV_PATH,
                        progress_callback=cb,
                        max_per_class=max_per,
                    )

                prog_bar.empty()
                prog_text.empty()

                if result["total_images"] > 0:
                    st.success(
                        f"✅ Berhasil memproses **{result['total_images']} gambar** → "
                        f"**{result['total_grains']} fitur butir** disimpan ke CSV."
                    )
                    pc = result["per_class"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Whole (butir)", pc.get("whole", 0))
                    c2.metric("Broken (butir)", pc.get("broken", 0))
                    c3.metric("Impurity (butir)", pc.get("impurity", 0))

                    if result["errors"]:
                        with st.expander(f"⚠️ {len(result['errors'])} error/warning"):
                            for e in result["errors"][:20]:
                                st.caption(e)

                    # Auto-retrain secara langsung jika dataset siap
                    new_stats = get_dataset_stats(CSV_PATH)
                    if new_stats.get("ready"):
                        st.divider()
                        with st.spinner("Melatih ulang model secara otomatis..."):
                            retrain_res = clf.retrain_from_csv(CSV_PATH)
                        if retrain_res["success"]:
                            st.success("🔁 Model berhasil dilatih ulang secara otomatis menggunakan data real terbaru!")
                            st.rerun()
                else:
                    st.error("❌ Tidak ada gambar yang berhasil diproses.")
                    if result["errors"]:
                        for e in result["errors"]:
                            st.caption(f"• {e}")

    # ════════════════ TAB 2: LABELING MANUAL ════════════════
    with tab_manual:
        st.subheader("Labeling Manual dari Foto Beras Sendiri")
        st.info("""
        Upload foto beras → sistem otomatis mendeteksi butir →
        kamu assign label tiap butir → simpan ke dataset.

        **Tips:** Upload foto beras yang sudah *disortir* (satu tipe per foto)
        supaya labeling lebih cepat.
        """)

        uploaded_label = st.file_uploader(
            "Upload foto beras untuk dilabeli (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            key="labeling_uploader",
        )

        if uploaded_label is None:
            st.caption("👆 Upload gambar untuk memulai.")
            return

        pil_img = Image.open(uploaded_label)
        bgr     = pil_to_bgr(pil_img)

        # Proses pipeline
        with st.spinner("Mendeteksi butir..."):
            result_label = run_pipeline(bgr)
            grains_label = result_label["grains"]

        if not grains_label:
            st.warning("⚠️ Tidak ada butir yang terdeteksi pada gambar ini.")
            return

        st.success(f"✅ Ditemukan **{len(grains_label)} butir**. Assign label di bawah.")

        # Gambar dengan nomor butir
        canvas_num = result_label["original_bgr"].copy()
        for i, g in enumerate(grains_label):
            M = cv2.moments(g.contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.drawContours(canvas_num, [g.contour], -1, (0, 180, 255), 2)
                cv2.putText(canvas_num, str(i + 1), (cx - 8, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1, cv2.LINE_AA)

        col_img, col_form = st.columns([1, 1])

        with col_img:
            st.markdown("**Gambar dengan nomor butir:**")
            st.image(bgr_to_pil(canvas_num), use_container_width=True)

        with col_form:
            st.markdown("**Assign label per butir:**")

            # Jika butir terlalu banyak, tampilkan dengan batch label
            if len(grains_label) > 20:
                st.warning(f"Ada {len(grains_label)} butir terdeteksi. "
                           "Gunakan 'Batch Label' untuk lebih efisien.")
                mode = st.radio("Mode labeling:", ["Batch (semua satu label)", "Per butir"],
                                horizontal=True)
            else:
                mode = "Per butir"

            label_options = ["whole", "broken", "impurity", "skip"]

            if mode == "Batch (semua satu label)":
                batch_label = st.selectbox(
                    "Label untuk SEMUA butir di gambar ini:",
                    ["whole", "broken", "impurity"],
                )
                labels_override = {i: batch_label for i in range(len(grains_label))}
                n_to_save = len(grains_label)
            else:
                labels_override = {}
                n_to_save = 0
                # Tampilkan selectbox per butir dalam scrollable container
                for i, g in enumerate(grains_label):
                    area_str = f"{int(g.area):,}"
                    col_n, col_sel = st.columns([1, 3])
                    with col_n:
                        st.markdown(f"**#{i+1}** *(area: {area_str})*")
                    with col_sel:
                        lbl = st.selectbox(
                            f"Label #{i+1}",
                            label_options,
                            index=0,
                            key=f"lbl_{i}",
                            label_visibility="collapsed",
                        )
                    if lbl != "skip":
                        labels_override[i] = lbl
                        n_to_save += 1

            st.divider()
            st.caption(f"**{n_to_save}** butir akan disimpan (skip dikecualikan)")

            if st.button("💾 Simpan ke Dataset", type="primary", disabled=n_to_save == 0):
                saved = append_manual_labels(grains_label, labels_override, CSV_PATH)
                st.success(f"✅ **{saved} sampel** berhasil disimpan ke `data/training_data.csv`!")

                # Refresh stats
                new_stats = get_dataset_stats(CSV_PATH)
                st.info(f"Total dataset sekarang: **{new_stats['total']} sampel**")

                # Latih ulang secara otomatis jika siap
                if new_stats.get("ready"):
                    with st.spinner("Melatih ulang model secara otomatis..."):
                        retrain_res = clf.retrain_from_csv(CSV_PATH)
                    if retrain_res["success"]:
                        st.success("🔁 Model berhasil dilatih ulang secara otomatis menggunakan data real terbaru!")
                        st.rerun()

    # ════════════════ TAB 3: KELOLA DATASET ════════════════
    with tab_manage:
        st.subheader("Kelola Dataset Training")

        df_all = load_training_csv(CSV_PATH)
        if df_all.empty:
            st.info("📭 Dataset kosong. Import dari Kaggle atau labeling manual dulu.")
            return

        # Preview tabel
        st.markdown(f"**Total: {len(df_all)} baris**")
        st.dataframe(df_all, use_container_width=True, height=300)

        col_dl, col_del = st.columns(2)
        with col_dl:
            csv_bytes = df_all.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                csv_bytes,
                "training_data.csv",
                "text/csv",
                use_container_width=True,
            )
        with col_del:
            if st.button("🗑️ Hapus Semua Data Kaggle", use_container_width=True):
                df_keep = df_all[df_all["source"] != "kaggle"]
                from utils.dataset_builder import save_training_csv
                save_training_csv(df_keep, CSV_PATH)
                st.success(f"Dihapus. Sisa: {len(df_keep)} baris (manual).")
                st.rerun()

        st.divider()
        st.markdown("**Retrain Model**")
        if st.button("🔁 Retrain Model dari CSV Saat Ini", type="primary",
                     use_container_width=True):
            _do_retrain(clf)


def _do_retrain(clf: RiceClassifier):
    """Helper: retrain dan tampilkan hasilnya."""
    with st.spinner("Melatih ulang model..."):
        retrain_result = clf.retrain_from_csv(CSV_PATH)

    if retrain_result["success"]:
        st.success(retrain_result["message"])
        pc = retrain_result["n_per_class"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Whole", pc.get("whole", 0))
        c2.metric("Broken", pc.get("broken", 0))
        c3.metric("Impurity", pc.get("impurity", 0))
        st.caption(f"Total sampel training: **{retrain_result['n_samples']}**")
    else:
        st.warning(retrain_result["message"])


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: ANALISIS UTAMA
# ════════════════════════════════════════════════════════════════════════════
def page_analyze(clf: RiceClassifier, method: str, min_area: int, max_area: int,
                 use_watershed: bool = False, min_dist: float = 3.0, dilate_size: int = 15):
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🌾 Smart Rice Quality Analyzer</h1>
        <p>Analisis Kualitas Beras Menggunakan Pengolahan Citra Digital | Kelompok JYP Sarjana</p>
    </div>
    """, unsafe_allow_html=True)

    # Badge sumber model
    info = clf.get_training_info()
    if info["data_source"] == "csv_real":
        badge = f'<span class="source-badge-real">✅ Model: Real Data ({info["n_samples"]} sampel)</span>'
    else:
        badge = '<span class="source-badge-synth">⚠️ Model: Synthetic Data (belum ada real dataset)</span>'
    st.markdown(badge, unsafe_allow_html=True)

    st.divider()

    # ── Input gambar ─────────────────────────────────────────────────────────
    st.subheader("📸 Input Citra Beras")
    input_tab1, input_tab2 = st.tabs(["📁 Upload Gambar", "📷 Kamera"])

    uploaded = None
    with input_tab1:
        uploaded_file = st.file_uploader(
            "Upload foto beras (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            uploaded = Image.open(uploaded_file)

    with input_tab2:
        cam_img = st.camera_input("Ambil foto beras langsung")
        if cam_img:
            uploaded = Image.open(cam_img)

    # ── Analisis ─────────────────────────────────────────────────────────────
    if uploaded is None:
        st.info("👆 Upload gambar atau ambil foto beras untuk memulai analisis.")
        st.markdown("---")
        _show_how_it_works()
        return

    bgr = pil_to_bgr(uploaded)

    with st.spinner("🔄 Memproses citra..."):
        result = run_pipeline(
            bgr,
            use_watershed=use_watershed,
            min_dist=min_dist,
            dilate_size=dilate_size,
        )
        grains = result["grains"]
        feats  = np.array([[g.area, g.perimeter, g.aspect_ratio, g.extent, g.solidity]
                           for g in grains], dtype=np.float32)

        if len(grains) > 0:
            labels, proba = clf.predict(feats, method=method)
            for g, lbl in zip(grains, labels):
                g.label = lbl
        else:
            labels = np.array([])

        annotated = draw_annotated(result["original_bgr"], grains)
        quality   = compute_quality(list(labels))

    # ── Layout hasil ─────────────────────────────────────────────────────────
    st.success(f"✅ Analisis selesai — ditemukan **{quality['total']} butir** beras.")
    st.divider()

    col_orig, col_ann = st.columns(2)
    with col_orig:
        st.markdown("**🖼 Citra Asli**")
        st.image(bgr_to_pil(result["original_bgr"]), use_container_width=True)
    with col_ann:
        st.markdown("**🔍 Hasil Deteksi Butir**")
        st.image(bgr_to_pil(annotated), use_container_width=True)

    st.divider()

    # ── Metrik ringkasan ─────────────────────────────────────────────────────
    st.subheader("📊 Ringkasan Kualitas")
    m1, m2, m3, m4 = st.columns(4)

    def metric_html(label, value, color="#333"):
        return f"""<div class="metric-card">
            <div class="label">{label}</div>
            <div class="value" style="color:{color}">{value}</div>
        </div>"""

    with m1:
        st.markdown(metric_html("Total Butir", quality["total"]), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_html("Utuh", f"{quality['whole_pct']}%", "#28a745"),
                    unsafe_allow_html=True)
    with m3:
        st.markdown(metric_html("Patah", f"{quality['broken_pct']}%", "#fd7e14"),
                    unsafe_allow_html=True)
    with m4:
        st.markdown(metric_html("Kotoran", f"{quality['impurity_pct']}%", "#dc3545"),
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    grade_text = quality["grade"]
    if "Baik" in grade_text:
        css_class = "grade-baik"
    elif "Sedang" in grade_text:
        css_class = "grade-sedang"
    elif "Rendah" in grade_text:
        css_class = "grade-rendah"
    else:
        css_class = "grade-unknown"

    st.markdown(
        f'<div class="grade-box {css_class}">Grade Kualitas: {grade_text}</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # ── Chart & pipeline tab ──────────────────────────────────────────────────
    tab_chart, tab_pipeline, tab_fitur, tab_data = st.tabs([
        "📈 Visualisasi", "🔬 Pipeline Citra", "📐 Distribusi Fitur", "📋 Data Butir"
    ])

    with tab_chart:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_pie(quality), use_container_width=True)
        with c2:
            st.pyplot(plot_bar(quality), use_container_width=True)

    with tab_pipeline:
        st.markdown("**Tahapan preprocessing citra:**")
        if use_watershed:
            p1, p2, p3, p4, p5 = st.columns(5)
        else:
            p1, p2, p3, p4 = st.columns(4)
            
        with p1:
            st.image(result["gray"],    caption="1. Grayscale",       use_container_width=True, clamp=True)
        with p2:
            st.image(result["blurred"], caption="2. Gaussian Blur",   use_container_width=True, clamp=True)
        with p3:
            st.image(result["binary"],  caption="3. Otsu Threshold",  use_container_width=True, clamp=True)
        with p4:
            st.image(result["morphed"], caption="4. Morfologi",       use_container_width=True, clamp=True)
        if use_watershed:
            with p5:
                st.image(result["morphed_separated"], caption="5. Pemisah Watershed", use_container_width=True, clamp=True)

    with tab_fitur:
        if grains:
            fig = plot_feature_dist(grains, list(labels))
            if fig:
                st.pyplot(fig, use_container_width=True)
        else:
            st.info("Tidak ada butir terdeteksi untuk ditampilkan distribusinya.")

    with tab_data:
        if grains:
            df = pd.DataFrame([{
                "No":          i + 1,
                "Label":       g.label.capitalize(),
                "Area (px²)":  int(g.area),
                "Perimeter":   round(g.perimeter, 1),
                "Aspect Ratio":round(g.aspect_ratio, 2),
                "Extent":      round(g.extent, 3),
                "Solidity":    round(g.solidity, 3),
            } for i, g in enumerate(grains)])
            st.dataframe(df, use_container_width=True, height=350)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", csv, "rice_analysis.csv", "text/csv")
        else:
            st.info("Tidak ada butir terdeteksi.")

    st.divider()

    # ── Laporan & download ────────────────────────────────────────────────────
    st.subheader("📄 Laporan Analisis")
    report_text = build_report(quality, method)
    st.code(report_text, language=None)
    st.download_button(
        "⬇️ Download Laporan (.txt)",
        report_text.encode("utf-8"),
        "laporan_analisis_beras.txt",
        "text/plain"
    )


# ── How it works section ──────────────────────────────────────────────────────
def _show_how_it_works():
    st.subheader("📖 Cara Kerja Sistem")
    steps = [
        ("1️⃣", "Preprocessing",       "Resize → Grayscale → Gaussian Blur → CLAHE"),
        ("2️⃣", "Segmentasi",          "Otsu's Thresholding untuk memisahkan butir dari background"),
        ("3️⃣", "Operasi Morfologi",   "Opening & Closing untuk membersihkan noise dan memisahkan butir"),
        ("4️⃣", "Ekstraksi Fitur",     "Area, Perimeter, Aspect Ratio, Extent, Solidity tiap butir"),
        ("5️⃣", "Klasifikasi ML",      "KNN atau SVM untuk mengklasifikasikan: Utuh / Patah / Kotoran"),
        ("6️⃣", "Output Kualitas",     "Persentase komposisi + Grade: Baik / Sedang / Rendah"),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#f8f9fa;border-radius:10px;padding:14px;
                        margin-bottom:12px;border:1px solid #e0e0e0;min-height:100px">
                <div style="font-size:1.5rem">{icon}</div>
                <div style="font-weight:700;margin:4px 0">{title}</div>
                <div style="font-size:0.82rem;color:#555">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    clf = load_classifier()

    # Auto-retrain jika model masih synthetic tapi CSV sudah siap
    if clf.get_training_info()["data_source"] == "synthetic":
        stats = get_dataset_stats(CSV_PATH)
        if stats.get("ready"):
            clf.retrain_from_csv(CSV_PATH)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Pengaturan")

        page = st.radio(
            "Halaman",
            ["🔍 Analisis Beras", "🏷️ Training Data"],
            label_visibility="collapsed",
        )

        st.divider()

        if page == "🔍 Analisis Beras":
            method = st.radio(
                "Metode Klasifikasi",
                ["svm", "knn"],
                format_func=lambda x: "SVM (Support Vector Machine)" if x == "svm"
                                       else "KNN (K-Nearest Neighbor)",
                help="SVM umumnya memberikan hasil lebih akurat untuk dataset kecil."
            )

            st.divider()
            st.subheader("📐 Parameter Pipeline")
            min_area    = st.slider("Min Area Butir (px²)", 100, 1000, 150, 50)
            max_area    = st.slider("Batas Area Maksimal (px²)", 500, 10000, 1500, 100)

            st.subheader("💧 Pemisah Butir Menempel")
            use_watershed = st.toggle(
                "Pemisah Butir (Watershed)",
                value=True,
                help="Pisahkan butir beras yang saling menempel agar dideteksi secara individual."
            )
            if use_watershed:
                min_dist = st.slider(
                    "Batas Jarak Minimum (px)",
                    1.0, 15.0, 1.5, 0.5,
                    help="Jarak minimum pusat butir ke background. Nilai kecil = deteksi butir kecil, nilai besar = kurangi noise/plateau."
                )
                dilate_size = st.slider(
                    "Perkiraan Lebar Butir (px)",
                    3, 31, 9, 2,
                    help="Lebar masker pencari titik pusat (harus ganjil). Nilai besar = cegah butir terbelah dua. Nilai kecil = pisahkan butir rapat."
                )
            else:
                min_dist = 1.5
                dilate_size = 9

            st.divider()
            st.subheader("ℹ️ Legenda Warna")
            st.markdown("""
            🟢 **Hijau** — Butir Utuh  
            🟠 **Oranye** — Butir Patah  
            🔴 **Merah** — Kotoran / Impuritas
            """)
        else:
            method   = "svm"
            min_area = 150
            max_area = 1500
            use_watershed = True
            min_dist = 1.5
            dilate_size = 9

        st.divider()
        st.caption("Universitas Telkom · Informatika · 2026")

    # ── Route halaman ─────────────────────────────────────────────────────────
    if page == "🔍 Analisis Beras":
        page_analyze(clf, method, min_area, max_area, use_watershed, min_dist, dilate_size)
    else:
        page_training_data(clf)


if __name__ == "__main__":
    main()
