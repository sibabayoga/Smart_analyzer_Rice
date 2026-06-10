"""
app.py — Smart Rice Quality Analyzer
Streamlit UI: upload / kamera → pipeline OpenCV → KNN/SVM → hasil analisis
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

# ── path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from utils.image_processing import run_pipeline, COLOR_MAP
from models.classifier import RiceClassifier, compute_quality

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
#  MAIN UI
# ════════════════════════════════════════════════════════════════════════════
def main():
    clf = load_classifier()

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🌾 Smart Rice Quality Analyzer</h1>
        <p>Analisis Kualitas Beras Menggunakan Pengolahan Citra Digital | Kelompok JYP Sarjana</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Pengaturan")

        method = st.radio(
            "Metode Klasifikasi",
            ["svm", "knn"],
            format_func=lambda x: "SVM (Support Vector Machine)" if x == "svm"
                                   else "KNN (K-Nearest Neighbor)",
            help="SVM umumnya memberikan hasil lebih akurat untuk dataset kecil."
        )

        st.divider()
        st.subheader("📐 Parameter Pipeline")
        min_area = st.slider("Min Area Butir (px²)", 100, 1000, 300, 50)
        max_area = st.slider("Max Area Butir (px²)", 5000, 80000, 50000, 1000)
        target_size = st.slider("Target Resize (px)", 400, 1200, 800, 100)

        st.divider()
        st.subheader("ℹ️ Legenda Warna")
        st.markdown("""
        🟢 **Hijau** — Butir Utuh  
        🟠 **Oranye** — Butir Patah  
        🔴 **Merah** — Kotoran / Impuritas
        """)

        st.divider()
        st.caption("Universitas Telkom · Informatika · 2026")

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
        result = run_pipeline(bgr)
        result["grains"] = [g for g in result["grains"]
                            if g.area >= min_area and g.area <= max_area]

        grains = result["grains"]
        feats  = np.array([[g.area, g.perimeter, g.aspect_ratio, g.extent, g.solidity]
                           for g in grains], dtype=np.float32)

        if len(grains) > 0:
            labels, proba = clf.predict(feats, method=method)
            for g, lbl in zip(grains, labels):
                g.label = lbl
        else:
            labels = np.array([])

        from utils.image_processing import draw_annotated
        annotated = draw_annotated(result["original_bgr"], grains)

        quality  = compute_quality(list(labels))

    # ── Layout hasil ─────────────────────────────────────────────────────────
    st.success(f"✅ Analisis selesai — ditemukan **{quality['total']} butir** beras.")
    st.divider()

    # Baris atas: gambar asli + gambar annotasi
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

    # Grade box
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
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.image(result["gray"],     caption="1. Grayscale",  use_container_width=True, clamp=True)
        with p2:
            st.image(result["blurred"],  caption="2. Gaussian Blur", use_container_width=True, clamp=True)
        with p3:
            st.image(result["binary"],   caption="3. Otsu Threshold", use_container_width=True, clamp=True)
        with p4:
            st.image(result["morphed"],  caption="4. Morfologi", use_container_width=True, clamp=True)

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


if __name__ == "__main__":
    main()
