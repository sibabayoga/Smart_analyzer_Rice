# 🌾 Smart Rice Quality Analyzer

**Analisis Kualitas Beras Menggunakan Pengolahan Citra Digital**

> Kelompok JYP Sarjana — Program Studi Sarjana Informatika, Fakultas Informatika, Universitas Telkom 2026

---

## 📋 Deskripsi

Smart Rice Quality Analyzer adalah aplikasi berbasis **Streamlit** yang menganalisis kualitas beras secara otomatis dari foto menggunakan pipeline pengolahan citra digital dan machine learning.

Sistem mengklasifikasikan setiap butir beras ke dalam tiga kategori:
- 🟢 **Utuh (Whole Grain)** — butir beras berkualitas baik
- 🟠 **Patah (Broken Grain)** — butir beras yang pecah/patah  
- 🔴 **Kotoran (Impurity)** — benda asing / beras abnormal

---

## 🛠️ Tech Stack

| Komponen | Library |
|---|---|
| UI / Web App | Streamlit |
| Pengolahan Citra | OpenCV |
| Machine Learning | Scikit-learn (KNN & SVM) |
| Komputasi Numerik | NumPy, SciPy |
| Visualisasi | Matplotlib |
| Data | Pandas |

---

## 🔬 Pipeline Sistem

```
Input Gambar
    ↓
1. Preprocessing     → Resize, Grayscale, Gaussian Blur, CLAHE
    ↓
2. Segmentasi        → Otsu's Thresholding
    ↓
3. Operasi Morfologi → Opening + Closing
    ↓
4. Contour Detection → FindContours
    ↓
5. Feature Extraction → Area, Perimeter, Aspect Ratio, Extent, Solidity
    ↓
6. Klasifikasi ML    → KNN / SVM
    ↓
Output: Grade Kualitas (Baik / Sedang / Rendah) + Persentase
```

---

## 🚀 Cara Menjalankan

### 1. Clone repository
```bash
git clone https://github.com/username/smart-rice-quality-analyzer.git
cd smart-rice-quality-analyzer
```

### 2. Buat virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi
```bash
streamlit run app.py
```

Buka browser ke `http://localhost:8501`

---

## 📁 Struktur Project

```
smart_rice_quality_analyzer/
├── app.py                  # Main Streamlit UI
├── requirements.txt        # Dependencies
├── .gitignore
├── .streamlit/
│   └── config.toml         # Streamlit theme config
├── utils/
│   ├── __init__.py
│   └── image_processing.py # Pipeline OpenCV lengkap
├── models/
│   ├── __init__.py
│   └── classifier.py       # KNN & SVM classifier
└── assets/
    └── .gitkeep
```

---

## 📊 Kriteria Grading

| Grade | Butir Utuh | Kotoran |
|---|---|---|
| 🟢 **Baik** | ≥ 80% | ≤ 2% |
| 🟡 **Sedang** | ≥ 60% | ≤ 5% |
| 🔴 **Rendah** | < 60% | > 5% |

---

## 🌐 Deploy ke Streamlit Cloud

1. Push repo ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → pilih `app.py` sebagai entry point
4. Deploy! (gratis)

---
