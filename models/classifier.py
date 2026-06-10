"""
models/classifier.py
KNN dan SVM classifier untuk klasifikasi butir beras.

Prioritas training data:
  1. File CSV real data (data/training_data.csv) — jika ada & cukup
  2. Synthetic data sebagai fallback

Support retrain dari luar (dipanggil dari UI Streamlit).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings("ignore")

# ── Label encoding ────────────────────────────────────────────
LABEL_MAP    = {"whole": 0, "broken": 1, "impurity": 2}
LABEL_DECODE = {0: "whole", 1: "broken", 2: "impurity"}

# ── Path default CSV ──────────────────────────────────────────
_DEFAULT_CSV = Path(__file__).parent.parent / "data" / "training_data.csv"

# Minimum sampel per kelas agar CSV dianggap "cukup"
MIN_SAMPLES_PER_CLASS = 10


# ─────────────────────────────────────────────
# Synthetic training data (fallback)
# ─────────────────────────────────────────────
def _generate_training_data(n_per_class: int = 300, seed: int = 42) -> tuple:
    """
    Generate synthetic training samples dengan distribusi realistik
    untuk tiap kelas berdasarkan fitur:
    [area, perimeter, aspect_ratio, extent, solidity]
    """
    rng = np.random.default_rng(seed)

    def samples(mean, std, n):
        return rng.normal(mean, std, (n, len(mean)))

    # Whole grain: area besar, aspect ratio sedang, solidity & extent tinggi
    whole = samples(
        mean=[4500, 280, 2.2, 0.68, 0.88],
        std= [ 800,  40, 0.4, 0.06, 0.04],
        n=n_per_class
    )

    # Broken grain: area lebih kecil, aspect ratio lebih pendek/tidak konsisten
    broken = samples(
        mean=[1800, 175, 1.6, 0.58, 0.80],
        std= [ 500,  30, 0.5, 0.08, 0.06],
        n=n_per_class
    )

    # Impurity: area kecil, bentuk tidak beraturan, solidity rendah
    impurity = samples(
        mean=[ 600, 100, 1.3, 0.45, 0.68],
        std= [ 200,  25, 0.4, 0.10, 0.08],
        n=n_per_class
    )

    X = np.vstack([whole, broken, impurity])
    y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    # Clip supaya tidak ada nilai negatif/tidak realistis
    X[:, 0] = np.clip(X[:, 0], 100, 60000)   # area
    X[:, 1] = np.clip(X[:, 1], 30, 2000)      # perimeter
    X[:, 2] = np.clip(X[:, 2], 1.0, 10.0)     # aspect_ratio
    X[:, 3] = np.clip(X[:, 3], 0.1, 1.0)      # extent
    X[:, 4] = np.clip(X[:, 4], 0.3, 1.0)      # solidity

    return X, y


# ─────────────────────────────────────────────
# Load data dari CSV
# ─────────────────────────────────────────────
def _load_csv_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load fitur dari CSV training data.
    Return (X, y) numpy arrays, atau None jika tidak memenuhi syarat.

    Kolom CSV yang dipakai: area, perimeter, aspect_ratio, extent, solidity
    (color_mean dikecualikan supaya konsisten dengan fitur dari run_pipeline)
    """
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    # Validasi kolom
    required = ["area", "perimeter", "aspect_ratio", "extent", "solidity", "label"]
    if not all(c in df.columns for c in required):
        return None

    # Filter label valid saja
    df = df[df["label"].isin(LABEL_MAP.keys())].copy()

    # Cek kecukupan sampel per kelas
    for lbl in LABEL_MAP:
        if df[df["label"] == lbl].shape[0] < MIN_SAMPLES_PER_CLASS:
            return None  # Belum cukup → pakai synthetic

    feat_cols = ["area", "perimeter", "aspect_ratio", "extent", "solidity"]
    X = df[feat_cols].values.astype(np.float32)
    y = df["label"].map(LABEL_MAP).values.astype(int)

    return X, y


def get_data_source(csv_path: Path | None = None) -> str:
    """
    Return string sumber data yang digunakan: 'csv_real' atau 'synthetic'.
    Berguna untuk ditampilkan di UI.
    """
    path = csv_path or _DEFAULT_CSV
    result = _load_csv_data(path)
    if result is None:
        return "synthetic"
    X, y = result
    # Cek apakah ada data non-synthetic (source != synthetic)
    try:
        df = pd.read_csv(path)
        if "source" in df.columns:
            non_synth = df[df["source"] != "synthetic"]
            return "csv_real" if len(non_synth) >= MIN_SAMPLES_PER_CLASS * 3 else "synthetic"
    except Exception:
        pass
    return "csv_real"


# ─────────────────────────────────────────────
# Build model pipelines
# ─────────────────────────────────────────────
def build_knn(n_neighbors: int = 5) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean"))
    ])

def build_svm(C: float = 1.0, kernel: str = "rbf") -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(C=C, kernel=kernel, probability=True, random_state=42))
    ])


# ─────────────────────────────────────────────
# RiceClassifier
# ─────────────────────────────────────────────
class RiceClassifier:
    """
    Wrapper yang menyimpan KNN dan SVM, keduanya dilatih saat init.

    Prioritas data:
      1. CSV di data/training_data.csv (jika ada >= MIN_SAMPLES_PER_CLASS per kelas)
      2. Synthetic data sebagai fallback

    Bisa dipakai untuk prediksi per-butir maupun batch.
    Mendukung retrain via retrain_from_csv() — dipanggil dari UI.
    """

    def __init__(self, knn_k: int = 5, svm_C: float = 1.0, svm_kernel: str = "rbf",
                 csv_path: str | None = None):
        self.knn_k      = knn_k
        self.svm_C      = svm_C
        self.svm_kernel = svm_kernel
        self.csv_path   = Path(csv_path) if csv_path else _DEFAULT_CSV

        self._knn: Pipeline | None = None
        self._svm: Pipeline | None = None
        self._trained       = False
        self._data_source   = "synthetic"   # 'synthetic' atau 'csv_real'
        self._n_samples     = 0
        self._n_per_class   = {}

        # Langsung train saat init
        self._train()

    def _train(self, X: np.ndarray | None = None, y: np.ndarray | None = None):
        """
        Internal training. Jika X, y disediakan → pakai itu.
        Jika tidak → coba load CSV → fallback synthetic.
        """
        if X is None or y is None:
            csv_data = _load_csv_data(self.csv_path)
            if csv_data is not None:
                X, y = csv_data
                self._data_source = "csv_real"
            else:
                X, y = _generate_training_data()
                self._data_source = "synthetic"
        else:
            self._data_source = "csv_real"

        self._knn = build_knn(self.knn_k)
        self._svm = build_svm(self.svm_C, self.svm_kernel)
        self._knn.fit(X, y)
        self._svm.fit(X, y)
        self._trained     = True
        self._n_samples   = len(y)
        self._n_per_class = {
            LABEL_DECODE[lbl]: int(np.sum(y == lbl))
            for lbl in LABEL_DECODE
        }

    def retrain_from_csv(self, csv_path: str | None = None) -> dict:
        """
        Retrain model dari CSV terbaru.
        Dipanggil dari UI Streamlit setelah labeling manual / import Kaggle.

        Returns:
            dict dengan info training: data_source, n_samples, n_per_class, success
        """
        path = Path(csv_path) if csv_path else self.csv_path
        csv_data = _load_csv_data(path)
        if csv_data is not None:
            X, y = csv_data
            self._train(X, y)
            return {
                "success":      True,
                "data_source":  self._data_source,
                "n_samples":    self._n_samples,
                "n_per_class":  self._n_per_class,
                "message":      f"✅ Model dilatih ulang dari {self._n_samples} sampel real data.",
            }
        else:
            # Fallback synthetic
            self._train()
            return {
                "success":      False,
                "data_source":  "synthetic",
                "n_samples":    self._n_samples,
                "n_per_class":  self._n_per_class,
                "message":      f"⚠️ Data CSV belum cukup (min {MIN_SAMPLES_PER_CLASS} per kelas). "
                                f"Menggunakan synthetic data.",
            }

    def get_training_info(self) -> dict:
        """Return informasi singkat tentang data training yang dipakai."""
        return {
            "data_source":  self._data_source,
            "n_samples":    self._n_samples,
            "n_per_class":  self._n_per_class,
            "csv_path":     str(self.csv_path),
        }

    def predict(
        self,
        features: np.ndarray,
        method: str = "svm"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prediksi label untuk array fitur [N x 5].
        Returns (labels_str, probabilities [N x 3])
        """
        if not self._trained:
            raise RuntimeError("Model belum dilatih.")

        if features.shape[0] == 0:
            return np.array([]), np.array([]).reshape(0, 3)

        model = self._svm if method == "svm" else self._knn
        preds = model.predict(features)
        proba = model.predict_proba(features)

        labels = np.array([LABEL_DECODE[p] for p in preds])
        return labels, proba

    def cv_scores(self, method: str = "svm", cv: int = 5) -> dict:
        """
        Cross-validation score pada data yang dipakai (CSV atau synthetic).
        Untuk ditampilkan di UI.
        """
        csv_data = _load_csv_data(self.csv_path)
        if csv_data is not None:
            X, y = csv_data
            label = "real data"
        else:
            X, y = _generate_training_data()
            label = "synthetic data"

        model = build_svm(self.svm_C, self.svm_kernel) if method == "svm" \
                else build_knn(self.knn_k)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        return {
            "mean":       float(scores.mean()),
            "std":        float(scores.std()),
            "scores":     scores.tolist(),
            "data_label": label,
        }


# ─────────────────────────────────────────────
# Quality scoring
# ─────────────────────────────────────────────
def compute_quality(labels: list[str]) -> dict:
    """
    Hitung statistik kualitas dari list label per butir.
    Returns dict dengan persen tiap kategori + grade akhir.
    """
    total = len(labels)
    if total == 0:
        return {"total": 0, "whole_pct": 0, "broken_pct": 0,
                "impurity_pct": 0, "grade": "Tidak Terdeteksi"}

    whole_n    = labels.count("whole")
    broken_n   = labels.count("broken")
    impurity_n = labels.count("impurity")

    whole_pct    = whole_n    / total * 100
    broken_pct   = broken_n   / total * 100
    impurity_pct = impurity_n / total * 100

    # Grading sesuai proposal: Baik / Sedang / Rendah
    if whole_pct >= 80 and impurity_pct <= 2:
        grade = "Baik 🟢"
    elif whole_pct >= 60 and impurity_pct <= 5:
        grade = "Sedang 🟡"
    else:
        grade = "Rendah 🔴"

    return {
        "total":        total,
        "whole_n":      whole_n,
        "broken_n":     broken_n,
        "impurity_n":   impurity_n,
        "whole_pct":    round(whole_pct, 1),
        "broken_pct":   round(broken_pct, 1),
        "impurity_pct": round(impurity_pct, 1),
        "grade":        grade,
    }
