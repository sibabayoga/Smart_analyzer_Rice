"""
models/classifier.py
KNN dan SVM classifier untuk klasifikasi butir beras.
Menggunakan synthetic training data berbasis rule-based labeling dari proposal.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings("ignore")

# Label encoding
LABEL_MAP    = {"whole": 0, "broken": 1, "impurity": 2}
LABEL_DECODE = {0: "whole", 1: "broken", 2: "impurity"}

# ─────────────────────────────────────────────
# Synthetic training data generator
# Dibuat berdasarkan karakteristik geometris dari literatur
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
# Build & train models
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


class RiceClassifier:
    """
    Wrapper yang menyimpan KNN dan SVM, keduanya dilatih saat init.
    Bisa dipakai untuk prediksi per-butir maupun batch.
    """

    def __init__(self, knn_k: int = 5, svm_C: float = 1.0, svm_kernel: str = "rbf"):
        self.knn_k      = knn_k
        self.svm_C      = svm_C
        self.svm_kernel = svm_kernel

        self._knn: Pipeline | None = None
        self._svm: Pipeline | None = None
        self._trained = False

        # Langsung train saat init
        self._train()

    def _train(self):
        X, y = _generate_training_data()

        self._knn = build_knn(self.knn_k)
        self._svm = build_svm(self.svm_C, self.svm_kernel)

        self._knn.fit(X, y)
        self._svm.fit(X, y)
        self._trained = True

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
        """Cross-validation score pada synthetic data (untuk tampil di UI)."""
        X, y = _generate_training_data()
        model = build_svm(self.svm_C, self.svm_kernel) if method == "svm" \
                else build_knn(self.knn_k)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        return {
            "mean":   float(scores.mean()),
            "std":    float(scores.std()),
            "scores": scores.tolist(),
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
