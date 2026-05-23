import numpy as np
import pandas as pd
import logging
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from config import TFIDF_PARAMS, MODEL_DIR
from src.labeler import POSITIVE_LEXICON, NEGATIVE_LEXICON, NEGATION_WORDS

logger = logging.getLogger(__name__)


 
# FUNGSI ATTENTION


def compute_attention_weights(tfidf_matrix: np.ndarray) -> np.ndarray:
    
    # Stabilisasi: kurangi nilai maksimum per baris (log-sum-exp trick)
    shifted  = tfidf_matrix - tfidf_matrix.max(axis=1, keepdims=True)
    exp_vals = np.exp(shifted)

    # Softmax normalisasi per baris
    attention = exp_vals / (exp_vals.sum(axis=1, keepdims=True) + 1e-9)

    return attention


def apply_attention(tfidf_matrix: np.ndarray) -> np.ndarray:
    
    attention = compute_attention_weights(tfidf_matrix)
    attended  = tfidf_matrix * attention

    logger.debug(
        "Attention applied. Shape: %s, Non-zero: %d",
        attended.shape,
        np.count_nonzero(attended),
    )
    return attended



# HANDCRAFTED FEATURES

def extract_handcrafted_features(texts: pd.Series) -> np.ndarray:


    def _featurize(text: str) -> list:
        words = text.split() if text else []
        n = len(words) + 1e-9  # hindari pembagian nol

        # Hitung kata sentimen menggunakan set intersection (O(1))
        word_set  = set(words)
        pos_count = len(word_set & POSITIVE_LEXICON)
        neg_count = len(word_set & NEGATIVE_LEXICON)
        neg_gate  = len(word_set & NEGATION_WORDS)

        return [
            len(text),                        # 0: panjang teks
            len(words),                       # 1: jumlah kata
            pos_count / n,                    # 2: rasio kata positif
            neg_count / n,                    # 3: rasio kata negatif
            neg_gate,                         # 4: jumlah kata negasi
            (pos_count - neg_count) / n,      # 5: skor sentimen bersih
        ]

    features = np.array([_featurize(t) for t in texts])
    logger.debug("Handcrafted features shape: %s", features.shape)
    return features



# KELAS FEATURE BUILDER UTAMA

class FeatureBuilder:
  

    def __init__(self):
        """Inisialisasi TF-IDF vectorizer dengan parameter dari config."""
        self.tfidf     = TfidfVectorizer(**TFIDF_PARAMS)
        self.le_label  = LabelEncoder()   # encode label sentimen
        self.le_aspect = None             # diisi saat fit
        self._fitted   = False            # flag apakah sudah di-fit

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """
        Fit vectorizer & encoder pada data training, kemudian
        transformasi menjadi matriks fitur.

        Args:
            df: DataFrame dengan kolom 'text_clean', 'sentimen', 'aspek'

        Returns:
            X : numpy array fitur (n_samples × n_features_total)
            y : numpy array label terenkode
        """
        logger.info("Membangun fitur training ...")

        # ── 1. TF-IDF ────────────────────────────────────────
        tfidf_matrix = self.tfidf.fit_transform(df["text_clean"]).toarray()
        logger.info("TF-IDF shape: %s", tfidf_matrix.shape)

        # ── 2. Attention-Weighted TF-IDF ──────────────────────
        X_attended = apply_attention(tfidf_matrix)

        # ── 3. Handcrafted Features ───────────────────────────
        X_hand = extract_handcrafted_features(df["text_clean"])

        # ── 4. Aspek One-Hot Encoding ─────────────────────────
        X_asp = pd.get_dummies(df["aspek"], prefix="asp").values
        self._asp_columns = pd.get_dummies(df["aspek"], prefix="asp").columns.tolist()

        # ── 5. Gabungkan semua fitur (horizontal stack) ────────
        X = np.hstack([X_attended, X_hand, X_asp])
        logger.info("Total fitur gabungan: %d", X.shape[1])

        # ── 6. Encode label ───────────────────────────────────
        y = self.le_label.fit_transform(df["sentimen"])
        logger.info("Kelas label: %s", list(self.le_label.classes_))

        self._fitted = True
        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
       
        if not self._fitted:
            raise RuntimeError(
                "FeatureBuilder belum di-fit. Panggil fit_transform() dahulu."
            )

        # TF-IDF transform (gunakan kosakata dari training)
        tfidf_matrix = self.tfidf.transform(df["text_clean"]).toarray()
        X_attended   = apply_attention(tfidf_matrix)
        X_hand       = extract_handcrafted_features(df["text_clean"])

        # One-hot aspek: pastikan kolom sama dengan training
        asp_dummies = pd.get_dummies(df["aspek"], prefix="asp")
        for col in self._asp_columns:
            if col not in asp_dummies.columns:
                asp_dummies[col] = 0           # kolom baru diisi 0
        asp_dummies = asp_dummies[self._asp_columns]  # urutkan kolom

        X = np.hstack([X_attended, X_hand, asp_dummies.values])
        return X

    def save(self, path: str = None):
        
        path = path or os.path.join(MODEL_DIR, "feature_builder.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("FeatureBuilder disimpan → %s", path)

    @staticmethod
    def load(path: str = None) -> "FeatureBuilder":
        
        path = path or os.path.join(MODEL_DIR, "feature_builder.pkl")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("FeatureBuilder dimuat dari %s", path)
        return obj
