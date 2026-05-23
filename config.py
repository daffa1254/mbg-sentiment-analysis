"""
=============================================================
  config.py — Konfigurasi Global Proyek ABSA MBG
=============================================================
  Semua parameter (path, hyperparameter, threshold, dll.)
  dipusatkan di sini agar mudah diubah tanpa menyentuh
  kode utama.
=============================================================
"""

import os

# ─────────────────────────────────────────────────────────────
# PATH KONFIGURASI
# Gunakan os.path agar berjalan di Windows maupun Linux/macOS
# ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "komentarMBG_YT.csv")   # dataset utama
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")               # grafik & CSV hasil
MODEL_DIR   = os.path.join(BASE_DIR, "models")               # model & artefak tersimpan
LOG_DIR     = os.path.join(BASE_DIR, "logs")                 # log aplikasi

# Buat folder jika belum ada
for _d in [OUTPUT_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# MLFLOW KONFIGURASI
# ─────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI  = "file:///" + os.path.join(BASE_DIR, "mlruns").replace("\\", "/")  # simpan lokal
MLFLOW_EXPERIMENT    = "ABSA_MBG_YouTube"                # nama eksperimen
MLFLOW_RUN_NAME      = "AttentionXGBoost_v1"             # nama run

# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────
# Kata-kata sentimen yang TIDAK boleh dihapus saat stopword removal
SENTIMENT_PRESERVE = {
    "tidak", "bukan", "jangan", "belum", "tanpa", "kurang",
    "bagus", "baik", "buruk", "jelek", "enak", "setuju",
    "menolak", "dukung", "kecewa", "senang", "sedih", "marah",
    "puas", "berhasil", "gagal", "sukses", "sehat", "kotor",
    "bersih", "aman", "bahaya", "baik", "buruk",
}

# ─────────────────────────────────────────────────────────────
# TFIDF KONFIGURASI
# ─────────────────────────────────────────────────────────────
TFIDF_PARAMS = {
    "ngram_range" : (1, 2),   # unigram + bigram
    "max_features": 6000,     # batas fitur
    "sublinear_tf": True,     # gunakan log(tf) agar tidak bias kata frekuensi tinggi
    "min_df"      : 2,        # abaikan kata yang hanya muncul 1x
    "max_df"      : 0.95,     # abaikan kata yang ada di >95% dokumen
}

# ─────────────────────────────────────────────────────────────
# XGBOOST HYPERPARAMETER
# Parameter ini di-log otomatis oleh MLflow
# ─────────────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators"     : 300,
    "max_depth"        : 6,
    "learning_rate"    : 0.08,
    "subsample"        : 0.85,
    "colsample_bytree" : 0.75,
    "min_child_weight" : 3,
    "gamma"            : 0.1,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 1.5,
    "eval_metric"      : "mlogloss",
    "random_state"     : 42,
    "n_jobs"           : 1,
    "tree_method"      : "hist",   # lebih cepat dari "exact"
    
}

# ─────────────────────────────────────────────────────────────
# TRAINING KONFIGURASI
# ─────────────────────────────────────────────────────────────
TEST_SIZE   = 0.2    # proporsi data uji
CV_FOLDS    = 5      # jumlah lipatan cross-validation
RANDOM_SEED = 42     # seed reproduksibilitas

# ─────────────────────────────────────────────────────────────
# FLASK DEPLOYMENT
# ─────────────────────────────────────────────────────────────
FLASK_HOST  = "0.0.0.0"    # dengarkan semua interface
FLASK_PORT  = 5000
FLASK_DEBUG = False         # matikan debug saat produksi

# ─────────────────────────────────────────────────────────────
# LABEL KELAS SENTIMEN
# ─────────────────────────────────────────────────────────────
SENTIMENT_LABELS = ["negatif", "netral", "positif"]  # urutan sesuai LabelEncoder

# ─────────────────────────────────────────────────────────────
# ASPEK YANG DIANALISIS
# ─────────────────────────────────────────────────────────────
ASPECT_LABELS = [
    "anggaran",
    "distribusi",
    "guru_sekolah",
    "kualitas_makanan",
    "program",
    "umum",
]
