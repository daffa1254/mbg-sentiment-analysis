"""
=============================================================
  train.py — Pipeline Training + MLflow Tracking
=============================================================
  Menjalankan seluruh alur:
    1. Load & validasi data
    2. EDA (Exploratory Data Analysis)
    3. Preprocessing teks
    4. Pelabelan sentimen & aspek
    5. Rekayasa fitur (TF-IDF + Attention)
    6. SMOTE (handle class imbalance)
    7. Training Attention-Enhanced XGBoost
    8. Evaluasi (accuracy, F1, CV, confusion matrix)
    9. Logging semua metrik & artefak ke MLflow
   10. Simpan model & komponen ke disk

  Cara menjalankan:
    python train.py
    python train.py --no-mlflow   # tanpa MLflow tracking
    python train.py --sample 3000 # gunakan subset data
=============================================================
"""

# ─────────────────────────────────────────────────────────────
# 0. AUTO-INSTALL DEPENDENCIES
# Memastikan semua library terinstal sebelum import utama.
# ─────────────────────────────────────────────────────────────
import subprocess
import sys


def install_if_missing(packages: dict):
    """
    Install package jika belum ada.

    Args:
        packages: dict {nama_pip: nama_import}
    """
    import importlib
    for pip_name, import_name in packages.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"  [INSTALL] Menginstal {pip_name} ...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name, "-q"]
            )


# Mapping: {nama package pip} → {nama modul saat import}
REQUIRED_PACKAGES = {
    "xgboost"        : "xgboost",
    "PySastrawi"     : "Sastrawi",
    "nltk"           : "nltk",
    "scikit-learn"   : "sklearn",
    "mlflow"         : "mlflow",
    "imbalanced-learn": "imblearn",
    "wordcloud"      : "wordcloud",
    "seaborn"        : "seaborn",
    "matplotlib"     : "matplotlib",
}

print("=" * 65)
print("  ABSA MBG — Aspect-Based Sentiment Analysis")
print("  Attention-Enhanced XGBoost + MLflow")
print("=" * 65)
print("\n[0] Memeriksa dependensi ...")
install_if_missing(REQUIRED_PACKAGES)

# ─────────────────────────────────────────────────────────────
# IMPORT UTAMA
# ─────────────────────────────────────────────────────────────
import os
import sys
import re
import time
import logging
import warnings
import argparse
import pickle
import json
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend untuk server/CI
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE

# Tambahkan root project ke sys.path agar modul src/ bisa diimport
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_PATH, OUTPUT_DIR, MODEL_DIR, LOG_DIR,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, MLFLOW_RUN_NAME,
    XGB_PARAMS, TEST_SIZE, CV_FOLDS, RANDOM_SEED,
)
from src.preprocessor import IndonesianPreprocessor
from src.labeler import SentimentLabeler
from src.feature_engineering import FeatureBuilder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# LOGGING SETUP
# Log ke file DAN konsol secara bersamaan
# ─────────────────────────────────────────────────────────────
log_path = os.path.join(LOG_DIR, "train.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("train")


# ═════════════════════════════════════════════════════════════
# FUNGSI BANTU VISUALISASI
# ═════════════════════════════════════════════════════════════

def plot_eda(df: pd.DataFrame) -> str:
    """
    Buat grafik EDA (4 panel):
        1. Histogram panjang komentar (karakter)
        2. Histogram jumlah kata per komentar
        3. Bar chart 20 kata paling sering (sebelum preprocessing)
        4. Histogram distribusi like count

    Returns:
        path file gambar yang disimpan
    """
    logger.info("Membuat grafik EDA ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "EDA — Komentar YouTube Program MBG",
        fontsize=14, fontweight="bold"
    )

    # Panel 1: distribusi panjang karakter
    axes[0, 0].hist(
        df["text_len"], bins=40,
        color="#4C72B0", edgecolor="white", alpha=0.85
    )
    axes[0, 0].set_title("Distribusi Panjang Komentar")
    axes[0, 0].set_xlabel("Jumlah Karakter")
    axes[0, 0].set_ylabel("Frekuensi")

    # Panel 2: distribusi jumlah kata
    axes[0, 1].hist(
        df["word_count"], bins=40,
        color="#DD8452", edgecolor="white", alpha=0.85
    )
    axes[0, 1].set_title("Distribusi Jumlah Kata")
    axes[0, 1].set_xlabel("Jumlah Kata")

    # Panel 3: top 20 kata (teks mentah)
    all_words  = " ".join(df["text"].astype(str)).lower().split()
    top_20     = Counter(all_words).most_common(20)
    words, cnt = zip(*top_20)
    axes[1, 0].barh(words[::-1], cnt[::-1], color="#55A868")
    axes[1, 0].set_title("Top 20 Kata (Sebelum Preprocessing)")
    axes[1, 0].set_xlabel("Frekuensi")

    # Panel 4: distribusi like count (clip atas 50 agar tidak skewed)
    axes[1, 1].hist(
        df["like_count"].clip(upper=50), bins=30,
        color="#C44E52", edgecolor="white", alpha=0.85
    )
    axes[1, 1].set_title("Distribusi Like Count (Clip @50)")
    axes[1, 1].set_xlabel("Like Count")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_eda.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_label_distribution(df: pd.DataFrame) -> str:
    """
    Visualisasi distribusi sentimen dan aspek setelah pelabelan.

    Returns:
        path file gambar
    """
    logger.info("Membuat grafik distribusi label ...")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Distribusi Label — Sentimen & Aspek",
        fontsize=13, fontweight="bold"
    )

    sent_color = {"positif": "#2ecc71", "negatif": "#e74c3c", "netral": "#f39c12"}
    sent_counts = df["sentimen"].value_counts()

    # Bar chart sentimen
    bars = axes[0].bar(
        sent_counts.index,
        sent_counts.values,
        color=[sent_color.get(s, "#aaa") for s in sent_counts.index],
        edgecolor="white",
    )
    axes[0].set_title("Distribusi Sentimen")
    axes[0].set_ylabel("Jumlah")
    # Tampilkan angka di atas bar
    for bar in bars:
        h = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            h + 20, str(int(h)),
            ha="center", fontweight="bold"
        )

    # Horizontal bar aspek
    asp_counts = df["aspek"].value_counts()
    axes[1].barh(asp_counts.index, asp_counts.values, color="#3498db", edgecolor="white")
    axes[1].set_title("Distribusi Aspek")
    axes[1].set_xlabel("Jumlah")

    # Pie chart proporsi sentimen
    axes[2].pie(
        sent_counts.values,
        labels=sent_counts.index,
        colors=[sent_color.get(s, "#aaa") for s in sent_counts.index],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[2].set_title("Proporsi Sentimen (%)")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "2_label_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_evaluation(
    y_test, y_pred, cv_scores, class_names, acc, f1
) -> str:
    """
    Buat grafik evaluasi model (2 panel):
        1. Confusion matrix (heatmap)
        2. Akurasi per fold cross-validation

    Returns:
        path file gambar
    """
    logger.info("Membuat grafik evaluasi ...")
    cm = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Evaluasi Model — Attention-Enhanced XGBoost\n"
        f"Akurasi: {acc*100:.2f}%  |  F1: {f1*100:.2f}%",
        fontsize=12, fontweight="bold"
    )

    # Confusion matrix heatmap
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0], linewidths=0.5,
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Prediksi")
    axes[0].set_ylabel("Aktual")

    # Bar chart CV per fold
    fold_labels = [f"Fold {i+1}" for i in range(len(cv_scores))]
    axes[1].bar(fold_labels, cv_scores * 100, color="#3498db", edgecolor="white")
    axes[1].axhline(
        y=cv_scores.mean() * 100, color="red",
        linestyle="--", label=f"Mean: {cv_scores.mean()*100:.2f}%"
    )
    axes[1].set_ylim(50, 105)
    axes[1].set_title("5-Fold Cross Validation Accuracy")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    for i, v in enumerate(cv_scores):
        axes[1].text(i, v * 100 + 0.5, f"{v*100:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "3_evaluasi.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_absa(df: pd.DataFrame) -> str:
    """
    Visualisasi Aspect-Based Sentiment Analysis:
    Bar chart bertumpuk sentimen per aspek.

    Returns:
        path file gambar
    """
    logger.info("Membuat grafik ABSA ...")
    absa = (
        df.groupby(["aspek", "sentimen"])
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    absa.plot(
        kind="bar", ax=ax,
        color={"negatif": "#e74c3c", "netral": "#f39c12", "positif": "#2ecc71"},
        edgecolor="white", width=0.7,
    )
    ax.set_title(
        "Aspect-Based Sentiment Analysis — Program MBG\n"
        "(Distribusi Sentimen per Aspek)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Aspek")
    ax.set_ylabel("Jumlah Komentar")
    ax.legend(title="Sentimen")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "4_absa_per_aspek.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_wordcloud(df: pd.DataFrame) -> str:
    """
    Buat WordCloud untuk setiap kelas sentimen.
    Menampilkan kata-kata paling sering dalam komentar
    positif, negatif, dan netral.

    Returns:
        path file gambar (atau None jika wordcloud tidak tersedia)
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("wordcloud tidak tersedia, skip plot wordcloud.")
        return None

    logger.info("Membuat WordCloud ...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("WordCloud per Sentimen", fontsize=12, fontweight="bold")

    cmap_dict = {"positif": "Greens", "negatif": "Reds", "netral": "Oranges"}
    for ax, sent in zip(axes, ["positif", "negatif", "netral"]):
        corpus = " ".join(df[df["sentimen"] == sent]["text_clean"].dropna())
        if not corpus.strip():
            ax.set_visible(False)
            continue
        wc = WordCloud(
            width=500, height=300,
            background_color="white",
            colormap=cmap_dict[sent],
            max_words=80,
        ).generate(corpus)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        n = (df["sentimen"] == sent).sum()
        ax.set_title(f"Sentimen: {sent.capitalize()} ({n:,} komentar)")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "5_wordcloud.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ═════════════════════════════════════════════════════════════
# PIPELINE UTAMA
# ═════════════════════════════════════════════════════════════

def run_pipeline(use_mlflow: bool = True, sample_size: int = None):
    """
    Jalankan pipeline training lengkap.

    Args:
        use_mlflow  : Apakah log ke MLflow tracking server
        sample_size : Jika diisi, gunakan subset data (untuk testing cepat)
    """

    # ── STEP 1: Load Data ────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 1: MEMUAT DATASET")
    logger.info("=" * 55)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}\n"
            "Pastikan file CSV ada di direktori yang sama dengan train.py"
        )

    df = pd.read_csv(DATA_PATH)
    logger.info("Data dimuat: %d baris, %d kolom", len(df), len(df.columns))

    # Gunakan subset jika diminta (untuk debug / testing cepat)
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=RANDOM_SEED)
        logger.info("Mode sample: menggunakan %d baris", len(df))

    # Hapus baris dengan teks kosong
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)

    # Tambah kolom statistik untuk EDA
    df["text_len"]   = df["text"].apply(len)
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))

    logger.info("Dataset final: %d baris", len(df))

    # ── STEP 2: EDA ──────────────────────────────────────────
    logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
    logger.info("  Rata-rata panjang  : %.1f karakter", df["text_len"].mean())
    logger.info("  Rata-rata kata     : %.1f kata", df["word_count"].mean())
    logger.info("  Komentar terpanjang: %d karakter", df["text_len"].max())
    logger.info("  Like count max     : %d", df["like_count"].max())

    eda_path = plot_eda(df)

    # ── STEP 3: Preprocessing ────────────────────────────────
    logger.info("STEP 3: PREPROCESSING ...")
    t0 = time.time()

    preprocessor = IndonesianPreprocessor()
    df["text_clean"] = preprocessor.transform_batch(df["text"].tolist())

    # Hapus teks yang menjadi kosong setelah preprocessing
    df = df[df["text_clean"].str.strip().ne("")].reset_index(drop=True)
    logger.info(
        "Preprocessing selesai (%.1fs). Sisa data: %d baris",
        time.time() - t0, len(df)
    )

    # ── STEP 4: Labeling ─────────────────────────────────────
    logger.info("STEP 4: PELABELAN SENTIMEN & ASPEK ...")
    labeler = SentimentLabeler()
    df      = labeler.label_dataframe(df)

    label_path = plot_label_distribution(df)
    plot_absa(df)

    sent_dist = df["sentimen"].value_counts().to_dict()
    logger.info("Distribusi sentimen: %s", sent_dist)

    df = pd.read_csv("D:\freecodecamp\\mbg_project\\Book1.csv")
    # ── STEP 5: Feature Engineering ──────────────────────────
    logger.info("STEP 5: REKAYASA FITUR ...")
    builder = FeatureBuilder()
    X, y    = builder.fit_transform(df)
    logger.info("Matriks fitur: %s", X.shape)

    # ── STEP 6: Split & SMOTE ────────────────────────────────
    logger.info("STEP 6: TRAIN-TEST SPLIT + SMOTE ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # SMOTE: buat sampel sintetis untuk menyeimbangkan kelas minoritas
    # Hanya diterapkan pada data TRAINING (bukan test)
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    bal_dist = Counter(y_train_bal)
    logger.info("Data train setelah SMOTE: %d baris", len(y_train_bal))
    logger.info(
        "Distribusi kelas: %s",
        {builder.le_label.classes_[k]: v for k, v in bal_dist.items()}
    )
    
    

    # ── STEP 7: Training ──────────────────────────────────────
    logger.info("STEP 7: TRAINING ATTENTION-ENHANCED XGBOOST ...")
 
    # Sample weight: pastikan kelas tetap seimbang setelah SMOTE
    sample_w = compute_sample_weight("balanced", y_train_bal)

    model = XGBClassifier(**XGB_PARAMS)
    t0    = time.time()
    model.fit(X_train_bal, y_train_bal, sample_weight=sample_w, verbose=False)
    train_time = time.time() - t0
    logger.info("Training selesai dalam %.1f detik", train_time)

    # ── STEP 8: Evaluasi ──────────────────────────────────────
    logger.info("STEP 8: EVALUASI MODEL ...")
    y_pred = model.predict(X_test)

    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall    = recall_score(y_test, y_pred, average="weighted")

    logger.info("=" * 40)
    logger.info("AKURASI TEST SET : %.2f%%", acc * 100)
    logger.info("F1 SCORE          : %.2f%%", f1 * 100)
    logger.info("PRECISION         : %.2f%%", precision * 100)
    logger.info("RECALL            : %.2f%%", recall * 100)
    logger.info("=" * 40)

    class_names = builder.le_label.classes_
    logger.info(
        "\n%s",
        classification_report(y_test, y_pred, target_names=class_names, digits=4)
    )

    # Cross-Validation pada data penuh (tanpa SMOTE agar tidak bias)
    logger.info("Cross-Validation (%d fold) ...", CV_FOLDS)
    cv_model = XGBClassifier(**XGB_PARAMS)
    cv_scores = cross_val_score(
        cv_model, X, y, cv=CV_FOLDS, scoring="accuracy", n_jobs=1
    )
    logger.info(
        "CV Accuracy: %.2f%% ± %.2f%%",
        cv_scores.mean() * 100, cv_scores.std() * 100
    )

    eval_path = plot_evaluation(y_test, y_pred, cv_scores, class_names, acc, f1)
    wc_path   = plot_wordcloud(df)

    # ── STEP 9: Simpan Model & Artefak ────────────────────────
    logger.info("STEP 9: MENYIMPAN MODEL & ARTEFAK ...")

    # Simpan model XGBoost
    model_path = os.path.join(MODEL_DIR, "xgboost_model.json")
    model.save_model(model_path)

    # Simpan FeatureBuilder (TF-IDF vectorizer + encoder)
    builder.save()

    # Simpan LabelEncoder terpisah untuk deployment
    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(builder.le_label, f)

    # Simpan metadata model
    metadata = {
        "accuracy"     : round(acc, 4),
        "f1_weighted"  : round(f1, 4),
        "precision"    : round(precision, 4),
        "recall"       : round(recall, 4),
        "cv_mean"      : round(cv_scores.mean(), 4),
        "cv_std"       : round(cv_scores.std(), 4),
        "n_samples"    : len(df),
        "n_features"   : X.shape[1],
        "classes"      : list(class_names),
        "train_time_s" : round(train_time, 2),
        "xgb_params"   : XGB_PARAMS,
    }
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("Model disimpan -> %s", model_path)

    # Simpan dataset hasil analisis
    result_df   = df[[
        "video_id", "author", "text", "text_clean",
        "sentimen", "aspek", "like_count", "published_at"
    ]]
    result_path = os.path.join(OUTPUT_DIR, "hasil_sentimen_absa.csv")
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    logger.info("Hasil disimpan -> %s", result_path)

    # ── STEP 10: MLflow Logging ───────────────────────────────
    if use_mlflow:
        logger.info("STEP 10: MLFLOW LOGGING ...")
        _log_to_mlflow(
            model=model,
            builder=builder,
            metadata=metadata,
            cv_scores=cv_scores,
            params=XGB_PARAMS,
            artifacts={
                "eda"          : eda_path,
                "label_dist"   : label_path,
                "evaluasi"     : eval_path,
                "wordcloud"    : wc_path,
                "model_json"   : model_path,
                "feature_builder": os.path.join(MODEL_DIR, "feature_builder.pkl"),
                "label_encoder": le_path,
                "metadata"     : meta_path,
                "hasil_csv"    : result_path,
            }
        )

    # ── RINGKASAN AKHIR ───────────────────────────────────────
    logger.info("=" * 55)
    logger.info("  RINGKASAN HASIL TRAINING")
    logger.info("=" * 55)
    logger.info("  Dataset             : %d komentar", len(df))
    logger.info("  Akurasi Test Set    : %.2f%%", acc * 100)
    logger.info("  F1-Score (weighted) : %.2f%%", f1 * 100)
    logger.info("  CV Mean Accuracy    : %.2f%% ± %.2f%%",
                cv_scores.mean() * 100, cv_scores.std() * 100)
    logger.info("  Output folder       : %s", OUTPUT_DIR)
    logger.info("  Model folder        : %s", MODEL_DIR)
    if use_mlflow:
        logger.info(
            "  MLflow UI           : jalankan 'mlflow ui' lalu buka http://localhost:5000"
        )
    logger.info("=" * 55)
    logger.info("  ✅ TRAINING SELESAI!")
    logger.info("=" * 55)

    return model, builder, metadata


def _log_to_mlflow(model, builder, metadata, cv_scores, params, artifacts):
    """
    Log semua parameter, metrik, dan artefak ke MLflow Tracking.

    MLflow menyimpan:
    - Parameters : semua hyperparameter XGBoost
    - Metrics    : accuracy, f1, precision, recall, CV scores
    - Artifacts  : grafik PNG, model JSON, feature builder PKL, CSV
    - Model      : model XGBoost dengan signature
    """
    # Set tracking URI ke folder lokal 'mlruns'
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME) as run:
        logger.info("MLflow Run ID: %s", run.info.run_id)

        # ── Log hyperparameter XGBoost ──────────────────────
        # Setiap parameter tercatat untuk reproduksibilitas eksperimen
        for key, val in params.items():
            mlflow.log_param(key, val)

        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("n_samples", metadata["n_samples"])
        mlflow.log_param("n_features", metadata["n_features"])
        mlflow.log_param("smote_enabled", True)

        # ── Log metrik evaluasi ─────────────────────────────
        mlflow.log_metric("accuracy_test",   metadata["accuracy"])
        mlflow.log_metric("f1_weighted",     metadata["f1_weighted"])
        mlflow.log_metric("precision",       metadata["precision"])
        mlflow.log_metric("recall",          metadata["recall"])
        mlflow.log_metric("cv_mean_accuracy", metadata["cv_mean"])
        mlflow.log_metric("cv_std",          metadata["cv_std"])
        mlflow.log_metric("train_time_s",    metadata["train_time_s"])

        # Log CV per fold untuk analisis variance
        for i, score in enumerate(cv_scores, start=1):
            mlflow.log_metric(f"cv_fold_{i}", float(score))

        # ── Log artefak (file) ──────────────────────────────
        # Artefak tersimpan di folder mlruns/ dan bisa dilihat di UI
        for name, path in artifacts.items():
            if path and os.path.exists(path):
                mlflow.log_artifact(path)
                logger.info("  Artefak di-log: %s", os.path.basename(path))

        # ── Log model XGBoost ke MLflow Model Registry ──────
        # Ini memungkinkan model di-load ulang via mlflow.xgboost.load_model()
        try:
            from mlflow.models import infer_signature
            # Buat sample input untuk signature
            import numpy as np
            sample_input = np.zeros((1, metadata["n_features"]))
            signature    = infer_signature(sample_input, model.predict(sample_input))

            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="xgboost_model",
                signature=signature,
                registered_model_name="ABSA_MBG_XGBoost",
            )
            logger.info("  Model di-log ke MLflow Model Registry")
        except Exception as e:
            logger.warning("Gagal log model ke registry: %s", e)
            # Fallback: log sebagai artefak biasa
            mlflow.log_artifact(artifacts.get("model_json", ""))

        logger.info(
            "MLflow logging selesai. Run ID: %s", run.info.run_id
        )


# ═════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Parsing argumen command line
    parser = argparse.ArgumentParser(
        description="Training ABSA MBG dengan Attention-Enhanced XGBoost"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Jalankan tanpa MLflow tracking",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Gunakan subset N baris data (untuk testing cepat)",
    )
    args = parser.parse_args()

    run_pipeline(
        use_mlflow=not args.no_mlflow,
        sample_size=args.sample,
    )
