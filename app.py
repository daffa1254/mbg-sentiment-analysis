import os
import sys
import json
import time
import pickle
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


# ============================================================
# IMPORT PROJECT
# ============================================================

try:
    from config import MODEL_DIR
    from src.preprocessor import IndonesianPreprocessor
    from src.labeler import SentimentLabeler
    from src.feature_engineering import FeatureBuilder
except Exception:
    st.set_page_config(
        page_title="MBG Sentiment Analysis",
        page_icon="🍱",
        layout="centered",
    )

    st.error("Gagal import file project.")
    st.code(traceback.format_exc())
    st.stop()


# ============================================================
# PATH MODEL
# ============================================================

def resolve_path(path_value):
    path = Path(path_value)

    if path.is_absolute():
        return path

    return BASE_DIR / path


MODEL_PATH = resolve_path(MODEL_DIR)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_models():
    components = {}

    try:
        from xgboost import XGBClassifier

        xgb_path = MODEL_PATH / "xgboost_model.json"
        feature_builder_path = MODEL_PATH / "feature_builder.pkl"
        label_encoder_path = MODEL_PATH / "label_encoder.pkl"
        metadata_path = MODEL_PATH / "model_metadata.json"

        required_files = [
            xgb_path,
            feature_builder_path,
            label_encoder_path,
        ]

        missing_files = []

        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))

        if missing_files:
            raise FileNotFoundError(
                "File model belum lengkap:\n\n"
                + "\n".join(missing_files)
                + "\n\nPastikan file tersebut sudah ada di folder models dan sudah di-commit ke GitHub."
            )

        xgb_model = XGBClassifier()
        xgb_model.load_model(str(xgb_path))
        components["xgb_model"] = xgb_model

        components["feature_builder"] = FeatureBuilder.load(str(feature_builder_path))

        with open(label_encoder_path, "rb") as f:
            components["label_encoder"] = pickle.load(f)

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                components["metadata"] = json.load(f)
        else:
            components["metadata"] = {}

        components["preprocessor"] = IndonesianPreprocessor()
        components["labeler"] = SentimentLabeler()

        return components, None

    except Exception:
        return None, traceback.format_exc()


# ============================================================
# FUNGSI PREDIKSI SATU TEKS
# ============================================================

def predict_single(text, components):
    t0 = time.time()

    text_clean = components["preprocessor"].transform(text)

    if not text_clean.strip():
        return {
            "teks_asli": text,
            "teks_bersih": text_clean,
            "sentimen": "netral",
            "aspek": "umum",
            "confidence": {
                "negatif": 0.0,
                "netral": 1.0,
                "positif": 0.0,
            },
            "waktu_proses": round(time.time() - t0, 4),
        }

    aspek = components["labeler"].get_aspect(text_clean)

    df_single = pd.DataFrame({
        "text_clean": [text_clean],
        "aspek": [aspek],
    })

    X = components["feature_builder"].transform(df_single)

    proba = components["xgb_model"].predict_proba(X)[0]
    pred_id = proba.argmax()

    label_encoder = components["label_encoder"]
    sentimen = label_encoder.inverse_transform([pred_id])[0]

    confidence = {
        str(label_encoder.classes_[i]): round(float(p), 4)
        for i, p in enumerate(proba)
    }

    return {
        "teks_asli": text,
        "teks_bersih": text_clean,
        "sentimen": str(sentimen),
        "aspek": str(aspek),
        "confidence": confidence,
        "waktu_proses": round(time.time() - t0, 4),
    }


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="MBG Sentiment Analysis",
    page_icon="🍱",
    layout="wide",
)

st.title("🍱 MBG Sentiment Analysis")
st.write(
    "Aplikasi analisis sentimen dan aspek komentar terkait "
    "Program Makan Bergizi Gratis."
)


components, error = load_models()

if components is None:
    st.error("Model belum berhasil dimuat.")
    st.write("Detail error:")
    st.code(error)

    st.warning(
        "Pastikan folder `models/` berisi file berikut:\n\n"
        "- `xgboost_model.json`\n"
        "- `feature_builder.pkl`\n"
        "- `label_encoder.pkl`\n"
        "- `model_metadata.json`"
    )

    st.stop()


metadata = components.get("metadata", {})

with st.sidebar:
    st.header("ℹ️ Informasi Model")

    accuracy = metadata.get("accuracy", "N/A")
    f1_weighted = metadata.get("f1_weighted", "N/A")

    st.write("Akurasi:", accuracy)
    st.write("F1 Weighted:", f1_weighted)
    st.write("Kelas Sentimen:")
    st.write(list(components["label_encoder"].classes_))

    st.divider()

    st.write("Status:")
    st.success("Model berhasil dimuat")


tab_single, tab_batch, tab_info = st.tabs([
    "Prediksi Satu Teks",
    "Prediksi Batch",
    "Info Model",
])


# ============================================================
# TAB 1: PREDIKSI SATU TEKS
# ============================================================

with tab_single:
    st.subheader("Prediksi Satu Komentar")

    text = st.text_area(
        "Masukkan komentar:",
        placeholder="Contoh: Makanannya enak, tapi distribusinya masih terlambat.",
        height=160,
    )

    if st.button("Analisis Sentimen", type="primary"):
        if not text.strip():
            st.warning("Komentar tidak boleh kosong.")
        elif len(text) > 5000:
            st.warning("Teks terlalu panjang. Maksimal 5000 karakter.")
        else:
            with st.spinner("Sedang menganalisis..."):
                result = predict_single(text, components)

            st.subheader("Hasil Prediksi")

            col1, col2, col3 = st.columns(3)

            col1.metric("Sentimen", result["sentimen"])
            col2.metric("Aspek", result["aspek"])
            col3.metric("Waktu Proses", f"{result['waktu_proses']} s")

            st.write("Teks Bersih:")
            st.code(result["teks_bersih"])

            st.write("Confidence:")

            confidence_df = pd.DataFrame(
                list(result["confidence"].items()),
                columns=["Label", "Probabilitas"],
            )

            st.dataframe(confidence_df, use_container_width=True)

            st.write("JSON Result:")
            st.json(result)


# ============================================================
# TAB 2: PREDIKSI BATCH
# ============================================================

with tab_batch:
    st.subheader("Prediksi Banyak Komentar")

    batch_text = st.text_area(
        "Masukkan banyak komentar, satu komentar per baris:",
        placeholder=(
            "Program MBG sangat bagus dan bermanfaat\n"
            "Makanannya basi dan tidak layak\n"
            "Distribusinya masih terlambat"
        ),
        height=220,
    )

    if st.button("Analisis Batch"):
        texts = [
            line.strip()
            for line in batch_text.splitlines()
            if line.strip()
        ]

        if len(texts) == 0:
            st.warning("Masukkan minimal satu komentar.")
        elif len(texts) > 100:
            st.warning("Maksimal 100 komentar per batch.")
        else:
            results = []

            progress = st.progress(0)

            with st.spinner("Sedang menganalisis batch..."):
                for idx, item in enumerate(texts):
                    try:
                        result = predict_single(item, components)
                        results.append(result)
                    except Exception as e:
                        results.append({
                            "teks_asli": item,
                            "teks_bersih": "",
                            "sentimen": "error",
                            "aspek": "error",
                            "confidence": {},
                            "waktu_proses": 0,
                            "error": str(e),
                        })

                    progress.progress((idx + 1) / len(texts))

            df_results = pd.DataFrame(results)

            st.subheader("Hasil Batch")
            st.dataframe(df_results, use_container_width=True)

            st.subheader("Ringkasan Sentimen")

            sentiment_summary = (
                df_results["sentimen"]
                .value_counts()
                .reset_index()
            )

            sentiment_summary.columns = ["Sentimen", "Jumlah"]

            st.dataframe(sentiment_summary, use_container_width=True)
            st.bar_chart(
                sentiment_summary,
                x="Sentimen",
                y="Jumlah",
            )

            st.download_button(
                label="Download Hasil CSV",
                data=df_results.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi_mbg.csv",
                mime="text/csv",
            )


# ============================================================
# TAB 3: INFO MODEL
# ============================================================

with tab_info:
    st.subheader("Informasi Model")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Folder model:")
        st.code(str(MODEL_PATH))

        st.write("File yang digunakan:")
        st.code(
            "\n".join([
                "xgboost_model.json",
                "feature_builder.pkl",
                "label_encoder.pkl",
                "model_metadata.json",
            ])
        )

    with col2:
        st.write("Metadata model:")
        st.json(metadata)

    st.subheader("Cek File Model")

    model_files = [
        "xgboost_model.json",
        "feature_builder.pkl",
        "label_encoder.pkl",
        "model_metadata.json",
    ]

    check_data = []

    for filename in model_files:
        file_path = MODEL_PATH / filename
        check_data.append({
            "File": filename,
            "Ada": "Ya" if file_path.exists() else "Tidak",
            "Path": str(file_path),
        })

    st.dataframe(pd.DataFrame(check_data), use_container_width=True)
