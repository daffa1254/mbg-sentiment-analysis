

import os
import sys
import json
import time
import logging
import pickle
import traceback
from datetime import datetime, timezone
import gradio as gr



# Tambahkan root project ke path agar config & src bisa diimport
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from flask import Flask, request, jsonify, render_template_string, abort

from config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    MODEL_DIR, LOG_DIR, SENTIMENT_LABELS,
)
from src.preprocessor import IndonesianPreprocessor
from src.labeler import SentimentLabeler
from src.feature_engineering import FeatureBuilder

# LOGGING SETUP
log_path = os.path.join(LOG_DIR, "api.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("api")

# INISIALISASI FLASK
app = Flask(__name__)

# Statistik penggunaan API (in-memory, reset saat restart)
API_STATS = {
    "total_requests"   : 0,
    "total_predictions": 0,
    "errors"           : 0,
    "start_time"       : datetime.now(timezone.utc).isoformat(),
}


# PEMUATAN MODEL


def load_models():
    
    logger.info("Memuat model dari disk ...")
    components = {}

    try:
        # ── 1. Load XGBoost Model ────────────────────────────
        from xgboost import XGBClassifier
        model_path = os.path.join(MODEL_DIR, "xgboost_model.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model tidak ditemukan: {model_path}\n"
                "Pastikan sudah menjalankan train.py terlebih dahulu."
            )
        xgb_model = XGBClassifier()
        xgb_model.load_model(model_path)
        components["xgb_model"] = xgb_model
        logger.info("  ✔ XGBoost model dimuat")

        # ── 2. Load FeatureBuilder ────────────────────────────
        # FeatureBuilder menyimpan TF-IDF vocabulary & konfigurasi attention
        fb_path = os.path.join(MODEL_DIR, "feature_builder.pkl")
        components["feature_builder"] = FeatureBuilder.load(fb_path)
        logger.info("  ✔ FeatureBuilder dimuat")

        # ── 3. Load LabelEncoder ──────────────────────────────
        le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
        with open(le_path, "rb") as f:
            components["label_encoder"] = pickle.load(f)
        logger.info("  ✔ LabelEncoder dimuat: %s", list(components["label_encoder"].classes_))

        # ── 4. Load Metadata ──────────────────────────────────
        meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                components["metadata"] = json.load(f)
            logger.info(
                "  ✔ Metadata dimuat (akurasi: %.2f%%)",
                components["metadata"].get("accuracy", 0) * 100
            )
        else:
            components["metadata"] = {}

        # ── 5. Inisialisasi Preprocessor & Labeler ───────────
        # Dibuat sekali di sini untuk efisiensi (tidak rekonstruksi setiap request)
        components["preprocessor"] = IndonesianPreprocessor()
        components["labeler"]      = SentimentLabeler()
        logger.info("  ✔ Preprocessor & Labeler siap")

        logger.info("Semua komponen model berhasil dimuat.")
        return components

    except Exception as e:
        logger.error("GAGAL memuat model: %s", e)
        logger.error(traceback.format_exc())
        return None


# Muat model saat server start
COMPONENTS = load_models()


# FUNGSI PREDIKSI INTI


def predict_single(text: str) -> dict:
   
    if not COMPONENTS:
        raise RuntimeError("Model belum dimuat. Jalankan train.py terlebih dahulu.")

    t0 = time.time()

    # 1. Preprocessing: bersihkan teks
    text_clean = COMPONENTS["preprocessor"].transform(text)

    if not text_clean.strip():
        # Teks kosong setelah preprocessing → kembalikan netral
        return {
            "teks_asli"   : text,
            "teks_bersih" : text_clean,
            "sentimen"    : "netral",
            "aspek"       : "umum",
            "confidence"  : {"negatif": 0.0, "netral": 1.0, "positif": 0.0},
            "waktu_proses": round(time.time() - t0, 4),
        }

    # 2. Deteksi aspek
    aspek = COMPONENTS["labeler"].get_aspect(text_clean)

    # 3. Buat DataFrame minimal untuk feature builder
    import pandas as pd
    df_single = pd.DataFrame({
        "text_clean": [text_clean],
        "aspek"     : [aspek],
    })

    # 4. Rekayasa fitur (TF-IDF + Attention + Handcrafted + Aspek OH)
    X = COMPONENTS["feature_builder"].transform(df_single)

    # 5. Prediksi probabilitas per kelas
    proba   = COMPONENTS["xgb_model"].predict_proba(X)[0]
    pred_id = proba.argmax()

    # 6. Decode angka → nama kelas sentimen
    label_enc = COMPONENTS["label_encoder"]
    sentimen  = label_enc.inverse_transform([pred_id])[0]

    # Buat dict confidence {nama_kelas: probabilitas}
    confidence = {
        label_enc.classes_[i]: round(float(p), 4)
        for i, p in enumerate(proba)
    }

    return {
        "teks_asli"   : text,
        "teks_bersih" : text_clean,
        "sentimen"    : sentimen,
        "aspek"       : aspek,
        "confidence"  : confidence,
        "waktu_proses": round(time.time() - t0, 4),
    }


# MIDDLEWARE: Logging setiap request

@app.before_request
def log_request():
    """Catat semua request yang masuk ke log."""
    API_STATS["total_requests"] += 1
    logger.info(
        "REQUEST %s %s dari %s",
        request.method, request.path, request.remote_addr
    )


@app.after_request
def add_cors_headers(response):
    
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ENDPOINT: Halaman Utama

@app.route("/", methods=["GET"])
def index():
    
    status = "✅ Siap" if COMPONENTS else "❌ Model belum dimuat"
    metadata = COMPONENTS.get("metadata", {}) if COMPONENTS else {}

    html = f"""
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <title>ABSA MBG API</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; max-width: 900px;
                   margin: 40px auto; padding: 0 20px; background: #f8f9fa; color: #333; }}
            h1   {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2   {{ color: #2980b9; }}
            .card {{ background: white; border-radius: 8px; padding: 20px;
                    margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .status-ok  {{ color: #27ae60; font-weight: bold; font-size: 1.1em; }}
            .status-err {{ color: #e74c3c; font-weight: bold; font-size: 1.1em; }}
            code {{ background: #ecf0f1; padding: 3px 8px; border-radius: 4px;
                   font-size: 0.9em; color: #2c3e50; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #ddd; }}
            th {{ background: #3498db; color: white; }}
            tr:hover {{ background: #f5f5f5; }}
            .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
                     font-size: 0.85em; font-weight: bold; }}
            .badge-get  {{ background: #2ecc71; color: white; }}
            .badge-post {{ background: #e67e22; color: white; }}
        </style>
    </head>
    <body>
        <h1>🎯 ABSA MBG — Sentiment Analysis API</h1>
        <div class="card">
            <p>Status Model: <span class="{'status-ok' if COMPONENTS else 'status-err'}">{status}</span></p>
            <p>Akurasi Model: <strong>{metadata.get('accuracy', 'N/A') if metadata else 'Belum training'}</strong></p>
            <p>F1-Score: <strong>{metadata.get('f1_weighted', 'N/A') if metadata else '-'}</strong></p>
            <p>Total Request: <strong>{API_STATS['total_requests']}</strong></p>
            <p>Server Aktif: <strong>{API_STATS['start_time']}</strong></p>
        </div>
        <div class="card">
            <h2>📋 Daftar Endpoint</h2>
            <table>
                <tr><th>Method</th><th>URL</th><th>Deskripsi</th></tr>
                <tr><td><span class="badge badge-get">GET</span></td>
                    <td><code>/health</code></td><td>Status kesehatan server</td></tr>
                <tr><td><span class="badge badge-post">POST</span></td>
                    <td><code>/predict</code></td><td>Prediksi sentimen satu teks</td></tr>
                <tr><td><span class="badge badge-post">POST</span></td>
                    <td><code>/predict/batch</code></td><td>Prediksi batch banyak teks</td></tr>
                <tr><td><span class="badge badge-get">GET</span></td>
                    <td><code>/model/info</code></td><td>Metadata & metrik model</td></tr>
                <tr><td><span class="badge badge-get">GET</span></td>
                    <td><code>/docs</code></td><td>Dokumentasi API lengkap</td></tr>
            </table>
        </div>
        <div class="card">
            <h2>⚡ Quick Test</h2>
            <p>Contoh curl:</p>
            <code>curl -X POST http://localhost:{FLASK_PORT}/predict \\<br>
            &nbsp;&nbsp;&nbsp;&nbsp; -H "Content-Type: application/json" \\<br>
            &nbsp;&nbsp;&nbsp;&nbsp; -d '{{"text": "makanan MBG basi dan kotor tidak layak"}}'</code>
        </div>
    </body>
    </html>
    """
    return html


# ENDPOINT: Health Check

@app.route("/health", methods=["GET"])
def health():
    
    is_ready = COMPONENTS is not None
    payload  = {
        "status"       : "ok" if is_ready else "error",
        "model_loaded" : is_ready,
        "uptime_since" : API_STATS["start_time"],
        "timestamp"    : datetime.now(timezone.utc).isoformat(),
    }
    status_code = 200 if is_ready else 503
    return jsonify(payload), status_code


# ENDPOINT: Prediksi Satu Teks

@app.route("/predict", methods=["POST"])
def predict():
    
    if not COMPONENTS:
        return jsonify({
            "status" : "error",
            "message": "Model belum dimuat. Jalankan train.py terlebih dahulu."
        }), 503

    # Validasi request
    if not request.is_json:
        return jsonify({
            "status" : "error",
            "message": "Content-Type harus application/json"
        }), 400

    body = request.get_json()
    text = body.get("text", "").strip() if body else ""

    if not text:
        return jsonify({
            "status" : "error",
            "message": "Field 'text' wajib diisi dan tidak boleh kosong"
        }), 400

    if len(text) > 5000:
        return jsonify({
            "status" : "error",
            "message": "Teks terlalu panjang. Maksimal 5000 karakter."
        }), 400

    # Prediksi
    try:
        result = predict_single(text)
        API_STATS["total_predictions"] += 1
        logger.info(
            "Prediksi: sentimen=%s aspek=%s conf=%.3f",
            result["sentimen"], result["aspek"],
            result["confidence"].get(result["sentimen"], 0)
        )
        return jsonify({"status": "success", "data": result})

    except Exception as e:
        API_STATS["errors"] += 1
        logger.error("Error prediksi: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({
            "status" : "error",
            "message": f"Terjadi kesalahan saat prediksi: {str(e)}"
        }), 500


# ENDPOINT: Prediksi Batch

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    
    if not COMPONENTS:
        return jsonify({
            "status" : "error",
            "message": "Model belum dimuat."
        }), 503

    if not request.is_json:
        return jsonify({
            "status" : "error",
            "message": "Content-Type harus application/json"
        }), 400

    body  = request.get_json()
    texts = body.get("texts", []) if body else []

    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({
            "status" : "error",
            "message": "Field 'texts' harus berupa array dan tidak boleh kosong"
        }), 400

    # Batasi maksimal 100 teks per request untuk mencegah overload
    if len(texts) > 100:
        return jsonify({
            "status" : "error",
            "message": "Maksimal 100 teks per request batch."
        }), 400

    t0      = time.time()
    results = []
    errors  = []

    for i, text in enumerate(texts):
        try:
            r = predict_single(str(text).strip())
            results.append(r)
        except Exception as e:
            # Jangan gagalkan seluruh batch karena satu error
            results.append({
                "teks_asli" : text,
                "error"     : str(e),
            })
            errors.append(i)

    # Hitung ringkasan distribusi sentimen
    from collections import Counter
    ringkasan = Counter(
        r.get("sentimen", "error") for r in results if "sentimen" in r
    )

    total_waktu = round(time.time() - t0, 4)
    API_STATS["total_predictions"] += len(results)

    logger.info(
        "Batch prediksi: %d teks, %.3f detik, ringkasan=%s",
        len(texts), total_waktu, dict(ringkasan)
    )

    return jsonify({
        "status"      : "success",
        "total"       : len(results),
        "total_waktu" : total_waktu,
        "results"     : results,
        "ringkasan"   : dict(ringkasan),
    })


# ENDPOINT: Info Model

@app.route("/model/info", methods=["GET"])
def model_info():
   
    if not COMPONENTS:
        return jsonify({
            "status" : "error",
            "message": "Model belum dimuat."
        }), 503

    return jsonify({
        "status"   : "success",
        "metadata" : COMPONENTS.get("metadata", {}),
        "api_stats": {
            **API_STATS,
            "model_classes": list(COMPONENTS["label_encoder"].classes_),
        }
    })


# ENDPOINT: Dokumentasi API

@app.route("/docs", methods=["GET"])
def docs():
    
    html = """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <title>Dokumentasi ABSA MBG API</title>
        <style>
            body  {{ font-family: 'Segoe UI', sans-serif; max-width: 1000px;
                    margin: 40px auto; padding: 0 20px; background: #f8f9fa; color: #333; }}
            h1    {{ color: #2c3e50; }}
            h2    {{ color: #2980b9; margin-top: 40px; }}
            h3    {{ color: #27ae60; }}
            pre   {{ background: #2c3e50; color: #ecf0f1; padding: 15px;
                    border-radius: 8px; overflow-x: auto; font-size: 0.9em; }}
            code  {{ background: #ecf0f1; padding: 2px 6px; border-radius: 4px; }}
            .card {{ background: white; border-radius: 8px; padding: 20px;
                    margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .method-post {{ color: #e67e22; font-weight: bold; }}
            .method-get  {{ color: #27ae60; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>📖 Dokumentasi ABSA MBG API</h1>

        <div class="card">
            <h2><span class="method-post">POST</span> /predict</h2>
            <p>Prediksi sentimen dan aspek untuk satu teks komentar.</p>
            <h3>Request</h3>
            <pre>curl -X POST http://localhost:5000/predict \\
     -H "Content-Type: application/json" \\
     -d '{"text": "makanan MBG basi dan kotor tidak layak dimakan"}'</pre>
            <h3>Response</h3>
            <pre>{
  "status": "success",
  "data": {
    "teks_asli"   : "makanan MBG basi dan kotor tidak layak dimakan",
    "teks_bersih" : "makan bergizi gratis basi kotor tidak layak makan",
    "sentimen"    : "negatif",
    "aspek"       : "kualitas_makanan",
    "confidence"  : {"negatif": 0.89, "netral": 0.07, "positif": 0.04},
    "waktu_proses": 0.023
  }
}</pre>
        </div>

        <div class="card">
            <h2><span class="method-post">POST</span> /predict/batch</h2>
            <p>Prediksi batch untuk banyak teks sekaligus (maks. 100 teks).</p>
            <h3>Request</h3>
            <pre>curl -X POST http://localhost:5000/predict/batch \\
     -H "Content-Type: application/json" \\
     -d '{
       "texts": [
         "program mbg sangat bagus dan bermanfaat",
         "makanan basi gak layak dimakan",
         "biasa saja"
       ]
     }'</pre>
            <h3>Response</h3>
            <pre>{
  "status"      : "success",
  "total"       : 3,
  "total_waktu" : 0.15,
  "ringkasan"   : {"positif": 1, "negatif": 1, "netral": 1},
  "results"     : [ ... ]
}</pre>
        </div>

        <div class="card">
            <h2><span class="method-get">GET</span> /model/info</h2>
            <p>Informasi metadata dan metrik performa model aktif.</p>
            <h3>Request</h3>
            <pre>curl http://localhost:5000/model/info</pre>
        </div>

        <div class="card">
            <h2><span class="method-get">GET</span> /health</h2>
            <p>Health check — digunakan oleh monitoring tools.</p>
            <h3>Request</h3>
            <pre>curl http://localhost:5000/health</pre>
        </div>

        <div class="card">
            <h2>🏷️ Label Sentimen & Aspek</h2>
            <p><strong>Sentimen:</strong> <code>positif</code> / <code>negatif</code> / <code>netral</code></p>
            <p><strong>Aspek:</strong></p>
            <ul>
                <li><code>kualitas_makanan</code> — kondisi, rasa, kebersihan makanan</li>
                <li><code>distribusi</code> — ketepatan, pemerataan distribusi</li>
                <li><code>anggaran</code> — dana, korupsi, efisiensi biaya</li>
                <li><code>program</code> — kebijakan, pemerintah, evaluasi program</li>
                <li><code>guru_sekolah</code> — guru, siswa, pendidikan</li>
                <li><code>umum</code> — tidak terklasifikasi ke aspek spesifik</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return html


# ERROR HANDLERS

@app.errorhandler(404)
def not_found(e):
    
    return jsonify({
        "status" : "error",
        "message": f"Endpoint tidak ditemukan: {request.path}"
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        "status" : "error",
        "message": f"Method {request.method} tidak diizinkan untuk {request.path}"
    }), 405


@app.errorhandler(500)
def internal_error(e):
    API_STATS["errors"] += 1
    return jsonify({
        "status" : "error",
        "message": "Internal server error. Cek log untuk detail."
    }), 500


# ENTRY POINT

if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  ABSA MBG API Server")
    logger.info("  URL: http://%s:%d", FLASK_HOST, FLASK_PORT)
    logger.info("  Docs: http://localhost:%d/docs", FLASK_PORT)
    logger.info("=" * 55)

    # Peringatan jika model belum tersedia
    if not COMPONENTS:
        logger.warning(
            "⚠️  Model belum dimuat! Jalankan 'python train.py' dulu."
        )

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
    )
