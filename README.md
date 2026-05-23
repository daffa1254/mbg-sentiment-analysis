<<<<<<< HEAD
# 🎯 ABSA MBG — Aspect-Based Sentiment Analysis

## Komentar YouTube Program Makan Bergizi Gratis (MBG)

**Algoritma:** Attention-Enhanced XGBoost + MLflow Tracking + Flask Deployment

---

## 📁 Struktur Proyek

```
mbg_project/
│
├── config.py                  ← Semua konfigurasi terpusat (path, param, dll.)
├── train.py                   ← Pipeline training utama + MLflow logging
├── requirements.txt           ← Dependensi project
├── komentarMBG_YT.csv         ← Dataset (letakkan di sini)
│
├── src/
│   ├── preprocessor.py        ← Preprocessing teks (cleaning, slang, stemming)
│   ├── labeler.py             ← Pelabelan sentimen & aspek (rule-based)
│   └── feature_engineering.py ← TF-IDF + Attention + Handcrafted Features
│
├── deployment/
│   └── app.py                 ← Flask REST API untuk deployment
│
├── output/                    ← Grafik & CSV hasil analisis (dibuat otomatis)
├── models/                    ← Model tersimpan (dibuat otomatis)
├── mlruns/                    ← MLflow experiment tracking (dibuat otomatis)
└── logs/                      ← Log training & API (dibuat otomatis)
```

---

## 🚀 Cara Menjalankan di VSCode

### 1. Persiapan

```bash
# Clone / salin semua file ke satu folder
# Letakkan komentarMBG_YT.csv di folder mbg_project/

# Buat virtual environment (direkomendasikan)
python -m venv venv

# Aktifkan virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

> ⚠️ **Windows:** Ganti `gunicorn` dengan `waitress` di requirements.txt

### 3. Training Model

```bash
# Training lengkap dengan MLflow tracking
python train.py

# Training tanpa MLflow (lebih cepat)
python train.py --no-mlflow

# Training dengan subset data (untuk testing cepat)
python train.py --sample 3000
```

**Output training:**

- `output/1_eda.png` → Grafik EDA
- `output/2_label_distribution.png` → Distribusi sentimen & aspek
- `output/3_evaluasi.png` → Confusion matrix & CV accuracy
- `output/4_absa_per_aspek.png` → ABSA heatmap
- `output/5_wordcloud.png` → WordCloud per sentimen
- `output/hasil_sentimen_absa.csv` → Dataset dengan prediksi
- `models/xgboost_model.json` → Model XGBoost
- `models/feature_builder.pkl` → TF-IDF vectorizer + encoder
- `models/label_encoder.pkl` → Label encoder
- `models/model_metadata.json` → Metadata & metrik

### 4. Lihat MLflow UI

```bash
# Buka MLflow tracking UI
mlflow ui --backend-store-uri ./mlruns

# Buka browser:

# (atau port lain jika 5000 sudah dipakai)
```

Di MLflow UI Anda bisa melihat:

- Semua parameter XGBoost yang digunakan
- Metrik: accuracy, F1, precision, recall per run
- Grafik perbandingan antar eksperimen
- Artefak: grafik, model, CSV

### 5. Jalankan API Deployment

```bash
# Mode development
python deployment/app.py

# Mode produksi (Linux/macOS)
gunicorn -w 4 -b 0.0.0.0:5000 "deployment.app:app"

# Mode produksi (Windows)
waitress-serve --port=5000 deployment.app:app
```

---

## 🌐 Penggunaan API

### Prediksi Satu Teks

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "makanan MBG basi dan kotor tidak layak dimakan"}'
```

**Response:**

```json
{
  "status": "success",
  "data": {
    "teks_asli": "makanan MBG basi dan kotor tidak layak dimakan",
    "teks_bersih": "makan bergizi gratis basi kotor tidak layak makan",
    "sentimen": "negatif",
    "aspek": "kualitas_makanan",
    "confidence": {
      "negatif": 0.89,
      "netral": 0.07,
      "positif": 0.04
    },
    "waktu_proses": 0.023
  }
}
```

### Prediksi Batch

```bash
curl -X POST http://localhost:5000/predict/batch \
     -H "Content-Type: application/json" \
     -d '{
       "texts": [
         "program mbg sangat bagus bermanfaat untuk anak",
         "makanan basi jorok tidak layak",
         "yah biasa saja lah"
       ]
     }'
```

### Python (requests library)

```python
import requests

# Satu teks
resp = requests.post(
    "http://localhost:5000/predict",
    json={"text": "program MBG bagus untuk gizi anak"}
)
print(resp.json())

# Batch
resp = requests.post(
    "http://localhost:5000/predict/batch",
    json={"texts": ["teks 1", "teks 2", "teks 3"]}
)
print(resp.json()["ringkasan"])
```

### Endpoint Tersedia

| Method | URL              | Deskripsi                  |
| ------ | ---------------- | -------------------------- |
| GET    | `/`              | Halaman utama & status     |
| GET    | `/health`        | Health check               |
| POST   | `/predict`       | Prediksi satu teks         |
| POST   | `/predict/batch` | Prediksi batch (maks. 100) |
| GET    | `/model/info`    | Metadata & metrik model    |
| GET    | `/docs`          | Dokumentasi HTML           |

---

## 🏗️ Arsitektur Pipeline

```
Dataset CSV
    │
    ▼
[1] Load & Validasi Data
    │
    ▼
[2] EDA — panjang komentar, top kata, distribusi like
    │
    ▼
[3] Preprocessing
    ├── Lowercase
    ├── Hapus mention / hashtag / URL / emoji
    ├── Hapus angka & tanda baca
    ├── Normalisasi huruf berulang
    ├── 200+ kamus slang/singkatan → baku KBBI
    ├── Tokenisasi
    ├── Stopword Removal (Sastrawi + kata sentimen dipertahankan)
    └── Stemming Sastrawi
    │
    ▼
[4] Pelabelan (Rule-based Lexicon + Deteksi Negasi)
    ├── Sentimen: positif / negatif / netral
    └── Aspek: kualitas_makanan / distribusi / anggaran /
               program / guru_sekolah / umum
    │
    ▼
[5] Feature Engineering
    ├── TF-IDF (1-gram & 2-gram, 8000 fitur)
    ├── Softmax Attention Weights per dokumen  ← kunci inovasi
    ├── Attended TF-IDF = TF-IDF × Attention
    ├── Handcrafted Features (rasio sentimen, negasi, panjang)
    └── Aspek One-Hot Encoding
    │
    ▼
[6] SMOTE (penyeimbangan kelas minoritas)
    │
    ▼
[7] Training Attention-Enhanced XGBoost
    ├── 400 estimators, depth=7, lr=0.08
    └── Sample weighting otomatis
    │
    ▼
[8] Evaluasi
    ├── Accuracy & F1-Score pada test set
    ├── Classification Report per kelas
    └── 5-Fold Cross Validation
    │
    ▼
[9] MLflow Logging
    ├── Parameters: semua hyperparameter XGBoost
    ├── Metrics: accuracy, f1, precision, recall, CV per fold
    ├── Artifacts: grafik PNG, model, CSV hasil
    └── Model Registry: xgboost model tersimpan
    │
    ▼
[10] Deployment Flask API
    ├── GET  /health
    ├── POST /predict
    ├── POST /predict/batch
    └── GET  /model/info
```

---
=======
---
title: MBG Sentimen
emoji: 💻
colorFrom: gray
colorTo: pink
sdk: gradio
sdk_version: 6.14.0
python_version: '3.13'
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> c534b0d9cf39c0ea880e54d636f89c41c925fdac
