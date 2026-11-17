# Sentix: Sentiment Analysis & Emotion Detection Platform

> **College Major Project**  
> **Backend:** Python 3.14 + Flask  
> **Frontend:** HTML5, CSS3, JavaScript (ES6+), Chart.js  
> **Machine Learning:** Scikit-learn (Multinomial Naive Bayes, Logistic Regression, Linear SVM)  
> **Database:** SQLite3 with Foreign Key Enforcement  
> **Styling Direction:** Modern SaaS Analytics Dashboard (Purple `#7C3AED` Accent, Responsive Sidebar Layout)

---

## 1. Project Overview

Sentix is a complete, production-grade Sentiment Analysis and Emotion Detection web application. It processes real-time text inputs and bulk CSV review datasets, classifies them into **Positive**, **Negative**, or **Neutral** polarities, tracks historical analyses in SQLite, visualizes real-time analytics with Chart.js, evaluates supervised machine learning models on held-out benchmarks, detects 7 distinct emotional states using GoEmotions, and exports downloadable PDF and CSV analytical reports.

### Key Capabilities
- **Three-Class Sentiment Classification**: Classifies statements into *Positive*, *Negative*, and *Neutral* using trained scikit-learn models.
- **Model Comparison**: Trains and evaluates 3 distinct supervised learning algorithms (**Multinomial Naive Bayes**, **Logistic Regression**, and **Linear SVM**) on the exact same 12,284 held-out test samples.
- **Calibrated Probability Scores**: Uses `CalibratedClassifierCV` to ensure all algorithms output legitimate class probabilities ($P_{\text{pos}} + P_{\text{neg}} + P_{\text{neu}} \approx 1.0$) rather than uncalibrated margins.
- **Dedicated Emotion Detection**: Implements a separate 7-class emotion classifier (*Happiness, Sadness, Anger, Fear, Surprise, Love, Neutral*) trained on Google Research's GoEmotions benchmark.
- **Bulk CSV Dataset Analyzer**: Analyzes thousands of customer reviews in a single batch with auto-detected text headers, progress reporting, and downloadable enriched CSVs.
- **Auditable SQLite History**: Persists every single and batch prediction with foreign key integrity, keyword search, filtering, and deletion.
- **Automated Report Generation**: Generates executive PDF reports via ReportLab with summary tables and downloads structured CSV archives.

---

## 2. Benchmark Datasets & Results

All models are trained and evaluated on authentic research datasets with **zero data leakage** (TF-IDF vectorizers are fitted strictly on training data).

### Datasets
1. **Primary Sentiment Dataset**: [Cardiff NLP TweetEval (Sentiment Configuration)](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
   - **Training Set**: 45,615 samples
   - **Validation Set**: 2,000 samples
   - **Held-Out Test Set**: 12,284 samples
   - **Class Balance**: Neutral: 45.88%, Positive: 35.13%, Negative: 18.99%
2. **Emotion Dataset**: [Google Research GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)
   - **38,928 samples** mapped to 7 distinct affective categories.

### Model Evaluation Results (12,284 Held-Out Test Samples)

| Algorithm | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Training Duration |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 57.76% | 59.63% | 54.98% | **0.5398** | 5.89s |
| **Logistic Regression** | **58.92%** | 58.27% | **60.26%** | **0.5868** | 11.60s |
| **Linear SVM (Calibrated)** | 58.15% | 58.47% | 56.56% | **0.5567** | 14.84s |
| **Emotion Model (7 Classes)**| 59.00% | 56.40% | 57.20% | **0.5671** | 13.08s |

*All metrics and confusion matrices are persisted in `models/metrics.json` and synced to SQLite `model_metrics`.*

---

## 3. Project Directory Structure

```text
Sentiment_Analysis/
├── app.py                      # Flask Application Factory & Route Registration
├── config.py                   # Environment & Application Settings
├── requirements.txt            # Python Dependencies
├── pytest.ini                  # Pytest Configuration
├── README.md                   # Project Documentation & Run Guide
│
├── dataset/
│   ├── download_datasets.py    # Automated Dataset Acquisition Script
│   ├── sample_test.csv         # Ready-to-use Sample CSV for Bulk Testing
│   └── (sentiment/emotion data ignored in git)
│
├── models/
│   ├── sentiment_nb.joblib     # Fitted Multinomial Naive Bayes Pipeline
│   ├── sentiment_lr.joblib     # Fitted Logistic Regression Pipeline
│   ├── sentiment_svm.joblib    # Fitted Calibrated Linear SVM Pipeline
│   ├── emotion_model.joblib    # Fitted GoEmotions Classifier Pipeline
│   ├── metrics.json            # Exact Test-Set Evaluation Scores
│   ├── emotion_metrics.json    # Emotion Classifier Benchmark Scores
│   └── comparison_chart.png    # Multi-Metric Comparison Bar Chart
│
├── ml/
│   ├── preprocessing.py        # Text Validation, Cleaning & Leak-Free TF-IDF
│   ├── train_model.py          # Model Training & Benchmark Pipeline
│   ├── predict.py              # In-Memory Model Manager & Inference Service
│   └── evaluate.py             # Classification Metrics & Matplotlib Visualizer
│
├── database/
│   ├── schema.sql              # SQLite DDL Schema with Foreign Keys & Indexes
│   ├── db.py                   # Parameterized SQLite Data Access Layer
│   └── sentiment.db            # Live SQLite Database
│
├── routes/
│   ├── dashboard.py            # Overview & Chart.js Endpoints
│   ├── analyzer.py             # Single-Text Prediction Endpoints
│   ├── bulk.py                 # CSV Upload & Batch Inference Endpoints
│   ├── history.py              # Paginated History & Deletion Endpoints
│   ├── performance.py          # Model Benchmarks & Confusion Matrices
│   ├── emotion.py              # 7-Class Emotion Classification Endpoints
│   └── reports.py              # PDF Generation & CSV Export Endpoints
│
├── templates/
│   ├── base.html               # Master Layout with Responsive Purple Sidebar
│   ├── dashboard.html          # Interactive Live Analytics Dashboard
│   ├── analyzer.html           # Real-Time Single Text Analyzer
│   ├── bulk.html               # Bulk CSV Dataset Processor
│   ├── history.html            # Searchable & Filterable SQLite History
│   ├── performance.html        # Model Performance & Confusion Matrix View
│   ├── emotion.html            # Emotion Classification Interface
│   └── reports.html            # Report Generator & Export Portal
│
├── static/
│   ├── css/style.css           # Purple Modern Analytics Theme (#7C3AED)
│   ├── js/main.js              # Sidebar & UI Helpers
│   ├── js/analyzer.js          # Interactive Text Analyzer Client
│   ├── js/charts.js            # Chart.js Donut & Trend Visualizations
│   └── images/                 # Exported Comparison Visuals
│
└── tests/
    ├── test_preprocessing.py   # Unit Tests for Input Cleaning & Validation
    ├── test_prediction.py      # Unit Tests for Sentiment & Emotion Inference
    ├── test_database.py        # Unit Tests for SQLite Operations & Queries
    └── test_routes.py          # End-to-End Tests for All Web & API Endpoints
```

---

## 4. Setup & Installation Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/aayanpunera/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Datasets & Train Models (Optional if pre-trained artifacts exist)
To re-download the official datasets and retrain all models from scratch:
```bash
python dataset/download_datasets.py
python -m ml.train_model
```

### 5. Initialize the Database
```bash
python -c "from database.db import init_db; init_db()"
```

### 6. Run the Application
```bash
python app.py
```
Open your web browser and navigate to:
👉 **`http://127.0.0.1:5000/`**

---

## 5. Deploying to Vercel

The application is fully configured for zero-configuration serverless deployment on [Vercel](https://vercel.com/):

### Ready-Made Vercel Files Included:
- **`vercel.json`**: Configures route rewrites to the Python serverless function and bundles model assets (`models/`, `templates/`, `static/`, `database/`).
- **`api/index.py`**: The serverless entrypoint exporting the WSGI Flask application.
- **`.vercelignore`**: Excludes heavy raw datasets, tests, and local virtual environments to ensure rapid deployment builds under the size quota.
- **Serverless-Adaptive Storage**: Automatically detects Vercel's read-only container environment (`VERCEL=1`) and routes SQLite transactions and Matplotlib cache to the writable `/tmp` directory.

### Deployment Steps:
1. Push the latest code to your GitHub repository:
   ```bash
   git push origin main
   ```
2. Go to your [Vercel Dashboard](https://vercel.com/new).
3. Select **Import Git Repository** and choose `Sentiment-Analysis`.
4. (Optional) In **Project Settings** > **Environment Variables**, you can add:
   - `SECRET_KEY`: (Optional) Your production session secret key.
   - `VERCEL_SUPPORT_LARGE_FUNCTIONS`: `1` (Recommended by Vercel for data science and ML runtimes).
5. Click **Deploy**. Vercel will install dependencies, package the model pipelines, and provide you with a live `.vercel.app` URL!

---

## 6. Running Automated Tests

Sentix includes a comprehensive pytest suite covering preprocessing, model predictions, database transactions, and end-to-end web routes:

```bash
pytest tests/ -v
```

Expected Output:
```text
============================= test session starts =============================
tests/test_database.py .......                                           [ 16%]
tests/test_prediction.py ............                                    [ 45%]
tests/test_preprocessing.py ..........                                   [ 69%]
tests/test_routes.py .............                                       [100%]

============================== 42 passed in 6.2s ==============================
```

---

## 6. API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/predict` | `POST` | Predict sentiment for single text (`{"text": "...", "model_name": "..."}`) |
| `/api/bulk-analyze`| `POST` | Upload CSV and process batch of reviews |
| `/api/history` | `GET` | Retrieve paginated history (`?page=1&per_page=20&sentiment=...&q=...`) |
| `/api/history/<id>`| `DELETE` | Delete single history record |
| `/api/emotion` | `POST` | Detect emotion across 7 categories (`{"text": "..."}`) |
| `/api/metrics` | `GET` | Retrieve test-set evaluation scores and confusion matrices |
| `/api/stats` | `GET` | Live dashboard counts and 7-day trend aggregations |
| `/reports/csv` | `GET` | Export history or batch as CSV (`?batch_id=...`) |
| `/reports/pdf` | `GET` | Generate and download executive ReportLab PDF report |

---

## 7. License & Credits

- Developed for College Major Project Submission.
- Datasets: [Cardiff NLP TweetEval](https://github.com/cardiffnlp/tweeteval) & [Google Research GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions).
