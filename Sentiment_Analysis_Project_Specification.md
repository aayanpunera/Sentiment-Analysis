# Sentiment Analysis Using Machine Learning

## Major Project Specification and Development Guide

**Project type:** College Major Project\
**Backend:** Python + Flask\
**Frontend:** HTML, CSS, JavaScript\
**Machine Learning:** Scikit-learn\
**Database:** SQLite3\
**Design direction:** Modern analytics dashboard with a purple accent,
responsive layout, and sidebar navigation.

------------------------------------------------------------------------

## 1. Project Overview

Build a complete web-based Sentiment Analysis application using Python
and Machine Learning. The system accepts text from users or CSV files,
predicts sentiment, stores analysis history, visualizes results,
evaluates ML models, detects emotions, and generates downloadable
reports.

### Primary objectives

-   Classify text into **Positive**, **Negative**, or **Neutral**
    sentiment.
-   Train and compare three Machine Learning algorithms.
-   Provide a functional web frontend connected to a Flask backend.
-   Support single-text and bulk CSV analysis.
-   Store analysis history and batch results in SQLite.
-   Display model evaluation metrics and visualizations.
-   Provide a separate emotion-detection capability.
-   Export results and summaries as CSV and PDF.
-   Include validation, logging, exception handling, and tests.

### Important implementation principles

-   Write original, understandable code with meaningful comments.
-   Do not hardcode predictions, dashboard statistics, or model metrics.
-   Train and evaluate models using a proper train/test split.
-   Prevent data leakage by fitting text vectorizers only on training
    data.
-   Use real model outputs in the UI.
-   Document dataset sources, label definitions, model limitations, and
    any preprocessing decisions.
-   Build incrementally and test each module before moving on.

------------------------------------------------------------------------

## 2. Functional Modules

### 2.1 Dashboard

Display live statistics calculated from the database:

-   Total texts analyzed.
-   Positive, negative, and neutral counts.
-   Sentiment distribution chart.
-   Sentiment trend over time, where sufficient timestamped data exists.
-   Recent analysis activity.
-   Quick links to Text Analyzer and Bulk Analyzer.

Provide useful empty states when no data is available.

### 2.2 Text Analyzer

Allow a user to:

1.  Enter a sentence, paragraph, or review.
2.  Select one of the available sentiment models.
3.  Submit the text for analysis.
4.  View the predicted sentiment.
5.  View class scores or calibrated probabilities when supported.
6.  View the selected model name.
7.  Save the result in analysis history.

Validate empty input and configured maximum text length. Clearly explain
that model scores are estimates, not guarantees of correctness.

### 2.3 Bulk CSV Analyzer

Allow a user to:

1.  Upload a CSV file.
2.  Select the column containing text/reviews.
3.  Select a sentiment model.
4.  Validate the file and data.
5.  Analyze valid rows in batches.
6.  View a results table with original text and predicted sentiment.
7.  View batch summary counts.
8.  Download the enriched results as CSV.
9.  Store the batch and its results in SQLite.

Handle invalid file types, empty files, missing columns, empty cells,
malformed CSV files, and oversized uploads. Configure safe file-size and
row-count limits. Do not execute uploaded content.

### 2.4 Analysis History

Store and display:

-   Original input text.
-   Predicted sentiment.
-   Model used.
-   Class scores, if available.
-   Analysis type (single or bulk).
-   Timestamp.
-   Batch reference when applicable.

Provide search, filtering, sorting, pagination, and deletion with
confirmation.

### 2.5 Model Performance

Show actual test-set evaluation results for each trained algorithm:

-   Accuracy.
-   Macro precision.
-   Macro recall.
-   Macro F1-score.
-   Confusion matrix.
-   Per-class precision, recall, and F1.
-   Training and testing sample counts.
-   Model comparison charts.

Persist evaluation metrics with a training timestamp. Do not invent or
hardcode scores.

### 2.6 Emotion Detection

Implement emotion detection as a separate capability from sentiment
classification.

Suggested user-facing categories:

-   Happiness
-   Sadness
-   Anger
-   Fear
-   Surprise
-   Love
-   Neutral

Use a separate emotion dataset and model. Document the mapping from
original dataset labels to the broader categories. Emotion and sentiment
are related but are not interchangeable; do not simply map positive
sentiment to happiness and negative sentiment to sadness.

### 2.7 Reports

Provide:

-   CSV export of prediction results.
-   PDF report with analysis summary.
-   Sentiment distribution summary.
-   Model performance summary.
-   Batch report, where applicable.

Reports must be generated from actual stored or computed data.

------------------------------------------------------------------------

## 3. Dataset Recommendations

### Primary sentiment dataset: TweetEval Sentiment

Recommended because it provides three sentiment classes: negative,
neutral, and positive.

-   Dataset: TweetEval, sentiment configuration.
-   Source: https://huggingface.co/datasets/cardiffnlp/tweet_eval
-   Use the official train, validation, and test splits where available.
-   Inspect class balance and label definitions before training.
-   Record the dataset version and source in the project documentation.

### Alternative datasets

**IMDb Movie Reviews** - 50,000 movie reviews in the standard
benchmark. - Binary positive/negative labels. - Source:
https://ai.stanford.edu/\~amaas/data/sentiment/ - Useful for
experiments, but it does not directly satisfy the three-class
requirement.

**Sentiment140** - Large Twitter dataset with positive/negative
labels. - Source:
https://www.kaggle.com/datasets/kazanova/sentiment140 - Contains noisy
social-media text and is binary in its common form.

**Yelp Review Polarity** - Customer/business reviews with
positive/negative labels. - Source:
https://huggingface.co/datasets/fancyzhx/yelp_polarity - Useful for
review-oriented experiments, but binary in its standard version.

Do not combine datasets without checking differences in label
definitions, text domains, licensing, and class balance.

### Emotion dataset: GoEmotions

-   Approximately 58,000 Reddit comments.
-   27 emotion categories plus Neutral.
-   Source:
    https://huggingface.co/datasets/google-research-datasets/go_emotions

For a simpler interface, map the original labels to the selected broad
categories, document the mapping, and preserve Neutral. Emotion
detection may be implemented as a separate model or a carefully
documented pretrained classifier. The project must identify which
approach is used.

------------------------------------------------------------------------

## 4. Machine Learning Design

### Sentiment algorithms

Train and compare:

1.  **Multinomial Naive Bayes** --- baseline text classifier.
2.  **Logistic Regression** --- interpretable comparison model.
3.  **Linear Support Vector Machine (Linear SVM)** --- sparse text
    classification model.

### Shared text-processing pipeline

1.  Load and inspect the dataset.
2.  Validate required text and label columns.
3.  Handle missing and duplicate records appropriately.
4.  Normalize text conservatively.
5.  Split data using stratification, or use official dataset splits.
6.  Fit TF-IDF on training data only.
7.  Transform validation/test data using the fitted vectorizer.
8.  Train all three algorithms using comparable data and features.
9.  Evaluate using the same held-out test set.
10. Save trained pipelines and evaluation artifacts.

### Initial configuration

  Setting                  Suggested value
  ------------------------ -------------------------------------------------------
  Train/test split         80/20 if official splits are not used
  Stratification           Yes
  Random state             42
  TF-IDF n-grams           (1, 2)
  Maximum features         Start at 20,000
  Main comparison metric   Macro F1
  Additional metrics       Accuracy, macro precision, macro recall
  Validation               Use a validation split or cross-validation for tuning

Use a scikit-learn `Pipeline` where practical so vectorization is fitted
only on training data and the saved artifact is consistent at inference
time.

**SVM score handling:** A standard Linear SVM does not directly provide
probabilities. Either display decision scores with clear labeling or
calibrate it using `CalibratedClassifierCV` fitted only on training
data. Do not label raw decision scores as probabilities.

### Model artifacts

Save complete fitted pipelines where possible, rather than saving a
model and vectorizer in ways that can become mismatched. Use `joblib`
and load only trusted artifacts created by this project. Document model
version and training dataset metadata.

------------------------------------------------------------------------

## 5. System Architecture

Use a simple three-layer architecture.

### Presentation layer

-   HTML templates.
-   CSS.
-   JavaScript.
-   Chart.js for visualizations.
-   Responsive dashboard and navigation.

### Application layer

-   Flask application.
-   Routes and API endpoints.
-   Request validation.
-   Preprocessing and prediction services.
-   CSV processing.
-   Report generation.
-   Logging and exception handling.

### Data and ML layer

-   Scikit-learn trained pipelines.
-   Separate emotion model.
-   SQLite database.
-   Saved evaluation metrics.

### Request flow

1.  User interacts with the frontend.
2.  Frontend sends a request to Flask.
3.  Flask validates the request.
4.  The prediction service preprocesses input consistently with
    training.
5.  The selected model generates a prediction.
6.  The application stores the result in SQLite.
7.  Flask returns structured data.
8.  The frontend renders the result and charts.

------------------------------------------------------------------------

## 6. Suggested Flask Routes

  Route                 Method   Purpose
  --------------------- -------- -------------------------
  `/`                   GET      Dashboard
  `/analyze`            GET      Text analyzer page
  `/api/predict`        POST     Single-text prediction
  `/bulk`               GET      Bulk upload page
  `/api/bulk-analyze`   POST     CSV processing
  `/history`            GET      History page
  `/api/history`        GET      Retrieve history
  `/api/history/<id>`   DELETE   Delete a history record
  `/performance`        GET      Model performance page
  `/emotion`            GET      Emotion detection page
  `/api/emotion`        POST     Emotion prediction
  `/reports/csv`        GET      CSV export
  `/reports/pdf`        GET      PDF export

Use appropriate HTTP status codes and consistent JSON error responses
for API endpoints.

------------------------------------------------------------------------

## 7. Database Design

Use SQLite3. Create tables with primary keys, foreign keys, indexes
where useful, and parameterized SQL queries.

### 7.1 `analysis_history`

  Column           Type                  Description
  ---------------- --------------------- ------------------------------------------
  id               INTEGER PRIMARY KEY   Record ID
  input_text       TEXT NOT NULL         Original text
  sentiment        TEXT NOT NULL         Predicted sentiment
  model_name       TEXT NOT NULL         Model used
  positive_score   REAL                  Positive probability/score, if available
  negative_score   REAL                  Negative probability/score, if available
  neutral_score    REAL                  Neutral probability/score, if available
  emotion          TEXT                  Optional emotion prediction
  analysis_type    TEXT                  single or bulk
  batch_id         INTEGER               Optional batch reference
  created_at       DATETIME              Timestamp

Scores must be clearly defined. If they are not calibrated
probabilities, store and label them as scores rather than probabilities.

### 7.2 `batch_jobs`

  Column              Type                  Description
  ------------------- --------------------- -------------------------------
  id                  INTEGER PRIMARY KEY   Batch ID
  filename            TEXT                  Original filename for display
  total_records       INTEGER               Total rows
  processed_records   INTEGER               Successfully processed rows
  model_name          TEXT                  Selected model
  status              TEXT                  processing/completed/failed
  created_at          DATETIME              Timestamp

### 7.3 `batch_results`

  Column           Type                  Description
  ---------------- --------------------- ---------------------------
  id               INTEGER PRIMARY KEY   Result ID
  batch_id         INTEGER NOT NULL      Foreign key to batch_jobs
  input_text       TEXT NOT NULL         Original review
  sentiment        TEXT NOT NULL         Predicted label
  positive_score   REAL                  Optional score
  negative_score   REAL                  Optional score
  neutral_score    REAL                  Optional score

### 7.4 `model_metrics`

  Column             Type                  Description
  ------------------ --------------------- --------------------
  id                 INTEGER PRIMARY KEY   Metric record ID
  model_name         TEXT NOT NULL         Algorithm name
  accuracy           REAL                  Accuracy
  precision          REAL                  Macro precision
  recall             REAL                  Macro recall
  f1_score           REAL                  Macro F1
  training_samples   INTEGER               Training rows
  testing_samples    INTEGER               Test rows
  trained_at         DATETIME              Training timestamp

For emotion analysis, either add clearly separated emotion fields or
create a dedicated `emotion_history` table. Avoid mixing sentiment
labels and emotion labels in the same field.

Enable SQLite foreign keys on every connection using
`PRAGMA foreign_keys = ON`. Use transactions and close connections
reliably.

------------------------------------------------------------------------

## 8. Project Folder Structure

Create and maintain this structure. It is acceptable to add files as
implementation progresses.

``` text
Sentiment_Analysis/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── sentiment_dataset.csv
│   └── emotion_dataset.csv
│
├── models/
│   ├── sentiment_nb.joblib
│   ├── sentiment_lr.joblib
│   ├── sentiment_svm.joblib
│   └── emotion_model.joblib
│
├── ml/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   ├── evaluate.py
│   └── emotion_detection.py
│
├── database/
│   ├── db.py
│   ├── schema.sql
│   └── sentiment.db
│
├── routes/
│   ├── dashboard.py
│   ├── analyzer.py
│   ├── bulk.py
│   ├── history.py
│   ├── performance.py
│   ├── emotion.py
│   └── reports.py
│
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── analyzer.html
│   ├── bulk.html
│   ├── history.html
│   ├── performance.html
│   ├── emotion.html
│   └── reports.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── main.js
│   │   ├── analyzer.js
│   │   └── charts.js
│   └── images/
│
├── exports/
└── tests/
    ├── test_preprocessing.py
    ├── test_prediction.py
    ├── test_database.py
    └── test_routes.py
```

Do not commit large datasets, generated exports, secrets, or unnecessary
runtime files to source control. Add an appropriate `.gitignore`.

------------------------------------------------------------------------

## 9. Dependencies: `requirements.txt`

Start with these dependencies and verify compatible versions in the
actual development environment.

``` text
Flask>=3.0,<4.0
scikit-learn>=1.5,<2.0
pandas>=2.2,<4.0
numpy>=1.26,<3.0
joblib>=1.4,<2.0
matplotlib>=3.9,<4.0
reportlab>=4.2,<5.0
python-dotenv>=1.0,<2.0
pytest>=8.0,<10.0
```

SQLite3 is included with standard Python installations and does not need
a separate pip package.

Frontend libraries: - Chart.js for charts. - Bootstrap 5 optionally for
responsive layout. - Font Awesome for icons. - Inter or Poppins
typography.

These may be loaded through CDN links or stored locally. If using CDN
links, document the internet requirement.

After testing, record the exact working dependency versions (for
example, with `pip freeze`) so the project can be reproduced reliably.

------------------------------------------------------------------------

## 10. UI/UX Design Specification

Create a polished, modern analytics dashboard inspired by professional
SaaS products.

### Visual direction

  Element            Specification
  ------------------ -------------------------
  Primary accent     Purple `#7C3AED`
  Secondary accent   Violet `#A78BFA`
  Page background    `#F8F7FC`
  Cards              White
  Positive           Green
  Negative           Red
  Neutral            Grey or blue
  Typography         Inter or Poppins
  Card radius        12--16px
  Layout             Sidebar + main content
  Responsiveness     Desktop, tablet, mobile

### Sidebar navigation

1.  Dashboard
2.  Text Analyzer
3.  Bulk Analyzer
4.  Emotion Detection
5.  Analysis History
6.  Model Performance
7.  Reports

### UI requirements

-   Consistent sidebar and header across pages.
-   Clear active navigation state.
-   Responsive cards, tables, and charts.
-   Loading indicator while predictions run.
-   Toast or inline success/error messages.
-   Confirmation before destructive actions.
-   Empty states for new users.
-   Accessible labels and keyboard-friendly controls.
-   No fake or hardcoded analytical data.

Use a clean light theme with purple accents. The interface should
prioritize readability and ease of demonstration.

------------------------------------------------------------------------

## 11. Error Handling and Security

Implement robust error handling throughout the project.

### Required validation

-   Empty text input.
-   Text exceeding configured length.
-   Unsupported file extension.
-   Oversized CSV.
-   Empty or malformed CSV.
-   Missing text column.
-   Empty review cells.
-   Missing or corrupted model artifacts.
-   Database connection or transaction failures.
-   Invalid API payloads.
-   Unexpected server errors.

### Engineering practices

1.  Use `try-except` around file operations, model loading, database
    operations, and report generation.
2.  Use Python's `logging` module for technical diagnostics.
3.  Show friendly error messages to users; do not expose stack traces.
4.  Use parameterized SQL queries.
5.  Validate uploaded files by size, extension, and content.
6.  Use secure upload handling and never execute uploaded files.
7.  Configure Flask secret keys through environment variables.
8.  Add CSRF protection if forms or authenticated sessions are
    introduced.
9.  Restrict upload size and CSV row count.
10. Keep datasets and generated exports out of public static directories
    unless explicitly intended.
11. Load only trusted model artifacts created by this project.
12. Test all error paths.

------------------------------------------------------------------------

## 12. Testing Plan

Create automated tests using pytest and perform manual UI testing.

  Test                     Expected result
  ------------------------ -------------------------------------
  Positive sample text     Valid sentiment prediction
  Negative sample text     Valid sentiment prediction
  Neutral sample text      Valid sentiment prediction
  Empty input              Validation message
  Numbers-only input       No application crash
  Special characters       Safe processing
  Unseen vocabulary        Prediction without crash
  Invalid CSV              Helpful error
  Missing CSV column       Helpful error
  Empty CSV                Upload rejected
  Valid CSV                Rows processed and results returned
  History save             Record stored in SQLite
  History delete           Correct record removed
  Missing model artifact   Clear initialization error
  PDF generation           Valid PDF returned
  CSV export               Valid downloadable CSV
  Mobile viewport          Responsive layout

Test the application from a fresh environment using the documented setup
instructions before submission.

------------------------------------------------------------------------

## 13. Development Roadmap

Suggested duration: 6--8 weeks, adjusted to the student's schedule.

### Phase 1 --- Environment and Dataset (Week 1)

-   Create project directory and virtual environment.
-   Install dependencies.
-   Download and inspect TweetEval Sentiment.
-   Understand dataset fields and labels.
-   Check class distribution and missing data.

### Phase 2 --- Preprocessing (Weeks 1--2)

-   Implement text validation and preprocessing.
-   Prepare labels.
-   Establish train/validation/test strategy.
-   Implement TF-IDF.
-   Add preprocessing tests.

### Phase 3 --- Model Training (Weeks 2--3)

-   Train Multinomial Naive Bayes.
-   Train Logistic Regression.
-   Train Linear SVM.
-   Evaluate all models on the same held-out test data.
-   Save model artifacts and metrics.
-   Create model comparison charts.

### Phase 4 --- Flask Backend (Weeks 3--4)

-   Configure Flask application.
-   Create routes and service modules.
-   Load trusted model artifacts.
-   Implement prediction API.
-   Add validation, logging, and exception handling.

### Phase 5 --- Frontend (Weeks 4--5)

-   Create shared page layout and sidebar.
-   Build dashboard.
-   Build Text Analyzer.
-   Display predictions and charts.
-   Connect frontend to Flask APIs.

### Phase 6 --- Database and Bulk Analysis (Weeks 5--6)

-   Implement SQLite schema and database helpers.
-   Save single-text analysis history.
-   Implement CSV upload and validation.
-   Process batches and save results.
-   Add search, filters, and downloads.

### Phase 7 --- Advanced Features (Weeks 6--7)

-   Implement separate emotion detection.
-   Build model performance dashboard.
-   Generate PDF reports.
-   Complete CSV export functionality.

### Phase 8 --- Testing and Documentation (Weeks 7--8)

-   Run automated tests.
-   Perform manual testing.
-   Fix defects.
-   Verify installation from scratch.
-   Prepare screenshots, README, project report, and presentation.

------------------------------------------------------------------------

## 14. College Submission Deliverables

Prepare the following for final evaluation:

-   Complete source code.
-   Dataset sources and dataset description.
-   Saved trained models and vectorizer/pipeline artifacts.
-   SQLite schema and sample database.
-   `requirements.txt`.
-   README with setup and execution instructions.
-   Screenshots of each major module.
-   Test cases and test results.
-   Model comparison and evaluation results.
-   Project report.
-   Project presentation.

### Suggested report chapters

1.  Introduction
2.  Problem Statement
3.  Objectives
4.  Literature Review
5.  Existing System
6.  Proposed System
7.  System Requirements
8.  System Architecture
9.  Dataset Description
10. Methodology and Algorithms
11. Implementation
12. Database Design
13. Results and Evaluation
14. Testing
15. Limitations
16. Future Enhancements
17. Conclusion
18. References

------------------------------------------------------------------------

## 15. Acceptance Criteria

The project is ready for submission when:

-   [ ] All three sentiment models train and load successfully.
-   [ ] The system predicts Positive, Negative, and Neutral.
-   [ ] Predictions work through the frontend.
-   [ ] Metrics are calculated from held-out test data.
-   [ ] Bulk CSV analysis works and produces downloadable results.
-   [ ] Analysis history is stored in SQLite.
-   [ ] Model comparison is visible in the performance dashboard.
-   [ ] Emotion detection operates as a separate capability.
-   [ ] PDF and CSV reports are generated successfully.
-   [ ] All navigation and major pages work.
-   [ ] Invalid inputs and file errors are handled gracefully.
-   [ ] Automated tests pass.
-   [ ] The project runs from a fresh environment using the README
    instructions.

------------------------------------------------------------------------

## 16. Instructions for the AI Development Agent (Antigravity)

Use this document as the authoritative project specification.

### How to work

1.  First inspect the existing workspace. Do not overwrite existing user
    files without checking them.
2.  Create a clear implementation plan and explain the proposed
    architecture.
3.  Set up the project structure and Python virtual environment.
4.  Implement the Machine Learning pipeline first.
5.  Obtain and inspect the selected dataset. Do not fabricate training
    data or evaluation results.
6.  Train and evaluate all three sentiment algorithms using a fair,
    reproducible process.
7.  Save and test model artifacts.
8.  Build the Flask backend with clean, modular code.
9.  Build the frontend using HTML, CSS, and JavaScript.
10. Integrate SQLite, bulk processing, emotion detection,
    visualizations, and reports.
11. Write tests for the important functions and routes.
12. Run the application and tests, inspect errors, and fix them.
13. Update the README with exact installation, training, and run
    commands.

### Code quality requirements

-   Use modular, readable Python code.
-   Add meaningful comments and docstrings.
-   Use functions and classes where appropriate.
-   Avoid unnecessary complexity and overengineering.
-   Keep model training separate from web request handling.
-   Do not retrain models on every user request.
-   Do not hardcode model outputs, metrics, dashboard statistics, or
    sample predictions as real results.
-   Do not silently swallow exceptions.
-   Keep secrets and configuration outside source code.
-   Use consistent naming and formatting.

### Development workflow

Work in milestones. At the end of each milestone:

-   List files created or changed.
-   Explain what works.
-   Run relevant tests.
-   Report test results honestly.
-   Identify any remaining issues.
-   Do not claim a feature is complete unless it has been implemented
    and tested.

### First milestone

Begin with **Phase 1: Environment and Dataset**, followed by **Phase 2:
Preprocessing** and **Phase 3: Model Training**.

Do not start by generating every frontend page with placeholder data.
First establish a working, reproducible Machine Learning foundation.
Then integrate the real model into Flask and build the UI around actual
outputs.

------------------------------------------------------------------------

## 17. Final Project Description

**Sentiment Analysis Using Machine Learning** is a Python-based web
application that uses Natural Language Processing and supervised Machine
Learning algorithms to classify text into positive, negative, and
neutral sentiments. It compares Multinomial Naive Bayes, Logistic
Regression, and Linear SVM, and provides a separate emotion detection
module. The application includes an interactive Flask frontend, bulk CSV
analysis, SQLite-based history, model evaluation dashboards, and
downloadable PDF/CSV reports.

**Expected outcome:** A functional, tested, well-documented college
major project demonstrating Machine Learning, NLP, Python backend
development, frontend integration, database management, and data
visualization.
