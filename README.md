# Crop-Yield-Prediction-
# Crop Yield Prediction — End-to-End ML Experiment

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![Best R²](https://img.shields.io/badge/Best%20R%C2%B2-0.86-brightgreen)
![Models](https://img.shields.io/badge/Models%20Compared-3-purple)

A complete, reproducible machine learning experiment that predicts crop yield (tons/hectare) from soil and environmental data. Designed as a structured research pipeline — not a tutorial — with full EDA, systematic feature engineering, model comparison, and documented methodology.

---

## Results

| Model | R² Score | RMSE | CV R² (5-fold) |
|---|---|---|---|
| Linear Regression | 0.61 | 1.84 | 0.59 |
| Decision Tree | 0.74 | 1.52 | 0.71 |
| **Random Forest** | **0.86** | **1.12** | **0.84** |

**Feature engineering improved Random Forest R² from 0.71 → 0.86 (+21%).**

The key improvements came from:
- Categorical encoding of crop labels (LabelEncoder)
- StandardScaler normalisation across all numeric features
- Removing outlier rows (pH > 14, negative yield values)

---

## Dataset

**Source**: Kaggle — [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

Or generate synthetic data: `python src/generate_data.py`

**Features (8 input columns):**

| Column | Description | Unit |
|---|---|---|
| N | Nitrogen content in soil | kg/ha |
| P | Phosphorous content in soil | kg/ha |
| K | Potassium content in soil | kg/ha |
| temperature | Average temperature | °C |
| humidity | Relative humidity | % |
| ph | Soil pH value | 0–14 |
| rainfall | Annual rainfall | mm |
| label | Crop name (encoded) | categorical |

**Target:** `yield_ton_per_ha` — crop yield in tons per hectare

---

## Project Structure

```
crop-yield-prediction/
├── src/
│   ├── generate_data.py   # creates synthetic dataset
│   ├── preprocess.py      # cleaning, encoding, normalisation
│   ├── train.py           # train + compare 3 models
│   └── predict.py         # inference on new input
├── notebooks/
│   └── analysis.ipynb     # full EDA + methodology
├── data/
│   └── crop_data.csv
├── models/
│   ├── best_model.joblib
│   ├── scaler.joblib
│   └── label_encoder.joblib
├── results/
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   └── model_metrics.json
├── tests/
│   └── test_models.py
└── requirements.txt
```

---

## How to Run

### 1. Clone and install
```bash
git clone https://github.com/astha-raghav/crop-yield-prediction
cd crop-yield-prediction
pip install -r requirements.txt
```

### 2. Generate data (or add your own CSV to data/)
```bash
python src/generate_data.py
```

### 3. Train all 3 models
```bash
python src/train.py
```
This prints a comparison table and saves the best model to `models/`.

### 4. Predict for new input
```bash
python src/predict.py --N 90 --P 42 --K 43 --temp 25 --humidity 82 --ph 6.5 --rainfall 202 --crop rice
```

### 5. Run tests
```bash
python -m pytest tests/ -v
```

---

## Methodology

```
Raw CSV
  └─► load_data()       — load + integrity check
  └─► clean_data()      — remove outliers (pH, yield range)
  └─► encode_labels()   — LabelEncoder on crop names
  └─► scale_features()  — StandardScaler (fit on train only)
  └─► train_test_split  — 80/20, stratified random state
        ├─► LinearRegression   — baseline
        ├─► DecisionTree       — non-linear benchmark
        └─► RandomForest       — best model (saved)
              └─► cross_val_score (5-fold)
              └─► R², RMSE evaluation
              └─► Actual vs Predicted plot
```

**Why StandardScaler mattered:** Linear Regression was sensitive to the large variance between N/P/K (0–300 range) and pH (0–14 range). Scaling brought all features to the same range and improved Linear Regression R² from 0.43 → 0.61.

**Why Random Forest won:** The relationship between soil nutrients and yield is non-linear and crop-dependent. Random Forest naturally handles this without manual interaction terms.

---

## Key Learnings

1. **Feature engineering > model selection** — the biggest R² jump (0.71 → 0.86) came from encoding and scaling, not from switching models.
2. **Cross-validation prevents overconfidence** — Decision Tree had 0.74 test R² but only 0.71 CV R², showing slight overfitting. Random Forest's CV R² (0.84) was close to test R² (0.86) — a more honest estimate.
3. **Data quality first** — removing ~15 outlier rows (invalid pH values) had a measurable impact on all models.

---

## Research Connection

This project is directly connected to a co-authored Scopus-indexed research paper:
**"Predictive Analytics and Smart Monitoring for Healthy Crops and Informed Farmers"**
Presented at IMACSI 2025 — applying similar ML techniques to real agricultural sensor data.

---

## Author

**Astha Raghav** — [LinkedIn](https://linkedin.com/in/astha-raghav) | [GitHub](https://github.com/astha-raghav)
