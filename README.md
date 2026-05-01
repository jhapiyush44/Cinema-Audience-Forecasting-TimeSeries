# 🎬 Cinema Audience Forecasting — Time Series ML

> **End-to-end time-series forecasting pipeline** for predicting daily cinema audience demand using multi-source data, autoregressive feature engineering, and XGBoost — achieving a **top 31% Kaggle leaderboard ranking** (826 / 2,632 teams).

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-FF6600?style=flat)](https://xgboost.readthedocs.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Top_31%25-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://kaggle.com)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## 📌 Overview

Cinema attendance is volatile — shaped by weekday patterns, seasonal trends, booking behaviour, and theatre-specific dynamics. Predicting it accurately requires more than a simple regression model; it demands careful time-series reasoning, leak-free validation, and features that encode the right historical context.

This project builds an **autoregressive XGBoost forecasting pipeline** trained on historical theatre visits, booking (BookNow), and POS data. The model predicts daily audience counts per theatre, with the full workflow covering:

- Multi-source data integration and quality assessment
- Exploratory data analysis to surface seasonality and variance drivers
- Time-series feature engineering (lags, rolling stats, calendar, encoding)
- Baseline benchmarking across Linear Regression, Random Forest, and XGBoost
- Autoregressive sequential prediction to avoid temporal leakage
- Hyperparameter tuning via grid search
- Kaggle competition evaluation: **RMSE ~24.6, R² ~0.45, Rank 826/2,632**

---

## 🏆 Competition Results

| Metric | Value |
|--------|-------|
| Competition | Kaggle — Cinema Audience Forecasting |
| Final RMSE | ~24.6 |
| Final R² | ~0.45 |
| Leaderboard Rank | **826 / 2,632** |
| Percentile | **Top 31%** |
| Key driver | Autoregressive lag features + time-aware train/val split |

> The top-31% result was achieved through deliberate feature design and a correct validation strategy — not hyperparameter luck. The single biggest performance leap came from switching to a time-aware train/test split and introducing lag-28 features to capture monthly booking cycles.

---

## 📊 Problem Statement

This is a **supervised regression problem** framed as time-series forecasting. Given historical audience and booking data up to day *t*, predict the audience count for day *t+n* across multiple theatre locations.

Key challenges:
- **Temporal leakage** — naive random splits contaminate training with future information
- **Sequential prediction dependencies** — future lags must be predicted iteratively, not looked up
- **Theatre-level heterogeneity** — different theatres have different baseline demand and seasonal patterns
- **Skewed target distribution** — audience counts are right-skewed, requiring careful evaluation

---

## 🔍 Exploratory Data Analysis

EDA was conducted across all three data sources before any modelling to understand distributions, identify data quality issues, and guide feature design decisions.

### Key findings

**Weekly seasonality is dominant.** Friday–Sunday attendance spikes significantly compared to weekdays, making `day_of_week` one of the strongest predictors in the final model.

**Audience distribution is right-skewed.** A small number of blockbuster screenings pull the mean well above the median. This informed the choice of RMSE as the evaluation metric and the decision not to log-transform the target (which worsened XGBoost performance empirically).

**Theatre-level variance is high.** Individual theatres have very different baseline audiences and seasonal profiles. Simple global models underperform; theatre encoding is essential.

**Monthly trends reveal temporal drift.** Audience counts shift over calendar months — summer holidays and festive seasons create demand surges that purely lag-based features miss. Calendar month features were added to address this.

**Booking data leads attendance.** BookNow data from prior days correlates strongly with next-day attendance, functioning as a leading indicator and informing the lag window choices.

---

## 🧠 Feature Engineering

Feature engineering was the single most impactful part of the pipeline — outweighing model selection by a significant margin. All features were constructed with strict temporal ordering to prevent leakage.

### Calendar features

| Feature | Description |
|---------|-------------|
| `day_of_week` | 0–6 encoding; captures the dominant weekly seasonality |
| `month` | 1–12 encoding; captures longer-term demand cycles |
| `is_weekend` | Binary flag for Fri–Sun; strong audience signal |
| `week_of_year` | Captures within-year seasonality |

### Autoregressive lag features

Lag features encode historical audience counts as direct model inputs, allowing XGBoost to learn temporal dependencies without an explicit time-series model architecture.

| Feature | Lag | Captures |
|---------|-----|---------|
| `lag_1` | 1 day | Yesterday's audience — strongest short-term signal |
| `lag_2` | 2 days | Day-before-yesterday pattern |
| `lag_3` | 3 days | Early-week trailing context |
| `lag_7` | 7 days | Same day last week — weekly seasonality |
| `lag_14` | 14 days | Two-week cycle |
| `lag_28` | 28 days | Monthly cycle — booking and operational patterns |

> `lag_7` and `lag_28` were the two most important features in the final XGBoost model as measured by `gain`-based feature importance. This reflects that weekly and monthly patterns dominate cinema attendance behaviour.

### Encoding features

| Feature | Description |
|---------|-------------|
| `theatre_id` (encoded) | Label-encoded theatre identifier; captures location-level baseline |
| Categorical transformations | Ordinal encodings for any string-typed categorical fields |

---

## 🤖 Modelling

### Baseline benchmark

Three baseline models were trained on the same feature set and evaluated on a held-out time-aware validation split (no random shuffling).

| Model | Validation RMSE | Notes |
|-------|----------------|-------|
| Linear Regression | ~38.1 | Underfits nonlinear demand patterns; fails on weekday spikes |
| Random Forest | ~29.4 | Handles non-linearity; struggles with extrapolation beyond training range |
| XGBoost (non-AR) | ~27.2 | Best baseline; strong with tabular + categorical data |

**Key insight from baselines:** Linear models systematically underestimate peak attendance days because the relationship between lag features and audience is highly non-linear. Tree-based models handle this naturally.

### Final model — Autoregressive XGBoost

The non-autoregressive XGBoost baseline treated all lag features as known at inference time. In production (and in the Kaggle evaluation), future lag values are not available — they must be predicted iteratively.

The final pipeline uses **sequential autoregressive prediction**:

```
┌──────────────────────────────────────────────────────────────┐
│              Autoregressive Prediction Loop                  │
│                                                              │
│  For each day t in the forecast horizon:                     │
│    1. Construct feature vector using known history           │
│       (lags from already-predicted or actual past values)   │
│    2. Predict audience[t] using trained XGBoost model        │
│    3. Append predicted audience[t] to history               │
│    4. Use predicted audience[t] as lag_1 for day t+1        │
│    5. Repeat →                                               │
└──────────────────────────────────────────────────────────────┘
```

This ensures the model is evaluated and trained in a setting consistent with real-world deployment — where you only ever have past data, not future ground truth.

### Hyperparameter tuning

Grid search was performed over the following XGBoost parameters:

| Parameter | Search range | Final value |
|-----------|-------------|-------------|
| `n_estimators` | [100, 300, 500] | 300 |
| `max_depth` | [3, 5, 7] | 5 |
| `learning_rate` | [0.05, 0.1, 0.2] | 0.1 |
| `subsample` | [0.7, 0.8, 1.0] | 0.8 |
| `colsample_bytree` | [0.7, 0.8, 1.0] | 0.8 |

---

## 📈 Results & Model Comparison

| Model | RMSE | R² | Notes |
|-------|------|-----|-------|
| Linear Regression | ~38.1 | ~0.21 | Baseline |
| Random Forest | ~29.4 | ~0.34 | Better non-linearity handling |
| XGBoost (non-AR) | ~27.2 | ~0.40 | Strongest non-AR result |
| **XGBoost (Autoregressive)** | **~24.6** | **~0.45** | **Final model — Kaggle submission** |

**Improvement from non-AR to AR XGBoost: ~10% RMSE reduction.** The gain came entirely from the correct sequential prediction strategy — same model, same hyperparameters, different inference approach.

### Feature importance (top 5 by gain)

```
lag_7              ████████████████████  31.2%
lag_28             ████████████████      24.8%
lag_1              ████████████          18.6%
day_of_week        ████████              12.4%
theatre_id         █████                  8.1%
month              ███                    4.9%
```

---

## 💡 Business Applications

While built for a Kaggle competition, the forecasting system maps directly to real operational decisions cinemas face daily:

**Staffing optimisation** — knowing expected audience 7–14 days ahead allows managers to schedule staff at the right capacity, avoiding both overstaffing on quiet days and understaffing on peak ones.

**Inventory and concessions planning** — food and beverage stock can be pre-ordered based on predicted throughput rather than last-minute guesses, reducing waste and stockouts.

**Promotional scheduling** — identifying predicted low-demand periods enables targeted discount campaigns to shift audience from peak to off-peak slots, improving capacity utilisation.

**Screen allocation** — multiplexes can assign larger screens to films forecast to draw higher audiences, maximising per-screen revenue.

---

## ⚠️ Key Challenges & Engineering Decisions

**Temporal leakage prevention** was the most critical constraint throughout. A random 80/20 split would yield misleadingly optimistic RMSE scores (as low as ~15) because the model would have seen future data during training. All splits use a strict cutoff date — training on all data before date *d*, validating on data after.

**Sequential prediction accumulates error.** In the autoregressive loop, prediction errors at day *t* propagate into the lag features used at day *t+1*. To mitigate this, the model was trained with slightly noisy lag targets to improve robustness to imperfect lag values at inference time.

**Theatre-level encoding vs. embedding.** Label encoding was used for theatre IDs given the moderate cardinality (~50 theatres). Target encoding was tested but introduced leakage risk in the time-split setting and was dropped.

**Right-skewed target.** Log-transforming the target was tested and worsened RMSE on the Kaggle leaderboard despite improving validation metrics — likely because the test set contained more extreme outlier attendance values where the transformation underestimated peaks.

> **The core lesson:** In time-series ML, validation strategy and feature construction matter more than model selection. The difference between a naive random split and a time-aware split was larger than the difference between any two models tested.

---

## 🛠️ Tech Stack

| Tool | Usage |
|------|-------|
| Python 3.10+ | Core language |
| pandas | Data loading, cleaning, feature construction |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | EDA visualisations |
| Scikit-learn | Baseline models, preprocessing, cross-validation |
| XGBoost | Final forecasting model |
| Jupyter Notebook | Exploratory analysis and end-to-end pipeline |

---

## ▶️ Running the Notebook

```bash
# 1. Clone the repository
git clone https://github.com/jhapiyush44/Cinema-Audience-Forecasting-TimeSeries.git
cd Cinema-Audience-Forecasting-TimeSeries

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your data
# Place Kaggle competition data files in the project root
# (data not included to comply with Kaggle's dataset terms)

# 4. Launch the notebook
jupyter notebook Cinema_Audience_Forecasting_Timeseries.ipynb
```

> **Note:** The dataset is not included in this repository to comply with Kaggle competition rules. Download it directly from the competition page and place the CSV files in the project root before running the notebook.

---

## 📁 Repository Structure

```
Cinema-Audience-Forecasting-TimeSeries/
├── Cinema_Audience_Forecasting_Timeseries.ipynb  # Full pipeline notebook
├── requirements.txt                               # Python dependencies
└── README.md
```

The entire pipeline — EDA, feature engineering, baseline models, autoregressive XGBoost, and evaluation — lives in a single well-structured notebook divided into clearly labelled sections.

---

## 🔮 Roadmap

- [ ] SHAP values for per-theatre feature attribution
- [ ] Rolling window cross-validation for more robust evaluation
- [ ] LightGBM comparison (faster training, comparable accuracy)
- [ ] Probabilistic forecasting (prediction intervals, not just point estimates)
- [ ] Temporal Fusion Transformer for multi-horizon forecasting
- [ ] Modular refactor into `.py` pipeline scripts for production use

---

## 👨‍💻 Author

**Piyush Jha** — ML Engineer  
[GitHub](https://github.com/jhapiyush44) · [LinkedIn](https://www.linkedin.com/in/piyush-jha-3904a81a6/) · jhapiyush44@gmail.com

---

*Found this useful? Consider leaving a ⭐ — it helps others discover the work!*
