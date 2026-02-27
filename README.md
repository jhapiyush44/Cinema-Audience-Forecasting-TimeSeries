# 🎬 Cinema Audience Forecasting — Time Series Machine Learning Project

## 📌 Project Overview

This project predicts **daily cinema audience demand** using historical theatre visits, booking data, and engineered time-series features.

The goal is to build an accurate forecasting system that helps cinemas:

- predict daily audience load
- optimize staffing and scheduling
- improve operational planning

The project combines:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Baseline ML Models
- Autoregressive XGBoost Forecasting

---

## 📊 Problem Statement

Cinema attendance fluctuates due to:

- weekly seasonality
- booking behaviour
- time trends
- theatre-specific patterns

The task is to forecast future audience counts using historical booking and visit data while avoiding time leakage.

This is treated as a **supervised time-series regression problem**.

---

## 🧹 Data Understanding & EDA

### Key datasets used

- Theatre visit records (audience counts)
- Booking data (BookNow)
- POS data

### Major EDA Findings

- Strong weekly seasonality observed (weekend spikes).
- Audience distribution is highly skewed.
- Theatre-level behaviour contributes heavily to variance.
- Monthly trends indicate temporal demand shifts.

### Visual Analysis Included

- audience distribution
- average audience by weekday
- monthly trends
- time-series patterns

---

## 🧠 Feature Engineering

Features were designed to capture temporal and behavioural patterns:

### Time-based features
- day of week
- month
- calendar features

### Lag-based autoregressive features
- lag_1
- lag_2
- lag_3
- lag_7
- lag_14
- lag_28

These help model historical dependence in audience behaviour.

### Static Features
- theatre encoding
- categorical transformations

---

## 🤖 Modeling Approach

### Baseline Models

1️⃣ Linear Regression  
2️⃣ Random Forest  
3️⃣ XGBoost (non-autoregressive)

Purpose:

- benchmark simple vs nonlinear models
- measure contribution of feature complexity

### Baseline Insights

- Linear models underfit nonlinear demand patterns.
- Tree-based models handled variability better.
- XGBoost provided strongest baseline performance.

---

## 🚀 Final Production Model — Autoregressive XGBoost

To improve temporal prediction:

- Manual lag features were added.
- Time-aware train/validation split used.
- Autoregressive prediction performed sequentially.

Key improvements:

- better capture of seasonality
- reduced forecasting error
- stronger generalization on future data.

---

## 📈 Model Evaluation

Evaluation included:

- RMSE
- R² score
- Baseline vs AR model comparison

### Key Observations

- Lag features significantly improved performance.
- Weekly historical patterns dominated feature importance.
- Autoregressive approach outperformed non-AR baseline.

---

## 💡 Business Insights & Practical Impact

This forecasting system can help cinemas:

- estimate crowd demand in advance
- optimize staff allocation
- plan promotions around low-demand periods
- reduce operational inefficiencies.

---

## ⚠️ Challenges & Learnings

- Preventing data leakage in time series splitting.
- Handling sequential prediction dependencies.
- Balancing model complexity vs interpretability.

Key learning:

> Feature engineering and validation strategy matter more than model selection.

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

## 📁 Repository Structure

```
Cinema-Audience-Forecasting-TimeSeries/
│
├── Cinema_Audience_Forecasting_Timeseries.ipynb
├── requirements.txt
└── README.md
```


---

## 🧾 Results Summary

- End-to-end forecasting pipeline built
- Multi-model benchmarking performed
- Autoregressive XGBoost achieved best results
- Ranked top ~31% in Kaggle competition

---

## 🔮 Future Improvements

- rolling window validation
- probabilistic forecasting
- advanced time-series models (LightGBM / Temporal models)
- feature importance interpretability (SHAP)

---

## ⭐ Notes

This project demonstrates applied machine learning, time-series forecasting, and feature engineering skills on a real-world Kaggle competition dataset.  
The dataset is not included to comply with Kaggle rules.
