# 🎬 Cinema Audience Forecasting using Time-Series Machine Learning

Time-series forecasting project built for a Kaggle competition to predict daily cinema audience attendance using booking data, feature engineering, lag-based autoregressive modeling, and XGBoost.

This repository contains **only the code and notebook**.  
The competition dataset is **not included** due to Kaggle's data sharing restrictions.

---

## 📊 Competition Results

- Leaderboard Score: **0.334**
- Cutoff Score: **0.30**
- Rank: **826 / 2632**
- Position: **Top ~31%**

---

## 🧠 Problem Statement

Predict the number of daily cinema visitors for each theatre using:

- Online bookings (BookNow)
- POS ticket sales (CinePOS)
- Calendar information
- Historical attendance patterns

This is a **time-series forecasting** task where past audience behavior strongly influences future attendance.

---

## ⚙️ Approach

### Data Processing
- Date parsing and cleaning
- Theatre ID mapping
- Merging multi-source booking systems
- Handling missing values

### Feature Engineering
- Year, month, day, weekday, weekend flags
- Booking aggregations (same-day vs advance bookings)
- POS ticket totals
- Interaction features
- Autoregressive lag features: 1, 2, 3, 7, 14, 28 days

### Model Comparison
Baseline models:
- Linear Regression
- Random Forest
- XGBoost (baseline)

Final model:
- Autoregressive XGBoost with lag features

### Validation Performance
- RMSE: ~24.6
- R²: ~0.45

The autoregressive model significantly outperformed baseline approaches.

---

## 📁 Repository Structure

```
Cinema-Audience-Forecasting-TimeSeries/
│
├── Cinema_Audience_Forecasting_XGBoost_AR.ipynb
├── requirements.txt
├── README.md
└── submission.csv (optional)
```

---

## ⚠️ Dataset Notice (Important)

The dataset belongs to a Kaggle competition and **cannot be redistributed publicly**.

Therefore:
- No CSV files are included in this repository
- You must download the dataset directly from Kaggle

To run locally, create:

```
data/
```

and place the Kaggle competition CSV files inside it.

---

## 🚀 How to Run

### Clone repository
```
git clone https://github.com/jhapiyush44/Cinema-Audience-Forecasting-TimeSeries.git
cd Cinema-Audience-Forecasting-TimeSeries
```

### Install dependencies
```
pip install -r requirements.txt
```

### Launch notebook
```
jupyter notebook Cinema_Audience_Forecasting_XGBoost_AR.ipynb
```

---

## 📦 Requirements

Main libraries used:

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn

---

## 🛠 Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 💡 Key Learnings

- Time-series feature engineering
- Autoregressive modeling
- Handling multi-source real-world data
- Lag-based forecasting
- Model comparison and validation
- Practical Kaggle competition workflow

---

## 👤 Author

Piyush Jha  
GitHub: https://github.com/jhapiyush44

---

## ⭐ Notes

This project demonstrates applied machine learning, time-series forecasting, and feature engineering skills on a real-world Kaggle competition dataset.  
The dataset is not included to comply with Kaggle rules.
