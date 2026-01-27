# 🎬 Cinema Audience Forecasting (Time-Series ML)

This project predicts daily cinema audience attendance using time-series forecasting and machine learning.

Built for a Kaggle competition using:
- Feature engineering
- Booking behavior aggregation
- Autoregressive lag features
- XGBoost regression

---

## 📊 Results
- Kaggle Leaderboard Score: 0.334
- Cutoff Score: 0.30
- Rank: 826 / 2632 (Top 31%)

---

## 🧠 Techniques Used

### Baseline Models
- Linear Regression
- Random Forest
- XGBoost

### Final Model
Autoregressive XGBoost with:
- Lag features (1,2,3,7,14,28)
- Time features (month, weekday, weekend)
- Booking aggregations
- POS + Online ticket sales

Validation:
RMSE: 24.58  
R²: 0.45

---

## 📁 Dataset Structure
Place competition CSVs inside:

data/
├── booknow_visits.csv
├── booknow_booking.csv
├── cinePOS_booking.csv
├── date_info.csv
├── theaters.csv
└── sample_submission.csv

---

## ▶️ How to Run

pip install -r requirements.txt
jupyter notebook

Run the notebook.

---

## 🛠 Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn

---

## ✨ Key Learning
- Time-series feature engineering
- Lag-based modeling
- Handling multi-source booking data
- Model comparison & validation
