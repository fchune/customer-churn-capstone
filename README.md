# Customer Churn Prediction System

A practical end-to-end machine learning capstone project that predicts whether a customer is likely to leave (churn), enabling businesses to take proactive retention actions.

---

## Business Problem

Customer churn — when customers stop using a product or service — directly impacts revenue. This project builds a **classification model** that scores each customer's churn probability based on their account and usage data. A deployed Flask web application allows non-technical staff to enter customer details and instantly see a churn-risk prediction.

---

## Project Structure

```
customer-churn-capstone/
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks for EDA, modeling, evaluation
├── models/                # Saved model files (.pkl)
├── flask_app/
│   ├── app.py             # Flask application
│   ├── templates/         # HTML templates
│   └── static/            # CSS and images
├── reports/               # Final report (PDF/Word)
├── images/                # Screenshots and charts
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Workflow

| Step | Description |
|------|-------------|
| 1 | Choose dataset (Telco Customer Churn — Kaggle) |
| 2 | Define business problem |
| 3 | Clean and explore data (EDA) |
| 4 | Feature engineering |
| 5 | Train and compare models |
| 6 | Select best model and save as `.pkl` |
| 7 | Build Flask web app |
| 8 | Write final report |
| 9 | Upload to GitHub |
| 10 | Post on LinkedIn |

---

## Models Used

- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting (XGBoost)

---

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas, NumPy | Data manipulation |
| Matplotlib, Seaborn | Visualization |
| Scikit-learn | Modeling |
| XGBoost | Gradient boosting |
| Flask | Web application |
| Joblib | Model serialization |
| HTML/CSS | Front-end |
| GitHub | Version control |

---

## How to Run the Flask App

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (run notebook first, or skip if model already exists)
# notebooks/churn_analysis.ipynb

# 3. Start the Flask app
cd flask_app
python app.py

# 4. Open browser
# http://127.0.0.1:5000
```

---

## Dataset

**Telco Customer Churn** — IBM Sample Dataset  
Available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and place it in the `data/` folder.

---

## Results

> *(To be completed after modeling — update with best model metrics, confusion matrix screenshot, and ROC curve)*

---

## Author

**Fatou Chune**  
MSc Data Science — University of Pittsburgh  
[GitHub](https://github.com/chune-pitt) | [LinkedIn](#)
