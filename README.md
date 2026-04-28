# 🧠 CustPredict — AI Customer Performance Prediction System

A production-style Flask web application that predicts customer churn and value
using Logistic Regression, Random Forest, and K-Means clustering.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Authentication** | Login / Logout with hashed passwords |
| **Dashboard** | KPI cards + 4 Chart.js charts |
| **Churn Prediction** | Logistic Regression & Random Forest (with probability) |
| **Value Classification** | High / Medium / Low via LR & RF |
| **Customer Segmentation** | K-Means 3-cluster (Gold / Silver / Bronze) |
| **Database** | SQLite via Flask-SQLAlchemy |
| **Saved Models** | `.pkl` files via joblib |

---

## 📁 Project Structure

```
customer_predict/
├── app.py                  # Flask application (routes, models, predictions)
├── train_models.py         # Generate dataset + train & save ML models
├── init_db.py              # Create DB tables + seed sample data
├── requirements.txt        # Python dependencies
│
├── models/                 # Saved .pkl files (after running train_models.py)
│   ├── lr_churn.pkl
│   ├── rf_churn.pkl
│   ├── lr_value.pkl
│   ├── rf_value.pkl
│   ├── scaler_churn.pkl
│   ├── scaler_value.pkl
│   ├── scaler_kmeans.pkl
│   ├── kmeans.pkl
│   ├── le_gender.pkl
│   └── le_value.pkl
│
├── data/
│   └── customers.csv       # Synthetic training dataset (1000 rows)
│
├── instance/
│   └── customers.db        # SQLite database (auto-created)
│
├── static/
│   ├── css/style.css       # Dark editorial theme
│   └── js/main.js          # UI interactions + counter animations
│
└── templates/
    ├── base.html           # Sidebar layout template
    ├── login.html          # Authentication page
    ├── dashboard.html      # KPI + charts overview
    ├── predict.html        # Customer input form + results
    ├── customers.html      # Paginated customer list
    └── customer_detail.html # Single customer profile
```

---

## 🚀 Quick Start

### 1. Clone / unzip the project
```bash
cd customer_predict
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train ML models (generates data/ and models/ folders)
```bash
python train_models.py
```
Expected output: 93% accuracy on churn, 99% on customer value.

### 5. Initialize the database and seed sample data
```bash
python init_db.py
```
This creates `instance/customers.db` and 20 sample customers.

### 6. Run the app
```bash
python app.py
```

### 7. Open in browser
```
http://127.0.0.1:5000
```

**Default credentials:** `admin` / `admin123`

---

## 🤖 Machine Learning Details

### Input Features
| Feature | Description |
|---|---|
| Age | Customer age (18–100) |
| Gender | Encoded: Male=1, Female=0 |
| Purchase Frequency | Purchases per year |
| Total Spending | Lifetime spend in USD |
| Days Since Last Purchase | Computed from Last Purchase Date |

### Models
| Task | Models | Accuracy |
|---|---|---|
| Churn Prediction | Logistic Regression, Random Forest | ~93%, ~91% |
| Customer Value | Logistic Regression, Random Forest | ~97%, ~99% |
| Segmentation | K-Means (k=3) | — (unsupervised) |

### Cluster Segments
- **Gold** — High frequency, high spend, recent
- **Silver** — Medium engagement
- **Bronze** — Low frequency, low spend, or lapsed

---

## 🔑 Default Login

| Username | Password |
|---|---|
| admin | admin123 |

> ⚠️ Change the `SECRET_KEY` in `app.py` before deploying to production.
