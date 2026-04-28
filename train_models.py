"""
train_models.py  –  v2
-----------------------
Generates synthetic dataset, trains:
  • Logistic Regression  (churn + value)
  • Random Forest        (churn + value)
  • XGBoost              (churn + value)
  • K-Means clustering
Saves all models + SHAP explainer + metrics JSON.
"""

import os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import xgboost as xgb
import shap

np.random.seed(42)
N = 1000

# ── 1. Synthetic dataset ───────────────────────────────────────────────────────
ages           = np.random.randint(18, 70, N)
genders        = np.random.choice(['Male', 'Female'], N)
purchase_freq  = np.random.randint(1, 50, N)
total_spending = np.random.uniform(50, 5000, N)
days_since     = np.random.randint(1, 365, N)

churn_score = (days_since / 365) - (purchase_freq / 50) - (total_spending / 5000)
churn       = (churn_score + np.random.normal(0, 0.2, N) > 0).astype(int)

def value_label(s, f):
    if s > 3000 and f > 30: return 'High'
    if s > 1500 or  f > 15: return 'Medium'
    return 'Low'

customer_value = [value_label(s, f) for s, f in zip(total_spending, purchase_freq)]

df = pd.DataFrame({
    'age': ages, 'gender': genders,
    'purchase_freq': purchase_freq,
    'total_spending': total_spending.round(2),
    'days_since_last': days_since,
    'churn': churn,
    'customer_value': customer_value,
})

os.makedirs('data', exist_ok=True)
df.to_csv('data/customers.csv', index=False)
print(f"[✓] Dataset saved → data/customers.csv  ({N} rows)")

# ── 2. Encoding ────────────────────────────────────────────────────────────────
le_gender = LabelEncoder()
le_value  = LabelEncoder()
df['gender_enc'] = le_gender.fit_transform(df['gender'])

FEATURES   = ['age', 'gender_enc', 'purchase_freq', 'total_spending', 'days_since_last']
FEAT_NAMES = ['Age', 'Gender', 'Purchase Freq', 'Total Spending', 'Days Since Last']
X = df[FEATURES].values

# ── 3. Churn models ────────────────────────────────────────────────────────────
y_churn = df['churn'].values
Xtr, Xte, ytr, yte = train_test_split(X, y_churn, test_size=0.2, random_state=42)

scaler_churn = StandardScaler()
Xtr_s = scaler_churn.fit_transform(Xtr)
Xte_s = scaler_churn.transform(Xte)

lr_churn = LogisticRegression(max_iter=500, random_state=42)
lr_churn.fit(Xtr_s, ytr)

rf_churn = RandomForestClassifier(n_estimators=100, random_state=42)
rf_churn.fit(Xtr, ytr)

xgb_churn = xgb.XGBClassifier(n_estimators=100, random_state=42,
                                eval_metric='logloss', verbosity=0)
xgb_churn.fit(Xtr, ytr)

def get_metrics(model, X_test, y_test, X_all, y_all, scaled=False):
    yp = model.predict(X_test)
    # CV on scaled data if needed
    cv = cross_val_score(model, X_all, y_all, cv=5, scoring='accuracy')
    return {
        'accuracy':  round(float(accuracy_score(y_test, yp)),  4),
        'precision': round(float(precision_score(y_test, yp, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_test, yp, zero_division=0)),    4),
        'f1':        round(float(f1_score(y_test, yp, zero_division=0)),        4),
        'cv_mean':   round(float(cv.mean()), 4),
        'cv_std':    round(float(cv.std()),  4),
        'confusion': confusion_matrix(y_test, yp).tolist(),
    }

scaler_churn_all = StandardScaler()
X_all_s = scaler_churn_all.fit_transform(X)

churn_metrics = {
    'lr':  get_metrics(lr_churn,  Xte_s, yte, X_all_s,  y_churn),
    'rf':  get_metrics(rf_churn,  Xte,   yte, X,         y_churn),
    'xgb': get_metrics(xgb_churn, Xte,   yte, X,         y_churn),
}
print(f"[✓] LR  churn accuracy : {churn_metrics['lr']['accuracy']}")
print(f"[✓] RF  churn accuracy : {churn_metrics['rf']['accuracy']}")
print(f"[✓] XGB churn accuracy : {churn_metrics['xgb']['accuracy']}")

# ── 4. Customer-value models ───────────────────────────────────────────────────
y_value = le_value.fit_transform(df['customer_value'])
Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y_value, test_size=0.2, random_state=42)

scaler_value = StandardScaler()
Xtr2_s = scaler_value.fit_transform(Xtr2)
Xte2_s = scaler_value.transform(Xte2)

lr_value = LogisticRegression(max_iter=500, random_state=42)
lr_value.fit(Xtr2_s, ytr2)

rf_value = RandomForestClassifier(n_estimators=100, random_state=42)
rf_value.fit(Xtr2, ytr2)

xgb_value = xgb.XGBClassifier(n_estimators=100, random_state=42,
                                eval_metric='mlogloss', verbosity=0)
xgb_value.fit(Xtr2, ytr2)

# ── 5. K-Means ─────────────────────────────────────────────────────────────────
scaler_kmeans = StandardScaler()
X_km = scaler_kmeans.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_km)
print(f"[✓] K-Means inertia: {kmeans.inertia_:.2f}")

# ── 6. Feature importance ──────────────────────────────────────────────────────
rf_imp = dict(zip(FEAT_NAMES, [round(float(v), 4) for v in rf_churn.feature_importances_]))

# ── 7. SHAP ────────────────────────────────────────────────────────────────────
print("[SHAP] Building TreeExplainer…")
shap_explainer = shap.TreeExplainer(rf_churn)
print("[SHAP] Done")

# ── 8. Save ────────────────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
artifacts = {
    'lr_churn.pkl':       lr_churn,
    'rf_churn.pkl':       rf_churn,
    'xgb_churn.pkl':      xgb_churn,
    'lr_value.pkl':       lr_value,
    'rf_value.pkl':       rf_value,
    'xgb_value.pkl':      xgb_value,
    'scaler_churn.pkl':   scaler_churn,
    'scaler_value.pkl':   scaler_value,
    'scaler_kmeans.pkl':  scaler_kmeans,
    'kmeans.pkl':         kmeans,
    'le_gender.pkl':      le_gender,
    'le_value.pkl':       le_value,
    'shap_explainer.pkl': shap_explainer,
}
for fname, obj in artifacts.items():
    joblib.dump(obj, f'models/{fname}')

meta = {
    'churn_metrics':      churn_metrics,
    'feature_importance': rf_imp,
    'feature_names':      FEAT_NAMES,
}
with open('models/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\n[✓] All artifacts saved to models/")
print("    LR | RF | XGBoost | KMeans | SHAP | meta.json")
