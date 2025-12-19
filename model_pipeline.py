"""
model_pipeline.py

Refactored training pipeline that:
- Matches your current pipeline logic (preprocessing, feature engineering, model training, evaluation)
- Saves artifacts for deployment:
    - best_model.joblib
    - scaler.joblib
    - label_encoders.joblib
    - feature_list.json
    - model_performance.json
    - feature_importance_top10.csv
    - shap_summary.png
    - shap_dependence_<feat>.png (top 3)
    - lime_explanation.html

Usage:
    python model_pipeline.py
"""

import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

# Helper to install packages if missing
def install_and_import(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing missing package: {package_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        __import__(import_name)

# Core packages (ensure installed)
packages = [
    ('pandas', None),
    ('numpy', None),
    ('matplotlib', None),
    ('seaborn', None),
    ('scikit-learn', 'sklearn'),
    ('joblib', None),
    ('shap', None),
    ('lime', None)
]
for pkg, imp in packages:
    install_and_import(pkg, imp)

# Imports
import os
import json
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)

from lime.lime_tabular import LimeTabularExplainer

# ---------------------------
# Configuration / filenames
# ---------------------------
DATA_PATH = "ODL 900.xlsx"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.joblib")
ENCODERS_PATH = os.path.join(ARTIFACT_DIR, "label_encoders.joblib")
FEATURE_LIST_PATH = os.path.join(ARTIFACT_DIR, "feature_list.json")
MODEL_PERF_PATH = os.path.join(ARTIFACT_DIR, "model_performance.json")
FI_CSV_PATH = os.path.join(ARTIFACT_DIR, "feature_importance_top10.csv")
SHAP_SUMMARY_PNG = os.path.join(ARTIFACT_DIR, "shap_summary.png")
LIME_HTML = os.path.join(ARTIFACT_DIR, "lime_explanation.html")

# ---------------------------
# Load data
# ---------------------------
print("[STEP 1] Loading dataset...")
try:
    df = pd.read_excel(DATA_PATH)
except Exception:
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print("Error loading dataset:", e)
        sys.exit(1)

print("Dataset shape:", df.shape)

# ---------------------------
# Identify target column (same heuristics as pipeline)
# ---------------------------
print("[STEP 2] Identifying target variable...")
target_candidates = [
    'Are you presently involved in e-learning',
    'presently involved in e-learning',
    'e-learning participation',
    'elearning'
]
target_col = None
for candidate in target_candidates:
    matches = [c for c in df.columns if candidate.lower() in c.lower()]
    if matches:
        target_col = matches[0]
        break

if target_col is None:
    print("ERROR: target column not found. Available columns:")
    print(df.columns.tolist())
    sys.exit(1)

print("Target variable:", target_col)

# Drop missing target
df = df.dropna(subset=[target_col])
print("Samples after removing missing target:", df.shape[0])

# Encode target
if df[target_col].dtype == object:
    df[target_col] = df[target_col].astype(str).str.strip().str.lower().map({'yes':1, 'no':0})

y = df[target_col].copy()
X = df.drop(columns=[target_col])
# drop timestamp/time columns
X = X.drop(columns=[c for c in X.columns if 'timestamp' in c.lower() or 'time' in c.lower()], errors='ignore')

# ---------------------------
# Preprocessing functions (fit/apply)
# ---------------------------
def fit_preprocessing(X_df):
    """
    Fit label encoders for object columns and scaler on numeric features.
    Returns fitted encoders dict, scaler, and feature list.
    """
    X_local = X_df.copy()
    # Fill missing values (mode for objects, median otherwise)
    for col in X_local.columns:
        if X_local[col].isnull().sum() > 0:
            if X_local[col].dtype == object:
                mode_val = X_local[col].mode()[0] if len(X_local[col].mode())>0 else "Unknown"
                X_local[col].fillna(mode_val, inplace=True)
            else:
                X_local[col].fillna(X_local[col].median(), inplace=True)

    # Label encode object columns
    label_encoders = {}
    cat_cols = X_local.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X_local[col] = le.fit_transform(X_local[col].astype(str))
        label_encoders[col] = le

    # Feature engineering: construct-based composite scores (keyword-based)
    likert_cols = X_local.columns.tolist()
    # PU
    pu_keywords = ['useful', 'facilitate', 'improve', 'enhance', 'benefit']
    pu_cols = [c for c in likert_cols if any(k in c.lower() for k in pu_keywords)]
    if pu_cols:
        X_local['perceived_usefulness_score'] = X_local[pu_cols].mean(axis=1)
    # PEOU
    peou_keywords = ['easy', 'simple', 'convenient', 'effort']
    peou_cols = [c for c in likert_cols if any(k in c.lower() for k in peou_keywords)]
    if peou_cols:
        X_local['perceived_ease_score'] = X_local[peou_cols].mean(axis=1)
    # BI
    bi_keywords = ['willing', 'intend', 'want', 'plan']
    bi_cols = [c for c in likert_cols if any(k in c.lower() for k in bi_keywords)]
    if bi_cols:
        X_local['willingness_score'] = X_local[bi_cols].mean(axis=1)

    # Ensure numeric and fill remaining missing by median
    X_local = X_local.apply(pd.to_numeric, errors='coerce')
    X_local = X_local.fillna(X_local.median())

    # Fit scaler on the resulting numeric frame
    scaler = StandardScaler()
    scaler.fit(X_local)

    feature_list = X_local.columns.tolist()
    return label_encoders, scaler, feature_list

def apply_preprocessing(X_df, label_encoders, scaler, feature_list):
    """
    Apply preprocessing to new dataframe using fitted encoders/scaler and produce numeric matrix matching feature_list order.
    """
    X_local = X_df.copy()

    # drop timestamp/time cols
    X_local = X_local.drop(columns=[c for c in X_local.columns if 'timestamp' in c.lower() or 'time' in c.lower()], errors='ignore')

    # Fill missing values similar to fit: mode for objects, median otherwise
    for col in X_local.columns:
        if X_local[col].isnull().sum() > 0:
            if X_local[col].dtype == object:
                mode_val = X_local[col].mode()[0] if len(X_local[col].mode())>0 else "Unknown"
                X_local[col].fillna(mode_val, inplace=True)
            else:
                X_local[col].fillna(X_local[col].median(), inplace=True)

    # Apply label encoders where applicable
    for col, le in label_encoders.items():
        if col in X_local.columns:
            # unseen labels -> map to '__unknown__' then transform; LabelEncoder cannot handle unseen; so we map unseen to mode index
            X_local[col] = X_local[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
            X_local[col] = le.transform(X_local[col].astype(str))
    # For any remaining object columns that were not encoded (unexpected), label-encode on the fly
    remaining_obj = X_local.select_dtypes(include=['object']).columns.tolist()
    for col in remaining_obj:
        le_temp = LabelEncoder()
        X_local[col] = le_temp.fit_transform(X_local[col].astype(str))

    # Feature engineering (same logic)
    likert_cols = X_local.columns.tolist()
    pu_keywords = ['useful', 'facilitate', 'improve', 'enhance', 'benefit']
    pu_cols = [c for c in likert_cols if any(k in c.lower() for k in pu_keywords)]
    if pu_cols and 'perceived_usefulness_score' not in X_local.columns:
        X_local['perceived_usefulness_score'] = X_local[pu_cols].mean(axis=1)
    peou_keywords = ['easy', 'simple', 'convenient', 'effort']
    peou_cols = [c for c in likert_cols if any(k in c.lower() for k in peou_keywords)]
    if peou_cols and 'perceived_ease_score' not in X_local.columns:
        X_local['perceived_ease_score'] = X_local[peou_cols].mean(axis=1)
    bi_keywords = ['willing', 'intend', 'want', 'plan']
    bi_cols = [c for c in likert_cols if any(k in c.lower() for k in bi_keywords)]
    if bi_cols and 'willingness_score' not in X_local.columns:
        X_local['willingness_score'] = X_local[bi_cols].mean(axis=1)

    # Ensure numeric and fill remaining missing by median
    X_local = X_local.apply(pd.to_numeric, errors='coerce')
    X_local = X_local.fillna(X_local.median())

    # Align features to the training feature list (add missing columns with 0)
    for feat in feature_list:
        if feat not in X_local.columns:
            X_local[feat] = 0.0
    # Reorder
    X_local = X_local[feature_list]
    # Scale
    X_scaled = scaler.transform(X_local)
    return X_local, X_scaled

# ---------------------------
# Fit preprocessing on training X
# ---------------------------
print("[STEP 3] Fitting preprocessing...")
label_encoders, scaler, feature_list = fit_preprocessing(X)

# Save encoders, scaler, feature list
joblib.dump(label_encoders, ENCODERS_PATH)
joblib.dump(scaler, SCALER_PATH)
with open(FEATURE_LIST_PATH, "w") as f:
    json.dump(feature_list, f)

print("Saved preprocessing artifacts.")

# ---------------------------
# Prepare train/test splits using the preprocessed feature matrix
# ---------------------------
print("[STEP 4] Preparing train-test sets...")
# Use apply_preprocessing to build numeric matrices
X_all_df, X_all_scaled = apply_preprocessing(X, label_encoders, scaler, feature_list)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_all_scaled, y, test_size=0.3, random_state=42, stratify=y)

print("Train samples:", X_train_scaled.shape[0], "Test samples:", X_test_scaled.shape[0])

# ---------------------------
# Train models (same set as pipeline)
# ---------------------------
print("[STEP 5] Training models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
fitted_models = {}
for name, model in models.items():
    print(f"Training {name} ...")
    model.fit(X_train_scaled, y_train)
    fitted_models[name] = model
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:,1] if hasattr(model, "predict_proba") else None

    res = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    res["cv_mean"] = float(cv_scores.mean())
    res["cv_std"] = float(cv_scores.std())
    results[name] = res

    print("  Accuracy:", res["accuracy"], "F1:", res["f1_score"])

# ---------------------------
# Model comparison & select best by F1
# ---------------------------
print("[STEP 6] Comparing models...")
results_df = pd.DataFrame(results).T
best_model_name = results_df["f1_score"].astype(float).idxmax()
best_model = fitted_models[best_model_name]
print("Best model selected:", best_model_name)

# Save best model & metadata
joblib.dump(best_model, BEST_MODEL_PATH)
with open(MODEL_PERF_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("Saved best model and model performance metadata.")

# ---------------------------
# Feature importance if available
# ---------------------------
if hasattr(best_model, "feature_importances_"):
    fi = pd.DataFrame({"feature": feature_list, "importance": best_model.feature_importances_})
    fi = fi.sort_values("importance", ascending=False)
    fi.head(10).to_csv(FI_CSV_PATH, index=False)
    # Plot top 10
    plt.figure(figsize=(8,6))
    top10 = fi.head(10).iloc[::-1]
    plt.barh(top10["feature"], top10["importance"])
    plt.xlabel("Importance")
    plt.title(f"Top 10 Feature Importance - {best_model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "feature_importance.png"), dpi=300)
    plt.close()
    print("Saved feature importance plot and csv.")
else:
    fi = None
    print("Best model does not expose feature_importances_.")

# ---------------------------
# Confusion matrix, ROC, PR curves
# ---------------------------
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Adopter","Adopter"], yticklabels=["Non-Adopter","Adopter"])
plt.title("Confusion Matrix - " + best_model_name)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

if hasattr(best_model, "predict_proba"):
    y_proba = best_model.predict_proba(X_test_scaled)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "roc_curve.png"), dpi=300)
    plt.close()

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    plt.figure(figsize=(6,5))
    plt.plot(recall_vals, precision_vals, label=f"AUC={pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "pr_curve.png"), dpi=300)
    plt.close()

# ---------------------------
# SHAP Explainability (global summary + top3 dependence)
# ---------------------------
print("[STEP 7] SHAP explainability...")
try:
    if hasattr(best_model, "predict_proba") and hasattr(best_model, "feature_importances_"):
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_all_df)
    elif hasattr(best_model, "predict_proba"):
        explainer = shap.KernelExplainer(best_model.predict_proba, shap.sample(X_all_df, 50))
        shap_values = explainer.shap_values(X_all_df, nsamples=100)
    else:
        raise RuntimeError("Selected model not supported by SHAP in this script.")

    plt.figure()
    shap.summary_plot(shap_values, X_all_df, show=False)
    plt.savefig(SHAP_SUMMARY_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved SHAP summary plot:", SHAP_SUMMARY_PNG)

    # top 3 features for dependence plots
    if fi is not None:
        top3 = fi.head(3)["feature"].tolist()
    else:
        top3 = feature_list[:3]
    for feat in top3:
        safe = re.sub(r"[^\w\d-]", "_", feat)
        shap.dependence_plot(feat, shap_values, X_all_df, show=False)
        plt.savefig(os.path.join(ARTIFACT_DIR, f"shap_dependence_{safe}.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved SHAP dependence:", feat)
except Exception as e:
    print("SHAP failed:", e)

# ---------------------------
# LIME explanation for one representative instance (Option 1)
# ---------------------------
print("[STEP 8] Generating LIME explanation for a representative instance...")
try:
    # Use the first test instance (X_test_scaled[0])
    inst_idx = 0
    # Need the unscaled row in dataframe form aligned with feature_list
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_list)
    # But we want the unscaled original values for LIME's training_data param:
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_list)
    # Construct Lime explainer on training data (unscaled values are fine; LIME works on numeric arrays)
    lime_explainer = LimeTabularExplainer(
        training_data=np.array(X_train_df),
        feature_names=feature_list,
        class_names=[str(c) for c in np.unique(y_train)],
        mode="classification"
    )

    # pick the instance in unscaled representation (X_test_scaled is scaled; LIME expects same scale as training passed)
    inst_for_lime = np.array(X_test_df.iloc[inst_idx])
    exp = lime_explainer.explain_instance(
        data_row=inst_for_lime,
        predict_fn=best_model.predict_proba,
        num_features=10
    )
    exp.save_to_file(LIME_HTML)
    print("Saved LIME explanation:", LIME_HTML)
except Exception as e:
    print("LIME failed:", e)

# ---------------------------
# Persist remaining artifacts: scalers, encoders, model, feature list already saved above
# ---------------------------
joblib.dump(best_model, BEST_MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(label_encoders, ENCODERS_PATH)
with open(MODEL_PERF_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("All artifacts saved to", ARTIFACT_DIR)
print("Finished training pipeline.")