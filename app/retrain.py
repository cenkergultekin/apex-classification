import os
import sys
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)

# Setup paths relative to script (app/retrain.py → project root = parent of app/)
APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent.resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.preprocessing import ProductionPreprocessor

# Setup paths
MODELS_DIR = PROJECT_ROOT / 'models'
ARTIFACT_DIR = MODELS_DIR / 'artifacts'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load cleaned data and split exactly as in the notebook
TARGET = 'Tükenmişlik'
CLEANED_PATH = PROCESSED_DIR / 'ai_student_impact_cleaned.csv'

print(f"Loading cleaned dataset from {CLEANED_PATH}...")
df_cleaned = pd.read_csv(CLEANED_PATH)

# Exclude 'Orta' if present (already removed, but let's be safe)
df_cleaned = df_cleaned[df_cleaned[TARGET] != 'Orta'].reset_index(drop=True)

X_raw_all = df_cleaned.drop(columns=[TARGET])
y_raw_all = df_cleaned[TARGET]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw_all,
    y_raw_all,
    test_size=0.20,
    stratify=y_raw_all,
    random_state=42
)

# Inner train/validation split — threshold validation üzerinde seçilir,
# böylece test seti tek-sefer prensibi korunur (test-leak yok).
X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
    X_train_raw,
    y_train,
    test_size=0.15,
    stratify=y_train,
    random_state=42,
)

print(f"Split done: Train={X_train_raw.shape}, Test={X_test_raw.shape}")
print(f"Inner split:  Train(fit)={X_tr_raw.shape}, Validation(threshold)={X_val_raw.shape}")

# Encode target Düşük -> 0, Yüksek -> 1
TARGET_ORDER = ['Düşük', 'Yüksek']
target_map = {v: i for i, v in enumerate(TARGET_ORDER)}
y_train_enc = y_train.map(target_map).astype(int)
y_test_enc = y_test.map(target_map).astype(int)
y_tr_enc = y_tr.map(target_map).astype(int)
y_val_enc = y_val.map(target_map).astype(int)

# 2. Fit Preprocessor (sadece outer train üzerinde — production artifact bu)
print("Fitting preprocessor on outer train set...")
prep = ProductionPreprocessor()
prep.fit(X_train_raw, y_train_enc)

X_train_final = prep.transform(X_train_raw)
X_test_final = prep.transform(X_test_raw)

# Inner train/val transformları için ayrı bir preprocessor (threshold seçimi temiz olsun)
prep_inner = ProductionPreprocessor()
prep_inner.fit(X_tr_raw, y_tr_enc)
X_tr_final = prep_inner.transform(X_tr_raw)
X_val_final = prep_inner.transform(X_val_raw)

print(f"Preprocessed train shape: {X_train_final.shape}")
print(f"Preprocessed test shape: {X_test_final.shape}")

# Save Preprocessor Artifact
preprocessor_path = ARTIFACT_DIR / 'full_preprocessor.pkl'
joblib.dump(prep, preprocessor_path)
print(f"Saved preprocessor artifact to {preprocessor_path}")

# Save Processed CSVs
train_out = X_train_final.copy()
train_out[TARGET] = y_train.values
test_out = X_test_final.copy()
test_out[TARGET] = y_test.values

train_out.to_csv(PROCESSED_DIR / 'train.csv', index=False, encoding='utf-8-sig')
test_out.to_csv(PROCESSED_DIR / 'test.csv', index=False, encoding='utf-8-sig')
print(f"Saved preprocessed train/test CSVs to {PROCESSED_DIR}")

# 3. Train Gradient Boosting Classifier with best params
# "best_params": { "subsample": 0.85, "n_estimators": 180, "min_samples_leaf": 20, "max_depth": 3, "learning_rate": 0.05 }
print("Training Gradient Boosting model with optimal parameters...")
best_params = {
    "subsample": 0.85,
    "n_estimators": 180,
    "min_samples_leaf": 20,
    "max_depth": 3,
    "learning_rate": 0.05,
    "random_state": 42
}

model = GradientBoostingClassifier(**best_params)
model.fit(X_train_final, y_train_enc)
print("Model training completed successfully.")

# Save Model
best_model_path = MODELS_DIR / 'best_model.joblib'
joblib.dump(model, best_model_path)
print(f"Saved raw model to {best_model_path}")

# ── Threshold seçimi: validation seti üzerinde (test-leak yok)
# Inner model (sadece X_tr üzerinde fit) ile validation üzerinde F1-macro optimum threshold bul.
inner_model = GradientBoostingClassifier(**best_params)
inner_model.fit(X_tr_final, y_tr_enc)
y_val_proba = inner_model.predict_proba(X_val_final)[:, 1]

best_thr, best_f1_val = 0.5, -1.0
for _thr in np.arange(0.30, 0.71, 0.01):
    _f1 = f1_score(y_val_enc, (y_val_proba >= _thr).astype(int), average='macro')
    if _f1 > best_f1_val:
        best_f1_val, best_thr = float(_f1), float(round(_thr, 2))

print(f"\n— THRESHOLD SEÇİMİ (Validation Set) —")
print(f"  Best threshold = {best_thr}  (validation F1-macro={best_f1_val:.4f})")

# Test seti üzerinde sadece UYGULA (öğrenme yok, tek-sefer ölçüm)
y_test_proba = model.predict_proba(X_test_final)[:, 1]

def _score_at(thr: float) -> dict:
    yp = (y_test_proba >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_test_enc, yp)),
        "f1_macro": float(f1_score(y_test_enc, yp, average='macro')),
        "recall_yuksek": float(recall_score(y_test_enc, yp)),
        "precision_yuksek": float(precision_score(y_test_enc, yp, zero_division=0)),
        "yuksek_tahmin_orani": float(yp.mean()),
    }

default_scores = _score_at(0.5)
tuned_scores = _score_at(best_thr)
roc_auc = float(roc_auc_score(y_test_enc, y_test_proba))

print(f"\n— SKORLAR (Lean Core, threshold val-üzerinde seçildi) —")
print(f"  Default (0.5): F1={default_scores['f1_macro']:.4f}  Recall(Yüksek)={default_scores['recall_yuksek']:.4f}")
print(f"  Tuned ({best_thr}): F1={tuned_scores['f1_macro']:.4f}  Recall(Yüksek)={tuned_scores['recall_yuksek']:.4f}")
print(f"  ROC-AUC: {roc_auc:.4f}")

threshold_decision = {
    "model": "Gradient Boosting",
    "model_source": "Lean Core (Phase 3 refactor)",
    "selected_by": "validation_f1_macro (inner train/val split, test seti tek-sefer kullanıldı)",
    "validation_best_f1_macro": float(best_f1_val),
    "target_order": TARGET_ORDER,
    "positive_class": "Yüksek",
    "roc_auc": roc_auc,
    "default": default_scores,
    "tuned": tuned_scores,
}

model_package = {
    'model': model,
    'model_name': 'Gradient Boosting',
    'model_source': 'final_model',
    'target': 'Tükenmişlik',
    'target_order': TARGET_ORDER,
    'positive_class': 'Yüksek',
    'feature_count': model.n_features_in_,
    'threshold_decision': threshold_decision,
    'middle_margin_decision': threshold_decision,
    'test_eval_decision': {
        'model': 'Gradient Boosting',
        'model_source': 'Lean Core (Phase 3 refactor)',
        'prediction_policy': f'Tuned binary threshold {best_thr}',
        'threshold': best_thr,
        'accuracy': tuned_scores['accuracy'],
        'f1_macro': tuned_scores['f1_macro'],
        'recall_macro': float(recall_score(y_test_enc, (y_test_proba >= best_thr).astype(int), average='macro')),
        'recall_yuksek': tuned_scores['recall_yuksek'],
        'precision_yuksek': tuned_scores['precision_yuksek'],
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix(y_test_enc, (y_test_proba >= best_thr).astype(int)).tolist(),
    },
    'best_params': best_params,
}

package_path = MODELS_DIR / 'best_model_package.joblib'
joblib.dump(model_package, package_path)
print(f"Saved model package to {package_path}")

# Write/Update metadata
metadata_path = MODELS_DIR / 'best_model_metadata.json'
metadata = {
    'model_name': 'Gradient Boosting',
    'model_source': 'final_model',
    'target': model_package['target'],
    'target_order': model_package['target_order'],
    'positive_class': model_package['positive_class'],
    'feature_count': model_package['feature_count'],
    'best_params': model_package['best_params'],
    'threshold_decision': model_package['threshold_decision'],
    'middle_margin_decision': model_package['middle_margin_decision'],
    'test_eval_decision': model_package['test_eval_decision'],
}
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f"Updated metadata to {metadata_path}")

print("Retraining completed successfully and all artifacts updated!")
