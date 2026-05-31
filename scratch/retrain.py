import os
import sys
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Setup paths relative to script
SCRATCH_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRATCH_DIR.parent.resolve()

sys.path.append(str(PROJECT_ROOT))

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

print(f"Split done: Train={X_train_raw.shape}, Test={X_test_raw.shape}")

# Encode target Düşük -> 0, Yüksek -> 1
TARGET_ORDER = ['Düşük', 'Yüksek']
target_map = {v: i for i, v in enumerate(TARGET_ORDER)}
y_train_enc = y_train.map(target_map).astype(int)
y_test_enc = y_test.map(target_map).astype(int)

# 2. Fit Preprocessor
print("Fitting preprocessor...")
prep = ProductionPreprocessor(use_compact_core=True)
prep.fit(X_train_raw, y_train_enc)

X_train_final = prep.transform(X_train_raw)
X_test_final = prep.transform(X_test_raw)

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

# Build and Save Package
threshold_decision = {
    "model": "Gradient Boosting",
    "model_source": "Tuned model (4.4)",
    "selected_by": "validation_f1_macro_then_high_recall",
    "target_order": TARGET_ORDER,
    "positive_class": "Yüksek",
    "default": {
        "threshold": 0.5,
        "f1_macro": 0.8057705598826115,
        "recall_yuksek": 0.7088506207448939,
        "precision_yuksek": 0.8352996696554978,
        "yuksek_tahmin_orani": 0.367180731242419
    },
    "tuned": {
        "threshold": 0.53,
        "f1_macro": 0.8082508383349987,
        "recall_yuksek": 0.6948338005606728,
        "precision_yuksek": 0.8555226824457594,
        "yuksek_tahmin_orani": 0.3514122335817016
    }
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
        'model_source': 'Tuned model (4.4)',
        'prediction_policy': 'Tuned binary threshold 0.53',
        'threshold': 0.53,
        'accuracy': 0.8028413028413028,
        'f1_macro': 0.7925432064265425,
        'recall_macro': 0.7871144808332922,
        'recall_yuksek': 0.6701361088871097,
        'confusion_matrix': [
            [2960, 314],
            [824, 1674]
        ],
        'confusion_matrix_normalized': [
            [0.9040928527794746, 0.09590714722052535],
            [0.32986389111289033, 0.6701361088871097]
        ]
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
