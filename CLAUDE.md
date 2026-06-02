# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**APEX** is a CRISP-DM-based ML classification project predicting academic burnout risk (Low/High) for ~28,000 students. The model is a tuned Gradient Boosting Classifier (12 features, threshold=0.54) exposed via a Streamlit web app.

**Current metrics:** Test F1-Macro=0.7822, ROC-AUC=0.8689, High-Risk Recall=0.6433

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app/streamlit_app.py

# Retrain the model (leak-safe, overwrites models/ and data/processed/)
python app/retrain.py

# Open the analysis notebook (Jupyter)
jupyter notebook notebooks/final_analysis.ipynb
```

The Streamlit app runs at `http://localhost:8501` and has two tabs: burnout prediction form and project analytics.

## Architecture

### Key Files

- [app/streamlit_app.py](app/streamlit_app.py) — main web UI (2 tabs: prediction + analytics)
- [app/preprocessing.py](app/preprocessing.py) — `ProductionPreprocessor` class (Lean Core, 12 features)
- [app/retrain.py](app/retrain.py) — end-to-end retraining script (stratified split → fit → threshold tune → save)
- [notebooks/final_analysis.ipynb](notebooks/final_analysis.ipynb) — single notebook covering all 6 CRISP-DM phases
- [models/best_model_package.joblib](models/best_model_package.joblib) — serialized model + threshold + metadata
- [models/artifacts/full_preprocessor.pkl](models/artifacts/full_preprocessor.pkl) — fitted `ProductionPreprocessor`

### Preprocessing Pipeline (`ProductionPreprocessor`)

The pipeline is always **fit on train set only** — test/production data only calls `transform()`. Steps:

1. `SimpleImputer` (median for numeric, most_frequent for categorical)
2. `OrdinalEncoder` for ordinal columns (Sınıf Düzeyi, Prompt Yazma Becerisi, Kurum Politikası)
3. `WOEEncoder` for nominal columns (Okunan Bölüm, Birincil Kullanım Amacı)
4. `StandardScaler` on all numeric features
5. Binary missing-value indicators for two columns (Beceri Kalıcılık Skoru, Prompt Yazma Becerisi)

**12 Lean Core features** — the following were dropped after ablation: Araç Çeşitliliği (redundant with Prompt), Ücretli Abonelik (zero CV gain), and the GPA block (Dönem Öncesi/Sonrası GNO — fragile in live scenarios, hurt high-risk recall).

### Data Leak Prevention

- Preprocessor is fit once on train split only
- Threshold (0.54) was selected on an inner validation split (15% of train), not the test set
- Test set is evaluated exactly once (one-shot principle)
- `data/raw/` is never modified

### Notebook Structure

```
Section 0: Hero & Proje Tanıtımı (HTML)
Section 1: Business Understanding       ← Feza
Section 2: Data Understanding / EDA     ← Feza
Section 3: Data Preparation + Pipeline  ← Cenker
Section 4: Modeling                     ← Berkay
Section 5: Evaluation                   ← Berkay
Section 6: Deployment Simulation        ← Ethem
```

### Team & Phase Ownership

| Member | Phases | Responsibility |
|--------|--------|---------------|
| Feza   | 1–2    | Business understanding, EDA |
| Cenker | 3      | Data prep, pipeline, `data/processed/` |
| Berkay | 4–5    | Modeling, evaluation, `models/` |
| Ethem  | 6      | Streamlit app, UI/UX |

## Technical Standards

- All visualizations use **Plotly** (matplotlib is not used)
- Every chart has a 2–3 sentence analyst commentary beneath it
- Model serialization uses `joblib` (`.joblib` extension)
- Stratified split: `test_size=0.2`, `random_state=42`
- At least 10 classification models compared in Section 4
- Required metrics everywhere: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Notebook must run clean with **Restart & Run All** — no hidden state
- `data/raw/` is read-only; `data/processed/` is written only by the retrain pipeline

## Loading the Model for Inference

```python
import joblib
import pandas as pd

preprocessor = joblib.load('models/artifacts/full_preprocessor.pkl')
package = joblib.load('models/best_model_package.joblib')
model = package['model']
threshold = package['threshold_decision']['tuned']['threshold']  # 0.54

X_proc = preprocessor.transform(input_df)
prob = model.predict_proba(X_proc)[:, 1]
prediction = (prob >= threshold).astype(int)  # 1 = Yüksek (High Risk)
```
