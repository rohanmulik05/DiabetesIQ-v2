# DiabetesIQ — Dual Model Prediction System

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

---

## Step 1 — Save Preprocessors from Colab (IMPORTANT)

Your models were trained with fitted preprocessors (RobustScaler, Pipeline).
**You must save these objects and place them in `/models/` alongside your model files.**
Without them, predictions will be wrong.

### Clinical Notebook — add to the END of your Colab:

```python
import joblib

# 1. Save the RobustScaler (already fitted as `transformer` in your code)
joblib.dump(transformer, "clinical_preprocessor.pkl")

# 2. Save outcome-stratified medians
#    Run this AFTER the NaN-filling loop, BEFORE LOF removal
#    Use df_original = your df before LOF step, or re-derive from the filled df
medians = {}
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    medians[col] = {
        0: df.loc[df['Outcome'] == 0, col].median(),
        1: df.loc[df['Outcome'] == 1, col].median()
    }
joblib.dump(medians, "clinical_medians.pkl")

# 3. Save Insulin IQR upper cap
#    `upper` is already computed in your outlier section
joblib.dump(upper, "clinical_insulin_cap.pkl")

# 4. Save your chosen ensemble model (pick one)
joblib.dump(ensemble_soft,     "clinical_model.pkl")   # OR
joblib.dump(ensemble_hard,     "clinical_model.pkl")   # OR
joblib.dump(ensemble_weighted, "clinical_model.pkl")   # OR
joblib.dump(ensemble_stacking, "clinical_model.pkl")
```

### Lifestyle Notebook — add to the END of your Colab:

```python
import joblib

# 1. Save the full sklearn pipeline (OHE + MinMaxScaler)
#    `pipeline` is already defined in your code
joblib.dump(pipeline, "lifestyle_pipeline.pkl")

# 2. Save your chosen ensemble model (pick one)
joblib.dump(ensemble_soft,     "lifestyle_model.pkl")  # OR
joblib.dump(ensemble_hard,     "lifestyle_model.pkl")  # OR
joblib.dump(ensemble_weighted, "lifestyle_model.pkl")  # OR
joblib.dump(ensemble_stacking, "lifestyle_model.pkl")
```

---

## Step 2 — Folder Structure

```
diabetes-app/
├── models/
│   ├── clinical_model.pkl          ← your clinical ensemble
│   ├── clinical_preprocessor.pkl   ← RobustScaler
│   ├── clinical_medians.pkl        ← NaN-fill medians
│   ├── clinical_insulin_cap.pkl    ← Insulin IQR cap float
│   ├── lifestyle_model.pkl         ← your lifestyle ensemble
│   └── lifestyle_pipeline.pkl      ← OHE + MinMaxScaler pipeline
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
└── README.md
```

---

## How Preprocessing Works

### Clinical (mirrors your Colab exactly)
| Step | What happens |
|---|---|
| 1 | Input values of 0 in Glucose, BP, SkinThickness, Insulin, BMI → replaced with NaN |
| 2 | NaN filled with average of outcome-stratified medians from training |
| 3 | Insulin capped at IQR upper bound from training |
| 4 | NewBMI, NewInsulinScore, NewGlucose engineered |
| 5 | One-hot encoded with drop_first=True (same as pd.get_dummies) |
| 6 | RobustScaler applied to 8 numeric columns |
| 7 | Scaled numerics + OHE columns concatenated → model input |

### Lifestyle (mirrors your Colab exactly)
| Step | What happens |
|---|---|
| 1 | Binary inputs (0/1) converted back to 'Yes'/'No' strings |
| 2 | Gender 0/1 converted to 'Female'/'Male' |
| 3 | DataFrame fed to `pipeline.transform()` → OHE + MinMaxScaler |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `GET /status` | GET | Check which models/preprocessors are loaded |
| `POST /predict/clinical` | POST | Run clinical model |
| `POST /predict/lifestyle` | POST | Run lifestyle model |
| `POST /predict/combined` | POST | Run both + average ensemble |

---

## Notes
- The combined prediction uses **equal-weight averaging** of both model probabilities.
- The clinical model has a graceful fallback scaler using hardcoded PIMA constants if `clinical_preprocessor.pkl` is missing — but accuracy will be reduced. Save the scaler for best results.
- The lifestyle model **requires** `lifestyle_pipeline.pkl` (no fallback possible without it).
