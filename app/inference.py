# app/inference.py
import os
import glob
import json
import joblib
import pandas as pd

# ---------- choose schema (prefer a stable pointer) ----------
def _latest_schema_path():
    explicit = "models/columns_latest.json"
    if os.path.exists(explicit):
        return explicit
    files = sorted(glob.glob("models/columns_*.json"))
    if not files:
        raise FileNotFoundError("No models/columns_*.json found in models/.")
    return files[-1]

# ---------- load artifacts ----------
def load_artifacts(schema_path=None):
    schema_path = schema_path or _latest_schema_path()
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    model_path  = schema["model_path"]
    scaler_path = schema.get("scaler_path")
    features    = schema["feature_columns"]

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path else None

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "schema_path": schema_path,
    }

# ---------- helpers: raw → one-hot row in correct column order ----------
_BINARY_YESNO = {"Yes": 1, "No": 0, 1: 1, 0: 0, "1": 1, "0": 0}

def _set_one_hot(row: pd.Series, features, prefix: str, category: str):
    """With drop_first encoding, only non-base categories exist as prefix_value."""
    col = f"{prefix}_{category}"
    if col in row.index:
        row[col] = 1

def preprocess_one(payload: dict, features, scaler=None) -> pd.DataFrame:
    # Start with all zeros in exact training column order
    row = pd.Series(0.0, index=features, dtype="float64")

    # 1) numeric fields (set if present)
    for num in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if num in payload and num in row.index:
            try:
                row[num] = float(payload[num])
            except Exception:
                pass

    # 2) Yes/No binaries that became *_Yes after get_dummies(drop_first=True)
    for base in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        col = f"{base}_Yes"
        if base in payload and col in row.index:
            row[col] = _BINARY_YESNO.get(payload[base], 0)

    # 3) multi-category one-hots
    if "gender" in payload:
        _set_one_hot(row, features, "gender", str(payload["gender"]))
    if "MultipleLines" in payload:
        _set_one_hot(row, features, "MultipleLines", str(payload["MultipleLines"]))
    for svc in ["InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
                "TechSupport","StreamingTV","StreamingMovies"]:
        if svc in payload:
            _set_one_hot(row, features, svc, str(payload[svc]))
    if "Contract" in payload:
        _set_one_hot(row, features, "Contract", str(payload["Contract"]))
    if "PaymentMethod" in payload:
        _set_one_hot(row, features, "PaymentMethod", str(payload["PaymentMethod"]))

    X = pd.DataFrame([row.values], columns=features)

    # 4) apply scaler if present
    if scaler is not None:
        sc_cols = getattr(scaler, "feature_names_in_", None)
        if sc_cols is None:
            sc_cols = [c for c in ["tenure","MonthlyCharges","TotalCharges"] if c in X.columns]
        cols = [c for c in X.columns if c in sc_cols]
        if cols:
            X.loc[:, cols] = scaler.transform(X[cols])

    return X

def predict_one(payload: dict, artifacts=None) -> dict:
    artifacts = artifacts or load_artifacts()
    model   = artifacts["model"]
    scaler  = artifacts["scaler"]
    features= artifacts["features"]

    X = preprocess_one(payload, features, scaler)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0, 1])
    else:
        score = float(model.decision_function(X)[0])
        import math
        prob = 1.0 / (1.0 + math.exp(-score))

    label = int(prob >= 0.5)
    return {"churn_prob": round(prob, 4), "churn_label": label}
