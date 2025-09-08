# app/inference_pipeline.py
import os, json, joblib
import pandas as pd

def load_pipeline(
    meta_path="models/xgb_pipeline_fe_meta.json",
    pipe_path=None,
    thr_path="models/xgb_pipeline_fe_threshold.json"
):
    """
    Load a scikit-learn Pipeline that includes preprocessing + model.
    - If meta JSON exists, it will read pipe_path and expected raw columns from there.
    - If meta is missing, it will:
        * use pipe_path if provided, else default to 'models/xgb_pipeline_fe.joblib'
        * introspect the pipeline's ColumnTransformer to discover raw input columns
    - Threshold is taken from thr_path (if present) or defaults to 0.5; env PRED_THRESHOLD overrides.
    """
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if pipe_path is None:
            pipe_path = meta.get("pipe_path")

    if pipe_path is None:
        pipe_path = "models/xgb_pipeline_fe.joblib"  # sensible default

    if not os.path.exists(pipe_path):
        raise FileNotFoundError(
            f"Pipeline file not found: '{pipe_path}'. "
            "Create it by running your training notebook (Cell 3 that saves xgb_pipeline_fe.joblib)."
        )

    pipe = joblib.load(pipe_path)

    # discover expected raw columns
    exp_cols = None
    num_cols = None
    cat_cols = None

    # Prefer meta if it has the lists
    if "num_cols_fe" in meta and "cat_cols_fe" in meta:
        num_cols = list(meta["num_cols_fe"])
        cat_cols = list(meta["cat_cols_fe"])
        exp_cols = num_cols + cat_cols
    else:
        # Introspect the fitted ColumnTransformer inside the pipeline
        preproc = getattr(pipe, "named_steps", {}).get("preprocessor")
        transformers = getattr(preproc, "transformers_", None)
        if transformers is not None:
            for name, transformer, cols in transformers:
                if name == "num":
                    num_cols = list(cols)
                elif name == "cat":
                    cat_cols = list(cols)
            if num_cols is None: num_cols = []
            if cat_cols is None: cat_cols = []
            exp_cols = num_cols + cat_cols

    if not exp_cols:
        raise RuntimeError(
            "Could not determine expected raw columns. "
            "Ensure your pipeline has a fitted ColumnTransformer named 'preprocessor', "
            "or provide 'num_cols_fe' and 'cat_cols_fe' in the meta JSON."
        )

    # Threshold
    thr = 0.5
    if os.path.exists(thr_path):
        try:
            with open(thr_path, "r", encoding="utf-8") as f:
                thr = float(json.load(f).get("threshold", 0.5))
        except Exception:
            pass
    env_thr = os.environ.get("PRED_THRESHOLD")
    if env_thr:
        try:
            thr = float(env_thr)
        except Exception:
            pass

    return {
        "pipe": pipe,
        "exp_cols": exp_cols,
        "threshold": float(thr),
        "meta_path": meta_path,
        "pipe_path": pipe_path,
        "thr_path": thr_path,
    }

def _align_payload(payload: dict, exp_cols):
    """Create a 1-row DataFrame with exactly the raw columns the pipeline expects."""
    row = {c: payload.get(c, None) for c in exp_cols}
    return pd.DataFrame([row], columns=exp_cols)

def predict_one_pipeline(payload: dict, arts=None):
    arts = arts or load_pipeline()
    pipe = arts["pipe"]
    X = _align_payload(payload, arts["exp_cols"])
    proba = float(pipe.predict_proba(X)[0, 1])
    label = int(proba >= arts["threshold"])
    return {"churn_prob": round(proba, 4), "churn_label": label, "threshold": arts["threshold"]}

def predict_batch_pipeline(items, arts=None):
    arts = arts or load_pipeline()
    out = []
    for obj in items:
        if not isinstance(obj, dict):
            raise ValueError("Each item must be a JSON object.")
        out.append(predict_one_pipeline(obj, arts))
    return out
