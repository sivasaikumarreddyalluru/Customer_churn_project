# app/app.py
from flask import Flask, request, jsonify
import os, json
from inference_pipeline import load_pipeline, predict_one_pipeline, predict_batch_pipeline

app = Flask(__name__)

# Load pipeline once at startup
ARTS = load_pipeline()
print("[Startup] Using pipeline:", ARTS["pipe_path"])
print("[Startup] Meta file:", ARTS["meta_path"])
print("[Startup] Threshold:", ARTS["threshold"])

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "pipeline": ARTS["pipe_path"],
        "meta": ARTS["meta_path"],
        "threshold": ARTS["threshold"]
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Expected a single JSON object (one customer)."}), 400
        return jsonify(predict_one_pipeline(data, ARTS)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    try:
        # CSV upload
        if "file" in request.files:
            from io import StringIO
            import pandas as pd
            content = request.files["file"].read().decode("utf-8")
            df = pd.read_csv(StringIO(content))
            items = [row.to_dict() for _, row in df.iterrows()]
            preds = predict_batch_pipeline(items, ARTS)
            return jsonify({"n": len(preds), "predictions": preds}), 200

        # JSON array
        data = request.get_json(force=True)
        if not isinstance(data, list):
            return jsonify({"error": "Expected a JSON array or a CSV file under 'file'."}), 400
        preds = predict_batch_pipeline(data, ARTS)
        return jsonify({"n": len(preds), "predictions": preds}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
