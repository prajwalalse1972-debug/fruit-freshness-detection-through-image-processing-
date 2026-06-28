from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
import joblib
import os, uuid, csv
from datetime import datetime
from feature_extraction import extract_features
from export_pdf import generate_pdf

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model      = joblib.load("fruit_model.pkl")
FRUIT_NAMES = model["fruit_names"]   # 5 fruits now

# ── Freshness prediction ───────────────────────────────────
def predict_freshness(image_path, fruit_type):
    fv = extract_features(image_path, fruit_type=fruit_type)
    if fv is None:
        return None, 0
    fv_scaled  = (fv - model["mean"]) / model["std"]
    fv_top     = fv_scaled[model["top_feats"]]
    k          = model["k"]
    dists      = np.sqrt(((model["X_train_fresh"] - fv_top) ** 2).sum(axis=1))
    k_labels   = model["y_train_fresh"][np.argsort(dists)[:k]]
    votes      = k_labels.sum()
    label      = 1 if votes > k // 2 else 0
    confidence = round((votes / k if label == 1 else (k - votes) / k) * 100, 1)
    return label, confidence

# ── Routes ─────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext      = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        fruit_type = request.form.get("fruit_type", "")
        if fruit_type not in FRUIT_NAMES:
            return jsonify({"error": "Please select a fruit type"}), 400

        label, freshness_conf = predict_freshness(filepath, fruit_type)
        if label is None:
            return jsonify({"error": "Could not process image"}), 500

        # CSV log
        log_file    = "predictions.csv"
        file_exists = os.path.exists(log_file)
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Fruit Type", "Image File",
                                 "Result", "Freshness Confidence %"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                fruit_type.capitalize(),
                file.filename,
                "FRESH" if label == 0 else "ROTTEN",
                freshness_conf,
            ])

        return jsonify({
            "fruit":          fruit_type,
            "label":          "FRESH" if label == 0 else "ROTTEN",
            "freshness_conf": freshness_conf,
            "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route("/export-pdf")
def export_pdf_route():
    success, result = generate_pdf()
    if success:
        return send_file(result, as_attachment=True,
                         download_name="Freshness_Report.pdf",
                         mimetype="application/pdf")
    return jsonify({"error": result}), 500

if __name__ == "__main__":
    print("\n🍎 Fruit Freshness Web App")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=False, port=5000, host="0.0.0.0", use_reloader=False)
    
