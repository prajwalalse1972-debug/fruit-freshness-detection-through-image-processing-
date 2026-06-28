import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
import joblib
import csv
import os
import threading
from datetime import datetime
from PIL import Image, ImageTk
from feature_extraction import extract_features

# ── Load model ────────────────────────────────────────────
model = joblib.load("fruit_model.pkl")
FRUIT_NAMES = model["fruit_names"]


# ── Shape feature extractor (fruit-type classifier only) ──
def extract_shape_features(image_path):
    """5 geometry features using a neutral HSV mask."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image   = cv2.resize(image, (224, 224))
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv,
                       np.array([0,  25, 25]),
                       np.array([179, 255, 255]))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(5)

    c            = max(contours, key=cv2.contourArea)
    area         = cv2.contourArea(c)
    perimeter    = cv2.arcLength(c, True) + 1e-6
    x, y, w, h   = cv2.boundingRect(c)
    circularity  = (4 * np.pi * area) / (perimeter ** 2)
    extent       = area / (w * h + 1e-6)
    aspect_ratio = w / (h + 1e-6)
    hull         = cv2.convexHull(c)
    solidity     = area / (cv2.contourArea(hull) + 1e-6)
    area_norm    = area / (224 * 224)

    return np.array([circularity, extent, aspect_ratio, solidity, area_norm])


# ── KNN distance helper ───────────────────────────────────
def _knn_top(X_train, fv, k):
    dists = np.sqrt(((X_train - fv) ** 2).sum(axis=1))
    return np.argsort(dists)[:k]


# ── Classifier 2: fruit type (shape-based) ────────────────
def detect_fruit_type(image_path):
    sf = extract_shape_features(image_path)
    if sf is None:
        return None, 0

    sf_scaled = (sf - model["sf_mean"]) / model["sf_std"]
    k         = model["k_fruit"]
    k_idx     = _knn_top(model["X_train_fruit"], sf_scaled, k)
    k_labels  = model["y_train_fruit"][k_idx]

    counts    = np.bincount(k_labels.astype(int), minlength=len(FRUIT_NAMES))
    pred_idx  = int(np.argmax(counts))
    confidence = round(counts[pred_idx] / k * 100, 1)
    return FRUIT_NAMES[pred_idx], confidence


# ── Classifier 1: freshness (binary) ─────────────────────
def predict_freshness(image_path, fruit_type):
    fv = extract_features(image_path, fruit_type=fruit_type)
    if fv is None:
        return None, 0

    fv_scaled = (fv - model["mean"]) / model["std"]
    fv_top    = fv_scaled[model["top_feats"]]
    k         = model["k"]
    k_idx     = _knn_top(model["X_train_fresh"], fv_top, k)
    k_labels  = model["y_train_fresh"][k_idx]
    votes     = k_labels.sum()
    label     = 1 if votes > k // 2 else 0
    confidence = round((votes / k if label == 1 else (k - votes) / k) * 100, 1)
    return label, confidence


# ── CSV Logger ────────────────────────────────────────────
def log_prediction(fruit_type, image_path, label, freshness_conf, fruit_conf):
    log_file   = "predictions.csv"
    file_exists = os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Fruit Type", "Detected By",
                             "Image File", "Result",
                             "Freshness Confidence %", "Fruit Confidence %"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            fruit_type.capitalize(),
            "Auto",
            os.path.basename(image_path),
            "FRESH" if label == 0 else "ROTTEN",
            freshness_conf,
            fruit_conf,
        ])


# ── GUI ───────────────────────────────────────────────────
class FreshApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit Freshness Detector")
        self.root.geometry("520x740")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(False, False)

        tk.Label(root, text="🍎 Fruit Freshness Detector",
                 font=("Helvetica", 18, "bold"),
                 bg="#1e1e2e", fg="#cdd6f4").pack(pady=20)

        # Auto-detect badge
        self.mode_var = tk.StringVar(value="🤖  Auto-detecting fruit type")
        self.mode_label = tk.Label(root, textvariable=self.mode_var,
                                   font=("Helvetica", 10, "italic"),
                                   bg="#313244", fg="#89b4fa",
                                   padx=10, pady=4)
        self.mode_label.pack(pady=(0, 8))

        # Override toggle
        self.override_var = tk.BooleanVar(value=False)
        tk.Checkbutton(root, text="Override: select fruit manually",
                       variable=self.override_var,
                       command=self.toggle_override,
                       bg="#1e1e2e", fg="#a6adc8",
                       selectcolor="#313244",
                       activebackground="#1e1e2e",
                       font=("Helvetica", 10)).pack()

        # Manual dropdown (disabled by default)
        self.fruit_var = tk.StringVar(value="banana")
        self.fruit_menu = ttk.Combobox(root, textvariable=self.fruit_var,
                                        values=FRUIT_NAMES, state="disabled",
                                        font=("Helvetica", 12), width=20)
        self.fruit_menu.pack(pady=6)

        # Upload button
        tk.Button(root, text="📁  Upload Image",
                  font=("Helvetica", 12, "bold"),
                  bg="#89b4fa", fg="#1e1e2e",
                  activebackground="#74c7ec",
                  padx=20, pady=8, bd=0,
                  command=self.upload_image).pack(pady=12)

        # Image display
        self.img_label = tk.Label(root, bg="#313244", width=300, height=280)
        self.img_label.pack(pady=5)

        # Detected fruit
        self.fruit_detected_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.fruit_detected_var,
                 font=("Helvetica", 11),
                 bg="#1e1e2e", fg="#89dceb").pack(pady=(8, 0))

        # Result
        self.result_var = tk.StringVar(value="Upload an image to begin")
        self.result_label = tk.Label(root, textvariable=self.result_var,
                                      font=("Helvetica", 15, "bold"),
                                      bg="#1e1e2e", fg="#a6adc8")
        self.result_label.pack(pady=6)

        # Confidence
        self.conf_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.conf_var,
                 font=("Helvetica", 11),
                 bg="#1e1e2e", fg="#6c7086").pack()

        # Log status
        self.log_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.log_var,
                 font=("Helvetica", 9),
                 bg="#1e1e2e", fg="#a6e3a1").pack(pady=4)

        # Buttons
        btn_frame = tk.Frame(root, bg="#1e1e2e")
        btn_frame.pack(pady=4)
        tk.Button(btn_frame, text="📊  View Log",
                  font=("Helvetica", 10), bg="#313244", fg="#cdd6f4",
                  activebackground="#45475a", padx=10, pady=5, bd=0,
                  command=self.view_log).pack(side="left", padx=6)
        tk.Button(btn_frame, text="📄  Export PDF",
                  font=("Helvetica", 10), bg="#313244", fg="#cdd6f4",
                  activebackground="#45475a", padx=10, pady=5, bd=0,
                  command=self.export_pdf).pack(side="left", padx=6)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(root, textvariable=self.status_var,
                 font=("Helvetica", 9), bg="#181825", fg="#585b70",
                 width=60).pack(side="bottom", fill="x", pady=5)

    def toggle_override(self):
        if self.override_var.get():
            self.fruit_menu.config(state="readonly")
            self.mode_var.set("🖐  Manual fruit selection")
            self.mode_label.config(fg="#f9e2af")
        else:
            self.fruit_menu.config(state="disabled")
            self.mode_var.set("🤖  Auto-detecting fruit type")
            self.mode_label.config(fg="#89b4fa")

    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return

        # Show image immediately (UI stays responsive)
        img   = Image.open(path).resize((300, 280))
        photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=photo, width=300, height=280)
        self.img_label.image = photo

        # Reset result labels
        self.result_var.set("Analysing...")
        self.result_label.configure(fg="#a6adc8")
        self.fruit_detected_var.set("")
        self.conf_var.set("")
        self.log_var.set("")
        self.status_var.set("Step 1/2 — Detecting fruit type...")
        self.root.update()

        # Run detection in background thread so UI doesn't freeze
        def run_analysis():
            try:
                # Step 1: fruit type
                if self.override_var.get():
                    fruit_type  = self.fruit_var.get()
                    fruit_conf  = 100.0
                    detect_text = f"🍓 Fruit: {fruit_type.capitalize()}  (manual)"
                else:
                    fruit_type, fruit_conf = detect_fruit_type(path)
                    if fruit_type is None:
                        self.root.after(0, lambda: self._show_error("❌ Could not read image"))
                        return
                    detect_text = (f"🍓 Detected: {fruit_type.capitalize()}"
                                   f"  ({fruit_conf}% confident)")

                # Update UI from main thread
                self.root.after(0, lambda: self.fruit_detected_var.set(detect_text))
                self.root.after(0, lambda: self.status_var.set("Step 2/2 — Checking freshness..."))

                # Step 2: freshness
                label, freshness_conf = predict_freshness(path, fruit_type)

                if label is None:
                    self.root.after(0, lambda: self._show_error("❌ Freshness check failed"))
                    return

                # Final UI update (must happen on main thread)
                def update_ui():
                    if label == 0:
                        self.result_var.set("✅ FRESH")
                        self.result_label.configure(fg="#a6e3a1")
                    else:
                        self.result_var.set("🚫 ROTTEN")
                        self.result_label.configure(fg="#f38ba8")
                    self.conf_var.set(f"Freshness confidence: {freshness_conf}%")
                    log_prediction(fruit_type, path, label, freshness_conf, fruit_conf)
                    self.log_var.set("✓ Prediction saved to predictions.csv")
                    self.status_var.set(f"Done — {os.path.basename(path)}")

                self.root.after(0, update_ui)

            except Exception as e:
                self.root.after(0, lambda err=e: self._show_error(f"❌ Error: {str(err)}"))

        threading.Thread(target=run_analysis, daemon=True).start()

    def _show_error(self, msg):
        self.result_var.set(msg)
        self.result_label.configure(fg="#f38ba8")
        self.status_var.set("Error")

    def view_log(self):
        log_file = "predictions.csv"
        if not os.path.exists(log_file):
            self.log_var.set("No predictions logged yet!")
            return
        with open(log_file, "r") as f:
            lines = f.readlines()

        popup = tk.Toplevel(self.root)
        popup.title("Prediction Log")
        popup.geometry("780x400")
        popup.configure(bg="#1e1e2e")

        tk.Label(popup, text="📊 Prediction History",
                 font=("Helvetica", 14, "bold"),
                 bg="#1e1e2e", fg="#cdd6f4").pack(pady=10)

        frame = tk.Frame(popup, bg="#1e1e2e")
        frame.pack(fill="both", expand=True, padx=10, pady=5)
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")
        text = tk.Text(frame, bg="#313244", fg="#cdd6f4",
                       font=("Courier", 10), yscrollcommand=scrollbar.set)
        text.pack(fill="both", expand=True)
        scrollbar.config(command=text.yview)
        for line in lines:
            text.insert("end", line)
        text.config(state="disabled")

        tk.Label(popup, text=f"Total predictions: {len(lines)-1}",
                 font=("Helvetica", 10),
                 bg="#1e1e2e", fg="#a6adc8").pack(pady=5)

    def export_pdf(self):
        from export_pdf import generate_pdf
        success, msg = generate_pdf()
        if success:
            self.log_var.set(f"✓ PDF saved as: {msg}")
            self.status_var.set("PDF report exported successfully!")
        else:
            self.log_var.set(f"✗ {msg}")


# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = FreshApp(root)
    root.mainloop()
