import numpy as np
import joblib
import os
import cv2

# ════════════════════════════════════════════════════════════
# CONFIG — 5 fruits now
# ════════════════════════════════════════════════════════════
FRUIT_LABEL = {"banana": 0, "apple": 1, "mango": 2, "orange": 3, "strawberry": 4}
FRUIT_NAMES = ["banana", "apple", "mango", "orange", "strawberry"]

DATASET_DIR = r"C:\Users\LENOVO\OneDrive\Desktop\IDT project 02\Fruits_Vegetables_Dataset(12000)\Fruits"

FOLDER_MAP = {
    ("banana",     "fresh"):  "FreshBanana",
    ("banana",     "rotten"): "RottenBanana",
    ("apple",      "fresh"):  "FreshApple",
    ("apple",      "rotten"): "RottenApple",
    ("mango",      "fresh"):  "FreshMango",
    ("mango",      "rotten"): "RottenMango",
    ("orange",     "fresh"):  "FreshOrange",
    ("orange",     "rotten"): "RottenOrange",
    ("strawberry", "fresh"):  "FreshStrawberry",
    ("strawberry", "rotten"): "RottenStrawberry",
}

# ════════════════════════════════════════════════════════════
# COMBINED SHAPE + COLOR EXTRACTOR (for fruit-type classifier)
# ════════════════════════════════════════════════════════════
def extract_fruit_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image   = cv2.resize(image, (224, 224))
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask    = cv2.inRange(hsv, np.array([0, 25, 25]), np.array([179, 255, 255]))

    # Shape (5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
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
        shape_feats  = np.array([circularity, extent, aspect_ratio, solidity, area_norm])
    else:
        shape_feats = np.zeros(5)

    # Color (5) — strawberry is distinctly red which helps a lot
    fruit_pixels = hsv[mask > 0]
    if len(fruit_pixels) > 0:
        mean_h      = np.mean(fruit_pixels[:, 0])
        mean_s      = np.mean(fruit_pixels[:, 1])
        mean_v      = np.mean(fruit_pixels[:, 2])
        hue_hist, _ = np.histogram(fruit_pixels[:, 0], bins=8, range=(0, 180))
        hue_hist    = hue_hist / (hue_hist.sum() + 1e-6)
        peak1       = float(np.argmax(hue_hist))
        peak2       = float(np.argsort(hue_hist)[-2])
        color_feats = np.array([mean_h, mean_s, mean_v, peak1, peak2])
    else:
        color_feats = np.zeros(5)

    return np.concatenate([shape_feats, color_feats])  # 10 values


# ════════════════════════════════════════════════════════════
# BUILD FRUIT-TYPE FEATURE MATRIX
# ════════════════════════════════════════════════════════════
print("Building fruit-type feature matrix (5 fruits, shape+color)...\n")
X_fruit_list, y_fruit_list = [], []
MAX_PER_FOLDER = 400

for fruit in FRUIT_NAMES:
    for status in ["fresh", "rotten"]:
        folder_name = FOLDER_MAP[(fruit, status)]
        folder      = os.path.join(DATASET_DIR, folder_name)
        if not os.path.exists(folder):
            print(f"  ⚠ Not found: {folder}")
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))][:MAX_PER_FOLDER]

        count = 0
        for fname in files:
            feats = extract_fruit_features(os.path.join(folder, fname))
            if feats is not None:
                X_fruit_list.append(feats)
                y_fruit_list.append(FRUIT_LABEL[fruit])
                count += 1

        print(f"  {folder_name}: {count} images → {fruit}")

X_fruit = np.array(X_fruit_list)
y_fruit = np.array(y_fruit_list)
print(f"\nFruit matrix: {X_fruit.shape}")

# Fruit train/test split
np.random.seed(42)
idx   = np.random.permutation(len(X_fruit))
split = int(0.8 * len(X_fruit))
Xf_tr, Xf_te = X_fruit[idx[:split]], X_fruit[idx[split:]]
yf_tr, yf_te = y_fruit[idx[:split]], y_fruit[idx[split:]]

sf_mean  = Xf_tr.mean(axis=0)
sf_std   = Xf_tr.std(axis=0) + 1e-8
Xf_tr_s  = (Xf_tr - sf_mean) / sf_std
Xf_te_s  = (Xf_te - sf_mean) / sf_std


# ════════════════════════════════════════════════════════════
# FRESHNESS DATA — rebuild from X.npy/y.npy
# (run build_dataset.py first to include strawberry)
# ════════════════════════════════════════════════════════════
print("\nLoading freshness dataset...")
X = np.load("X.npy")
y = np.load("y.npy")
print(f"X shape: {X.shape}  |  y shape: {y.shape}")

np.random.seed(42)
indices = np.random.permutation(len(X))
split2  = int(0.8 * len(X))
X_train, X_test = X[indices[:split2]], X[indices[split2:]]
y_train, y_test = y[indices[:split2]], y[indices[split2:]]

mean = X_train.mean(axis=0)
std  = X_train.std(axis=0) + 1e-8
X_tr_s = (X_train - mean) / std
X_te_s = (X_test  - mean) / std

variances = X_tr_s.var(axis=0)
top_feats = np.argsort(variances)[::-1][:50]
X_tr = X_tr_s[:, top_feats]
X_te = X_te_s[:, top_feats]


# ════════════════════════════════════════════════════════════
# KNN HELPERS
# ════════════════════════════════════════════════════════════
def knn_binary(X_train, y_train, X_test, k=5):
    preds = []
    for i, pt in enumerate(X_test):
        if i % 100 == 0: print(f"  {i}/{len(X_test)}...", end="\r")
        dists = np.sqrt(((X_train - pt) ** 2).sum(axis=1))
        votes = y_train[np.argsort(dists)[:k]].sum()
        preds.append(1 if votes > k // 2 else 0)
    return np.array(preds)

def knn_multiclass(X_train, y_train, X_test, k=7, n_classes=5):
    preds = []
    for i, pt in enumerate(X_test):
        if i % 100 == 0: print(f"  {i}/{len(X_test)}...", end="\r")
        dists    = np.sqrt(((X_train - pt) ** 2).sum(axis=1))
        k_labels = y_train[np.argsort(dists)[:k]]
        counts   = np.bincount(k_labels.astype(int), minlength=n_classes)
        preds.append(int(np.argmax(counts)))
    return np.array(preds)


# ════════════════════════════════════════════════════════════
# CLASSIFIER 1 — Freshness
# ════════════════════════════════════════════════════════════
print("\n── Classifier 1: Freshness (k=5) ──")
y_pred = knn_binary(X_tr, y_train, X_te, k=5)
acc    = (y_pred == y_test).mean()
tp = ((y_pred==1)&(y_test==1)).sum()
tn = ((y_pred==0)&(y_test==0)).sum()
fp = ((y_pred==1)&(y_test==0)).sum()
fn = ((y_pred==0)&(y_test==1)).sum()
print(f"\n✓ Freshness Accuracy: {acc*100:.2f}%")
print(f"  Precision: {tp/(tp+fp+1e-8):.2f}  Recall: {tp/(tp+fn+1e-8):.2f}")


# ════════════════════════════════════════════════════════════
# CLASSIFIER 2 — Fruit Type (5 classes)
# ════════════════════════════════════════════════════════════
print("\n── Classifier 2: Fruit Type (5 classes, k=7) ──")
yf_pred = knn_multiclass(Xf_tr_s, yf_tr, Xf_te_s, k=7, n_classes=5)
acc_f   = (yf_pred == yf_te).mean()
print(f"\n✓ Fruit Type Accuracy: {acc_f*100:.2f}%")
print("\nPer-fruit accuracy:")
for name, idx2 in FRUIT_LABEL.items():
    mask = yf_te == idx2
    if mask.sum() == 0: continue
    correct = (yf_pred[mask] == yf_te[mask]).sum()
    print(f"  {name:12s}: {correct}/{mask.sum()} = {correct/mask.sum()*100:.1f}%")


# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════
model = {
    "mean":           mean,
    "std":            std,
    "top_feats":      top_feats,
    "k":              5,
    "X_train_fresh":  X_tr,
    "y_train_fresh":  y_train,
    "sf_mean":        sf_mean,
    "sf_std":         sf_std,
    "k_fruit":        7,
    "X_train_fruit":  Xf_tr_s,
    "y_train_fruit":  yf_tr,
    "fruit_names":    FRUIT_NAMES,
}
joblib.dump(model, "fruit_model.pkl")
print(f"\n✓ Saved fruit_model.pkl")
print(f"  Freshness  : {acc*100:.2f}%")
print(f"  Fruit type : {acc_f*100:.2f}%")
