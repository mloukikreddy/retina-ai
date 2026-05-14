import os
import cv2
import numpy as np
import joblib
import pandas as pd

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from lightgbm import LGBMClassifier
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────
IMG_SIZE = 224

FUNDUS_DIR = "dataset/eye fundus (2)/eye fundus"
OCT_DIR    = "dataset/OCT/OCT_NEW"

FUNDUS_CSV = "dataset/EYE FUNDUS.csv"
OCT_CSV    = "dataset/OCT.csv"


# Augmented copies per original image
# 410 pairs x (1 original + 8 augmented) = ~3690 samples
AUG_FACTOR = 8

DR_MAP = {
    "0":    0,   # No DR
    "NPDR": 1,   # Non-Proliferative DR
    "PDR":  2,   # Proliferative DR
}

CLASS_NAMES = {0: "No DR", 1: "NPDR", 2: "PDR"}


# ── AUGMENTATION ──────────────────────────────────────────────
def augment_image(img):
    """
    Randomly applies:
      1. Horizontal flip
      2. Vertical flip
      3. Rotation ±20 degrees
      4. Brightness + contrast shift
      5. Random zoom (crop 80-100% then resize)
      6. Gaussian noise
    Input/output: float32 numpy array in [0, 1], shape (H, W, 3)
    """
    aug = img.copy()

    # 1. Horizontal flip
    if np.random.rand() > 0.5:
        aug = cv2.flip(aug, 1)

    # 2. Vertical flip
    if np.random.rand() > 0.5:
        aug = cv2.flip(aug, 0)

    # 3. Random rotation ±20°
    angle = np.random.uniform(-20, 20)
    h, w  = aug.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    aug   = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # 4. Brightness + contrast
    alpha = np.random.uniform(0.8, 1.2)   # contrast multiplier
    beta  = np.random.uniform(-0.1, 0.1)  # brightness offset
    aug   = np.clip(aug * alpha + beta, 0.0, 1.0)

    # 5. Random zoom
    zoom  = np.random.uniform(0.80, 1.0)
    ch    = int(h * zoom)
    cw    = int(w * zoom)
    top   = np.random.randint(0, h - ch + 1)
    left  = np.random.randint(0, w - cw + 1)
    aug   = aug[top:top + ch, left:left + cw]
    aug   = cv2.resize(aug, (IMG_SIZE, IMG_SIZE))

    # 6. Gaussian noise
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.02, aug.shape).astype("float32")
        aug   = np.clip(aug + noise, 0.0, 1.0)

    return aug


def augment_pair(f_img, o_img, n=AUG_FACTOR):
    """Generate n augmented copies of a (fundus, oct) image pair."""
    f_list, o_list = [], []
    for _ in range(n):
        f_list.append(augment_image(f_img))
        o_list.append(augment_image(o_img))
    return f_list, o_list


# ── IMAGE HELPERS ─────────────────────────────────────────────
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype("float32") / 255.0


def find_image(base_dir, name):
    """Search base_dir and one level of subfolders for name.*"""
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        direct = os.path.join(base_dir, name + ext)
        if os.path.exists(direct):
            return direct
        for sub in os.listdir(base_dir):
            sub_path = os.path.join(base_dir, sub)
            if os.path.isdir(sub_path):
                candidate = os.path.join(sub_path, name + ext)
                if os.path.exists(candidate):
                    return candidate
    return None


# ── LOAD CSVs ─────────────────────────────────────────────────
print("=" * 55)
print("  RetinaAI — Training with Data Augmentation")
print("=" * 55)
print("\nLoading CSVs...")

fundus_df = pd.read_csv(FUNDUS_CSV)
oct_df    = pd.read_csv(OCT_CSV)

fundus_df = fundus_df[fundus_df["DR"] != "-"].reset_index(drop=True)
oct_df    = oct_df[oct_df["DR"]    != "-"].reset_index(drop=True)

fundus_df["label"] = fundus_df["DR"].astype(str).map(DR_MAP)
oct_df["label"]    = oct_df["DR"].astype(str).map(DR_MAP)

fundus_df = fundus_df.dropna(subset=["label"]).reset_index(drop=True)
oct_df    = oct_df.dropna(subset=["label"]).reset_index(drop=True)

fundus_df["label"] = fundus_df["label"].astype(int)
oct_df["label"]    = oct_df["label"].astype(int)

print(f"Fundus labelled rows : {len(fundus_df)}")
print(f"OCT    labelled rows : {len(oct_df)}")


# ── LOAD IMAGES ───────────────────────────────────────────────
print("\nLoading fundus images...")
Xf_raw, yf_raw = [], []

for _, row in tqdm(fundus_df.iterrows(), total=len(fundus_df)):
    path = find_image(FUNDUS_DIR, str(row["Name"]))
    if path is None:
        continue
    img = load_image(path)
    if img is None:
        continue
    Xf_raw.append(img)
    yf_raw.append(row["label"])

print(f"Fundus images loaded : {len(Xf_raw)}")

print("\nLoading OCT images...")
Xo_raw, yo_raw = [], []

for _, row in tqdm(oct_df.iterrows(), total=len(oct_df)):
    path = find_image(OCT_DIR, str(row["Name"]))
    if path is None:
        continue
    img = load_image(path)
    if img is None:
        continue
    Xo_raw.append(img)
    yo_raw.append(row["label"])

print(f"OCT images loaded    : {len(Xo_raw)}")

# Pair by min length
min_total = min(len(Xf_raw), len(Xo_raw))
Xf_raw    = Xf_raw[:min_total]
Xo_raw    = Xo_raw[:min_total]
y_raw     = yf_raw[:min_total]

print(f"Paired samples       : {min_total}")

if min_total == 0:
    print("\n❌ ERROR: No images loaded. Check your folder paths.")
    exit(1)


# ── AUGMENTATION ──────────────────────────────────────────────
print(f"\nApplying augmentation  : x{AUG_FACTOR} per image")
print(f"Expected total samples : ~{min_total * (AUG_FACTOR + 1)}")

Xf_all, Xo_all, y_all = [], [], []

for i in tqdm(range(min_total)):
    f_img = Xf_raw[i]
    o_img = Xo_raw[i]
    label = y_raw[i]

    # Keep original
    Xf_all.append(f_img)
    Xo_all.append(o_img)
    y_all.append(label)

    # Augmented copies
    f_augs, o_augs = augment_pair(f_img, o_img, n=AUG_FACTOR)
    for fa, oa in zip(f_augs, o_augs):
        Xf_all.append(fa)
        Xo_all.append(oa)
        y_all.append(label)

Xf_all = np.array(Xf_all, dtype="float32")
Xo_all = np.array(Xo_all, dtype="float32")
y_all  = np.array(y_all)

print(f"\nTotal samples after augmentation : {len(y_all)}")
print("Class distribution:")
unique, counts = np.unique(y_all, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u} ({CLASS_NAMES[u]:6s}) : {c} samples")


# ── BUILD FEATURE EXTRACTORS ──────────────────────────────────
print("\nBuilding feature extractor models...")

fundus_base  = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
oct_base     = DenseNet121(weights="imagenet",    include_top=False, input_shape=(224, 224, 3))

fundus_model = Model(fundus_base.input, GlobalAveragePooling2D()(fundus_base.output))
oct_model    = Model(oct_base.input,   GlobalAveragePooling2D()(oct_base.output))


# ── EXTRACT FEATURES ──────────────────────────────────────────
print("\nExtracting fundus features (DenseNet121)...")
f_feat = fundus_model.predict(Xf_all, batch_size=16, verbose=1)

print("\nExtracting OCT features (DenseNet121)...")
o_feat = oct_model.predict(Xo_all, batch_size=16, verbose=1)

print(f"\nFundus feature shape : {f_feat.shape}")
print(f"OCT    feature shape : {o_feat.shape}")


# ── FUSE + SCALE ──────────────────────────────────────────────
print("\nFusing and scaling features...")
X        = np.concatenate([f_feat, o_feat], axis=1)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Fused + scaled shape : {X_scaled.shape}")


# ── TRAIN / TEST SPLIT ────────────────────────────────────────
print("\nSplitting 80 / 20 train-test...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all
)
print(f"Train : {len(y_train)} samples")
print(f"Test  : {len(y_test)}  samples")


# ── TRAIN LIGHTGBM ────────────────────────────────────────────
print("\nTraining LightGBM...")

lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=8,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",    # handles class imbalance automatically
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

lgbm.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
)


# ── EVALUATE ──────────────────────────────────────────────────
print("\n── Evaluation on Test Set ──────────────────────")
y_pred = lgbm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(classification_report(
    y_test, y_pred,
    target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
))
print("Confusion Matrix:")
print(cm)
print(f"Train accuracy : {lgbm.score(X_train, y_train) * 100:.2f}%")
print(f"Test  accuracy : {acc * 100:.2f}%")
print("─" * 50)


# ── SAVE MODELS ───────────────────────────────────────────────
print("\nSaving models...")
os.makedirs("models", exist_ok=True)

fundus_model.save("models/fundus_model.h5")
oct_model.save("models/oct_model.h5")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(lgbm,   "models/lgbm_model.pkl")

print("\n✅ Training complete! Models saved to models/")
print("   models/fundus_model.h5")
print("   models/oct_model.h5")
print("   models/scaler.pkl")
print("   models/lgbm_model.pkl")
print("\nNow run:  python app.py")