

import os, cv2, random, warnings, traceback
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight

import albumentations as A
from mtcnn import MTCNN

warnings.filterwarnings("ignore")
print(f"TensorFlow  : {tf.__version__}")
print(f"GPU devices : {tf.config.list_physical_devices('GPU')}")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — edit these paths if needed
# ══════════════════════════════════════════════════════════════════════════════
class CFG:
    DATASET_ROOT    = "dataset"              # dataset/real/ and dataset/fake/
    MODEL_SAVE      = "deepfake_model.h5"    # final model used by app.py
    FEATURES_CACHE  = "features_cache.npz"  # saves time on re-runs

    NUM_FRAMES      = 50       # frames sampled per video
    IMG_SIZE        = 299      # Xception native size

    BATCH_SIZE      = 8
    EPOCHS          = 50
    LEARNING_RATE   = 1e-4
    VAL_SPLIT       = 0.15
    TEST_SPLIT      = 0.10
    SEED            = 42

    LSTM_UNITS      = 128
    DROPOUT         = 0.4
    FINETUNE_LAYERS = 40       # unfreeze last N backbone layers in stage-2

    MAX_VIDEOS_PER_CLASS = None  # set e.g. 300 for quick test run
    FACE_CONFIDENCE      = 0.90
    FACE_MARGIN          = 0.20

random.seed(CFG.SEED)
np.random.seed(CFG.SEED)
tf.random.set_seed(CFG.SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  FOCAL LOSS
#  Stops model from predicting only one class (your original bug)
# ══════════════════════════════════════════════════════════════════════════════
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        bce     = keras.backend.binary_crossentropy(y_true, y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha  + (1 - y_true) * (1 - alpha)
        fl      = alpha_t * tf.pow(1.0 - p_t, gamma) * bce
        return tf.reduce_mean(fl)
    loss_fn.__name__ = "focal_loss"
    return loss_fn


# ══════════════════════════════════════════════════════════════════════════════
#  FACE DETECTOR  (MTCNN)
#  WHY: Deepfake artefacts live only in the face region.
#       Feeding the full frame wastes the model on background/hair/neck.
# ══════════════════════════════════════════════════════════════════════════════
print("\nInitialising MTCNN face detector...")
_mtcnn = MTCNN()
print("✓ MTCNN ready")

def detect_and_crop_face(frame_rgb: np.ndarray) -> np.ndarray:
    detections = _mtcnn.detect_faces(frame_rgb)
    H, W = frame_rgb.shape[:2]

    best, best_area = None, 0
    for d in detections:
        if d["confidence"] < CFG.FACE_CONFIDENCE:
            continue
        x, y, w, h = d["box"]
        if w * h > best_area:
            best_area = w * h
            best = (x, y, w, h)

    if best is not None:
        x, y, w, h = best
        mx = int(w * CFG.FACE_MARGIN)
        my = int(h * CFG.FACE_MARGIN)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(W, x + w + mx)
        y2 = min(H, y + h + my)
        face = frame_rgb[y1:y2, x1:x2]
    else:
        # Fallback: centre crop
        side = min(H, W)
        sx, sy = (W - side) // 2, (H - side) // 2
        face = frame_rgb[sy:sy + side, sx:sx + side]

    return cv2.resize(face, (CFG.IMG_SIZE, CFG.IMG_SIZE))


# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION PIPELINE
#  WHY: Forces model to learn face-structure features, not codec fingerprints.
#       JPEG compression simulation is especially important for generalisation.
# ══════════════════════════════════════════════════════════════════════════════
augment_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
    A.ImageCompression(quality_lower=60, quality_upper=95, p=0.5),
    A.Rotate(limit=10, p=0.4),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.CoarseDropout(max_holes=4, max_height=20, max_width=20, p=0.2),
])

def augment_frame(frame_uint8: np.ndarray) -> np.ndarray:
    return augment_pipeline(image=frame_uint8)["image"]


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFORM FRAME SAMPLER
#  WHY: Taking only the first N frames misses most of the video.
#       Deepfake quality varies — uniform sampling catches all artefacts.
# ══════════════════════════════════════════════════════════════════════════════
def extract_frames(video_path: str,
                   num_frames: int = CFG.NUM_FRAMES,
                   augment: bool = False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total   = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            if frames:
                frames.append(frames[-1].copy())
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if augment:
            frame_rgb = augment_frame(frame_rgb)
        face = detect_and_crop_face(frame_rgb)
        frames.append(face.astype(np.float32))

    cap.release()

    if not frames:
        return None

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return np.array(frames[:num_frames])


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADER
#  WHY: Labels come from folder name ONLY — no random labelling bug.
# ══════════════════════════════════════════════════════════════════════════════
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

def collect_video_paths(dataset_root: str, max_per_class=None):
    root      = Path(dataset_root)
    class_map = {"real": 0, "fake": 1}
    paths, labels = [], []

    for cls_name, label in class_map.items():
        cls_dir = root / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(
                f"\n✗ Folder not found: {cls_dir}\n"
                f"  Please create:\n"
                f"    {dataset_root}/real/  ← real .mp4 videos\n"
                f"    {dataset_root}/fake/  ← fake .mp4 videos"
            )
        vids = [p for p in cls_dir.iterdir()
                if p.suffix.lower() in VIDEO_EXTS]
        if max_per_class:
            random.shuffle(vids)
            vids = vids[:max_per_class]

        paths.extend([str(p) for p in vids])
        labels.extend([label] * len(vids))
        print(f"  {cls_name:>6}: {len(vids)} videos")

    print(f"\n  Total: {len(paths)} videos")
    return paths, labels


def balance_dataset(paths, labels):
    """
    WHY: Class imbalance = model predicts only one class.
         Balancing forces it to actually learn the boundary.
    """
    real_idx  = [i for i, l in enumerate(labels) if l == 0]
    fake_idx  = [i for i, l in enumerate(labels) if l == 1]
    min_count = min(len(real_idx), len(fake_idx))

    random.shuffle(real_idx)
    random.shuffle(fake_idx)
    keep = real_idx[:min_count] + fake_idx[:min_count]
    random.shuffle(keep)

    b_paths  = [paths[i]  for i in keep]
    b_labels = [labels[i] for i in keep]
    print(f"  After balancing: {len(b_paths)} videos ({min_count} per class)")
    return b_paths, b_labels


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTOR  (Xception backbone)
#  WHY: Xception's depthwise-separable convolutions are sensitive to local
#       texture anomalies — exactly what GAN face synthesis leaves behind.
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding Xception feature extractor...")
feature_extractor = Xception(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(CFG.IMG_SIZE, CFG.IMG_SIZE, 3)
)
feature_extractor.trainable = False
print(f"✓ Xception ready  (output dim=2048)")


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION WITH DISK CACHE
#  WHY: Saves hours on re-runs — extract once, train many times.
# ══════════════════════════════════════════════════════════════════════════════
def extract_all_features(paths, labels,
                          cache_path=CFG.FEATURES_CACHE,
                          augment=False,
                          force_recompute=False):
    if not force_recompute and Path(cache_path).exists():
        print(f"\n  Loading cached features from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        print(f"  X: {data['X'].shape}  y: {data['y'].shape}")
        return data["X"], data["y"]

    X, y    = [], []
    skipped = 0

    for path, label in tqdm(zip(paths, labels),
                             total=len(paths),
                             desc="Extracting features"):
        frames = extract_frames(path, augment=augment)
        if frames is None:
            skipped += 1
            continue

        frames_pp = preprocess_input(frames.copy())
        feats     = feature_extractor.predict(frames_pp, verbose=0)
        X.append(feats)
        y.append(label)

    print(f"\n  Done: {len(X)} videos  |  Skipped: {skipped}")
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    np.savez_compressed(cache_path, X=X, y=y)
    print(f"  Cached → {cache_path}")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  LSTM MODEL ARCHITECTURE
#  WHY: Mean-pooling destroys temporal order.
#       BiLSTM reads frame sequence forward + backward and learns
#       "artefacts flickering at frame 12, 27, 35 → FAKE"
# ══════════════════════════════════════════════════════════════════════════════
def build_model(num_frames=CFG.NUM_FRAMES,
                feat_dim=2048,
                lstm_units=CFG.LSTM_UNITS,
                dropout=CFG.DROPOUT) -> Model:
    inp = keras.Input(shape=(num_frames, feat_dim), name="frame_features")

    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True,
                    dropout=dropout, recurrent_dropout=0.2),
        name="bilstm_1"
    )(inp)
    x = layers.LayerNormalization()(x)

    x = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, return_sequences=False,
                    dropout=dropout, recurrent_dropout=0.2),
        name="bilstm_2"
    )(x)
    x = layers.LayerNormalization()(x)

    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(dropout)(x)
    x   = layers.Dense(64,  activation="relu")(x)
    x   = layers.Dropout(dropout / 2)(x)
    out = layers.Dense(1, activation="sigmoid", name="fake_prob")(x)

    model = Model(inputs=inp, outputs=out, name="deepfake_bilstm")
    model.summary()
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
def make_splits(X, y):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=CFG.TEST_SPLIT, stratify=y, random_state=CFG.SEED
    )
    val_ratio = CFG.VAL_SPLIT / (1.0 - CFG.TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio, stratify=y_trainval, random_state=CFG.SEED
    )
    print(f"  Train:{len(X_train)}  Val:{len(X_val)}  Test:{len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def plot_history(history, title="Training History"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontweight="bold")

    for ax, (tr, vl, label) in zip(axes, [
        ("accuracy", "val_accuracy", "Accuracy"),
        ("auc",      "val_auc",      "AUC"),
        ("loss",     "val_loss",     "Loss"),
    ]):
        if tr in history.history:
            ax.plot(history.history[tr],  label="Train")
        if vl in history.history:
            ax.plot(history.history[vl],  label="Val", linestyle="--")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("  Saved → training_history.png")


def evaluate(model, X_test, y_test, threshold=0.5):
    print("\n" + "="*50)
    print("  FINAL TEST SET EVALUATION")
    print("="*50)

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    cm  = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real","Fake"],
                yticklabels=["Real","Fake"], ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    axes[1].plot([0,1],[0,1],"k--", alpha=0.4)
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("evaluation.png", dpi=150)
    plt.show()

    print("\n" + classification_report(y_test, y_pred,
                                       target_names=["Real","Fake"]))
    print(f"  ROC-AUC : {auc:.4f}")
    tn, fp, fn, tp = cm.ravel()
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print("  Saved → evaluation.png")
    return auc


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run():
    print("\n" + "█"*55)
    print("  STEP 1/6 — Load & balance dataset")
    print("█"*55)
    paths, labels = collect_video_paths(
        CFG.DATASET_ROOT,
        max_per_class=CFG.MAX_VIDEOS_PER_CLASS
    )
    paths, labels = balance_dataset(paths, labels)

    print("\n" + "█"*55)
    print("  STEP 2/6 — Extract Xception features (with face detection)")
    print("█"*55)
    X, y = extract_all_features(
        paths, labels,
        cache_path=CFG.FEATURES_CACHE,
        augment=False,
        force_recompute=False   # set True to redo even if cache exists
    )

    print("\n" + "█"*55)
    print("  STEP 3/6 — Train / Val / Test split")
    print("█"*55)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    # Class weights (extra safety against imbalance)
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"  Class weights: {class_weights}")

    print("\n" + "█"*55)
    print("  STEP 4/6 — Build BiLSTM model")
    print("█"*55)
    feat_dim = X.shape[-1]
    model    = build_model(num_frames=CFG.NUM_FRAMES, feat_dim=feat_dim)

    schedule = CosineDecayRestarts(
        initial_learning_rate=CFG.LEARNING_RATE,
        first_decay_steps=500,
        t_mul=2.0, m_mul=0.9, alpha=1e-6
    )
    model.compile(
        optimizer=Adam(schedule),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
    )

    print("\n" + "█"*55)
    print("  STEP 5/6 — Stage-1 Training (backbone frozen, 20 epochs)")
    print("█"*55)
    callbacks = [
        ModelCheckpoint(CFG.MODEL_SAVE, monitor="val_auc",
                        mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_auc", patience=8,
                      mode="max", restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=CFG.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )
    plot_history(history, title="Stage-1: BiLSTM Head Training")

    # ── Stage-2: Fine-tune top Xception layers ────────────────────────────
    # Uncomment once Stage-1 reaches ≥ 75% val accuracy
    #
    # print("\n" + "█"*55)
    # print("  STEP 5b — Stage-2 Fine-tuning (top-40 Xception layers)")
    # print("█"*55)
    # feature_extractor.trainable = True
    # for layer in feature_extractor.layers[:-40]:
    #     layer.trainable = False
    # X2, y2 = extract_all_features(
    #     paths, labels,
    #     cache_path="features_cache_ft.npz",
    #     augment=True,
    #     force_recompute=True
    # )
    # X_tr2, X_v2, X_te2, y_tr2, y_v2, y_te2 = make_splits(X2, y2)
    # model.compile(optimizer=Adam(1e-5),
    #               loss=focal_loss(),
    #               metrics=["accuracy", keras.metrics.AUC(name="auc")])
    # history2 = model.fit(
    #     X_tr2, y_tr2,
    #     validation_data=(X_v2, y_v2),
    #     epochs=CFG.EPOCHS,
    #     batch_size=CFG.BATCH_SIZE,
    #     class_weight=get_class_weights(y_tr2),
    #     callbacks=callbacks,
    #     shuffle=True,
    # )
    # plot_history(history2, title="Stage-2: Fine-tuning")

    print("\n" + "█"*55)
    print("  STEP 6/6 — Final Evaluation")
    print("█"*55)
    model.load_weights(CFG.MODEL_SAVE)
    auc = evaluate(model, X_test, y_test)

    print("\n" + "="*55)
    print(f"  ✅ TRAINING COMPLETE!")
    print(f"  Test AUC      : {auc:.4f}")
    print(f"  Model saved   : {CFG.MODEL_SAVE}")
    print(f"  Now run       : python app.py")
    print("="*55 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run()