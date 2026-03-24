

from flask import Flask, request, jsonify, render_template
import os, cv2, tempfile, traceback
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
import tensorflow as tf


# ── Focal loss — must match what was used during training ──────────────────
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        bce     = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha  + (1 - y_true) * (1 - alpha)
        fl      = alpha_t * tf.pow(1.0 - p_t, gamma) * bce
        return tf.reduce_mean(fl)
    loss_fn.__name__ = "focal_loss"
    return loss_fn


# ─── Config ────────────────────────────────────────────────────────────────
IMG_SIZE        = 299
NUM_FRAMES      = 50
MODEL_PATH      = "deepfake_model.h5"
FACE_CONFIDENCE = 0.90
FACE_MARGIN     = 0.20
ALLOWED_EXTS    = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

app = Flask(__name__)


# ─── Load everything at startup ────────────────────────────────────────────
print("\n" + "="*55)
print("  DEEPFAKE DETECTION SERVER — Starting Up")
print("="*55)

# 1. Xception feature extractor
print("[1/3] Loading Xception feature extractor…")
feature_extractor = Xception(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
feature_extractor.trainable = False
print("      ✓ Xception ready  (output dim=2048)")

# 2. Trained LSTM classifier
print("[2/3] Loading trained LSTM model…")
try:
    deepfake_model = load_model(
    MODEL_PATH,
    custom_objects={"focal_loss": focal_loss(2.0, 0.25)}
)
    print(f"      ✓ Model loaded  ←  {MODEL_PATH}")
except Exception as e:
    deepfake_model = None
    print(f"      ✗ Model NOT loaded: {e}")
    print(f"        → Run deepfake_train.py first to generate {MODEL_PATH}")

# 3. MTCNN face detector
print("[3/3] Initialising MTCNN face detector…")
_mtcnn = MTCNN()
print("      ✓ MTCNN ready")
print("="*55 + "\n")


# ══════════════════════════════════════════════════════════════
#  FACE DETECTION HELPER
# ══════════════════════════════════════════════════════════════
def detect_and_crop_face(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Finds the largest high-confidence face and returns a square crop
    resized to (IMG_SIZE, IMG_SIZE). Centre-crops if no face found.
    """
    detections = _mtcnn.detect_faces(frame_rgb)
    H, W = frame_rgb.shape[:2]

    best, best_area = None, 0
    for d in detections:
        if d["confidence"] < FACE_CONFIDENCE:
            continue
        x, y, w, h = d["box"]
        if w * h > best_area:
            best_area = w * h
            best = (x, y, w, h)

    if best is not None:
        x, y, w, h = best
        mx = int(w * FACE_MARGIN)
        my = int(h * FACE_MARGIN)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(W, x + w + mx)
        y2 = min(H, y + h + my)
        face = frame_rgb[y1:y2, x1:x2]
    else:
        side = min(H, W)
        sx, sy = (W - side) // 2, (H - side) // 2
        face = frame_rgb[sy:sy + side, sx:sx + side]

    return cv2.resize(face, (IMG_SIZE, IMG_SIZE))


# ══════════════════════════════════════════════════════════════
#  UNIFORM FRAME SAMPLER
# ══════════════════════════════════════════════════════════════
def extract_frames(video_path: str,
                   num_frames: int = NUM_FRAMES):
    """
    Uniformly sample `num_frames` face-cropped frames from the full video.
    Returns float32 ndarray of shape (num_frames, IMG_SIZE, IMG_SIZE, 3).
    Returns None if video cannot be opened.
    """
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
        face      = detect_and_crop_face(frame_rgb)
        frames.append(face.astype(np.float32))

    cap.release()

    if not frames:
        return None

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return np.array(frames[:num_frames])


# ══════════════════════════════════════════════════════════════
#  MAIN PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════════
def predict_video(video_path: str):
    """
    Pipeline:
      video  →  50 face crops  →  Xception (2048-d each)
             →  BiLSTM  →  sigmoid score  →  label + confidence
    """
    if deepfake_model is None:
        raise RuntimeError(
            f"Model not loaded. Run deepfake_train.py to create {MODEL_PATH}"
        )

    print(f"  [1/3] Extracting {NUM_FRAMES} frames…")
    frames = extract_frames(video_path)
    if frames is None:
        raise ValueError("Could not read any frames from this video.")
    print(f"        shape = {frames.shape}")

    print("  [2/3] Running Xception feature extraction…")
    frames_pp = preprocess_input(frames.copy())
    features  = feature_extractor.predict(frames_pp, verbose=0, batch_size=8)
    print(f"        features shape = {features.shape}")

    print("  [3/3] Running BiLSTM classifier…")
    seq   = np.expand_dims(features, axis=0)        # (1, 50, 2048)
    score = float(deepfake_model.predict(seq, verbose=0)[0][0])

    prediction = "fake" if score > 0.40 else "real"
    confidence = score if prediction == "fake" else 1.0 - score
    print(f"        score={score:.4f}  →  {prediction.upper()}  conf={confidence:.2%}")
    return prediction, round(confidence, 4)


# ══════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        return f"<h3>index.html not found: {e}</h3>", 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts:  POST  multipart/form-data   field name: 'video'
    Returns:  JSON
      {
        "prediction" : "fake" | "real",
        "confidence" : 0.91,
        "label"      : "⚠ DEEPFAKE (91%)"
      }
    """
    temp_path = None
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video field in request"}), 400

        video = request.files["video"]
        if not video.filename:
            return jsonify({"error": "Empty filename"}), 400

        ext = os.path.splitext(video.filename.lower())[1]
        if ext not in ALLOWED_EXTS:
            return jsonify({
                "error": f"Unsupported format '{ext}'. "
                         f"Allowed: {', '.join(sorted(ALLOWED_EXTS))}"
            }), 400

        # Save to a uniquely-named temp file (thread-safe)
        temp_path = os.path.join(
            tempfile.gettempdir(),
            f"dfd_{os.urandom(6).hex()}{ext}"
        )
        video.save(temp_path)
        print(f"\n■ Received: {video.filename}")

        prediction, confidence = predict_video(temp_path)
        pct = int(confidence * 100)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "label": f"{'⚠ DEEPFAKE' if prediction == 'fake' else '✓ REAL'} ({pct}%)"
        }), 200

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503   # model not loaded
    except ValueError as e:
        return jsonify({"error": str(e)}), 422   # bad video
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {e}"}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/health", methods=["GET"])
def health():
    ok = deepfake_model is not None
    return jsonify({
        "status"      : "healthy" if ok else "degraded",
        "model_loaded": ok,
        "model_path"  : MODEL_PATH,
        "num_frames"  : NUM_FRAMES,
        "img_size"    : IMG_SIZE,
    }), 200 if ok else 206


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Server running at  http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
