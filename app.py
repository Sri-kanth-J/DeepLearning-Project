"""
Flask web application for skin disease prediction.
Uses EfficientNetV2S model trained on balanced dataset.

Run:
    pip install flask
    python app.py
Then open http://localhost:5000
"""

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "models" / "checkpoints" / "efficientnetv2s_finetuned.keras"
CLASS_INDICES_PATH = SCRIPT_DIR / "models" / "class_indices.json"

# ---------------------------------------------------------------------------
# Disease metadata (folder-id → display info)
# Folder IDs match the balanced_dataset subdirectory names.
# Update if your dataset uses different IDs/names.
# ---------------------------------------------------------------------------
DISEASE_METADATA = {
    "0":  {"name": "Actinic Keratoses",         "description": "Rough, scaly patches on sun-exposed skin that may become cancerous.",              "urgency": "Medium",  "urgency_color": "#F39C12"},
    "1":  {"name": "Basal Cell Carcinoma",       "description": "The most common type of skin cancer, typically appearing as a shiny bump.",        "urgency": "Medium",  "urgency_color": "#E74C3C"},
    "2":  {"name": "Benign Keratosis",           "description": "Non-cancerous growths that appear as scaly, rough patches.",                       "urgency": "Low",     "urgency_color": "#27AE60"},
    "3":  {"name": "Chickenpox",                 "description": "Viral infection causing itchy, blister-like rash across the body.",                "urgency": "Medium",  "urgency_color": "#E67E22"},
    "4":  {"name": "Cowpox",                     "description": "Viral infection causing localised lesions, typically contracted from animals.",     "urgency": "Medium",  "urgency_color": "#8E44AD"},
    "6":  {"name": "Dermatofibroma",             "description": "Common benign skin tumour that feels like a hard lump under the skin.",            "urgency": "Low",     "urgency_color": "#27AE60"},
    "7":  {"name": "Healthy Skin",               "description": "Normal healthy skin with no signs of disease or infection.",                        "urgency": "None",    "urgency_color": "#2ECC71"},
    "8":  {"name": "Hand Foot Mouth Disease",    "description": "Viral infection causing fever and rash on hands, feet, and inside the mouth.",     "urgency": "Medium",  "urgency_color": "#3498DB"},
    "9":  {"name": "Measles",                    "description": "Highly contagious viral infection causing fever and a characteristic red rash.",    "urgency": "High",    "urgency_color": "#E74C3C"},
    "10": {"name": "Melanocytic Nevi",           "description": "Common benign moles composed of pigment-producing cells (melanocytes).",           "urgency": "Low",     "urgency_color": "#9B59B6"},
    "11": {"name": "Melanoma",                   "description": "Serious skin cancer that develops in pigment-producing cells.",                     "urgency": "High",    "urgency_color": "#C0392B"},
    "12": {"name": "Monkeypox",                  "description": "Viral infection causing fever and a distinctive pustular rash.",                    "urgency": "High",    "urgency_color": "#8E44AD"},
}

# ---------------------------------------------------------------------------
# TensorFlow / model loading (imported lazily to keep startup fast if TF missing)
# ---------------------------------------------------------------------------
model = None
class_map = {}      # int index → folder-name string  e.g. {0: "0", 1: "1", 2: "10", ...}
class_names = []    # ordered list of folder-name strings matching model output indices


def load_model():
    """Load EfficientNetV2S model and class mapping at startup."""
    global model, class_map, class_names

    try:
        import tensorflow as tf
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

        # Suppress GPU warnings on machines without a GPU
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

        if not MODEL_PATH.exists():
            print(f"[WARNING] Model not found at: {MODEL_PATH}", file=sys.stderr)
            print("          Run train.py first, or copy your .keras file here.", file=sys.stderr)
            return

        print(f"Loading model from {MODEL_PATH} ...")
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        print("Model loaded successfully.")

    except ImportError:
        print("[ERROR] TensorFlow is not installed. Run: pip install tensorflow", file=sys.stderr)
        return
    except Exception as exc:
        print(f"[ERROR] Could not load model: {exc}", file=sys.stderr)
        return

    # ------------------------------------------------------------------
    # Load class mapping
    # ------------------------------------------------------------------
    if CLASS_INDICES_PATH.exists():
        with open(CLASS_INDICES_PATH) as f:
            raw = json.load(f)          # {"0": "0", "1": "1", "2": "10", ...}
        class_map = {int(k): v for k, v in raw.items()}
    else:
        # Derive mapping by sorting folder names alphabetically (same as image_dataset_from_directory)
        train_dir = SCRIPT_DIR / "balanced_dataset" / "train"
        if train_dir.exists():
            folders = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
        else:
            folders = [str(i) for i in range(model.output_shape[-1])]
        class_map = {i: name for i, name in enumerate(folders)}
        print("[INFO] class_indices.json not found — derived class order from directory listing.")

    class_names = [class_map[i] for i in range(len(class_map))]
    print(f"Classes ({len(class_names)}): {class_names}")


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Resize to 224×224 and convert to float32 array with batch dimension.
    EfficientNetV2S includes its own preprocessing — do NOT rescale to [0,1].
    """
    img = pil_image.convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)           # shape (224, 224, 3), values [0, 255]
    return np.expand_dims(arr, axis=0)              # shape (1, 224, 224, 3)


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit
ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    num_classes = len(class_names) if class_names else "N/A"
    model_loaded = model is not None
    model_size_mb = round(MODEL_PATH.stat().st_size / (1024 * 1024), 1) if MODEL_PATH.exists() else None
    return render_template(
        "index.html",
        model_loaded=model_loaded,
        num_classes=num_classes,
        model_size_mb=model_size_mb,
        model_name="EfficientNetV2S",
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded. Check server logs."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Upload a JPG, PNG, BMP, or WebP image."}), 400

    try:
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))
        img_array = preprocess_image(pil_img)
    except Exception as exc:
        return jsonify({"error": f"Could not read image: {exc}"}), 400

    try:
        import tensorflow as tf
        probs = model.predict(img_array, verbose=0)[0]          # shape (num_classes,)
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    # Build sorted results
    indices = np.argsort(probs)[::-1]
    results = []
    for rank, idx in enumerate(indices):
        folder_id = class_names[idx] if idx < len(class_names) else str(idx)
        meta = DISEASE_METADATA.get(folder_id, {"name": f"Class {folder_id}", "description": "", "urgency": "Unknown", "urgency_color": "#999"})
        results.append({
            "rank":          rank + 1,
            "folder_id":     folder_id,
            "display_name":  meta["name"],
            "description":   meta["description"],
            "urgency":       meta["urgency"],
            "urgency_color": meta["urgency_color"],
            "confidence":    float(probs[idx]),
            "confidence_pct": f"{probs[idx]:.1%}",
        })

    return jsonify({
        "prediction":  results[0],
        "top3":        results[:3],
        "all_classes": results,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
