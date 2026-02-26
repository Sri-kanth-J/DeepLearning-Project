# Project Structure

```
Skin-disease-prediction/
│
├── app.py                  # Flask web app for skin lesion prediction
├── train.py                # Two-phase training script (EfficientNetV2S)
├── test.py                 # Evaluate model on test set
├── download_dataset.py     # Download dataset from Hugging Face
├── simple_rebalance.py     # Balance dataset across classes
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── structure.md            # This file
│
├── models/
│   ├── checkpoints/
│   │   ├── efficientnetv2s_transfer.keras    # Phase 1 checkpoint (92 MB)
│   │   └── efficientnetv2s_finetuned.keras   # Final model - 81.6% accuracy (198 MB)
│   ├── class_indices.json          # Maps model index → folder name
│   ├── class_names.json            # Maps folder number → disease name
│   ├── test_metrics.json           # Last test run results
│   └── balanced_confusion_matrix.png
│
├── templates/
│   └── index.html          # Web app frontend
│
├── static/
│   ├── css/
│   │   └── style.css       # Web app styling
│   └── js/
│       └── main.js         # Upload and prediction handling
│
├── dataset/                # Raw dataset from Hugging Face (not in git)
│   ├── train/
│   ├── val/
│   └── test/
│
├── balanced_dataset/       # Created by simple_rebalance.py (not in git)
│   ├── train/
│   │   ├── 1/              # Basal cell carcinoma
│   │   ├── 2/              # Benign keratosis
│   │   ├── 6/              # Dermatofibroma
│   │   ├── 7/              # Healthy
│   │   ├── 9/              # Measles
│   │   ├── 10/             # Melanocytic nevi
│   │   └── 11/             # Melanoma
│   ├── val/
│   └── test/
│
└── venv/                   # Python virtual environment (not in git)
```

## Class Mapping

The folder numbers correspond to disease names from the original 14-class dataset:

| Folder | Disease Name |
|--------|--------------|
| 0 | Actinic keratoses |
| 1 | Basal cell carcinoma |
| 2 | Benign keratosis |
| 3 | Chickenpox |
| 4 | Cowpox |
| 5 | Dermatofibroma |
| 6 | Healthy |
| 7 | HFMD |
| 8 | Measles |
| 9 | Melanocytic nevi |
| 10 | Melanoma |
| 11 | Monkeypox |
| 12 | Squamous cell carcinoma |
| 13 | Vascular lesions |

After rebalancing, we use 7 classes with sufficient samples: 1, 2, 6, 7, 9, 10, 11.

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | EfficientNetV2S training with cosine LR warmup, label smoothing, class weights |
| `test.py` | Generates accuracy report, confusion matrix, per-class F1 scores |
| `app.py` | Flask server with drag-drop image upload and Chart.js visualization |
| `simple_rebalance.py` | Copies images to create balanced train/val/test splits |
| `download_dataset.py` | Downloads skin lesion dataset from Hugging Face |
