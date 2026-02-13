"""
================================================================================
QUICK START GUIDE - SKIN DISEASE CLASSIFICATION
================================================================================

Step-by-step instructions to get started with the project.

All commands assume you're in the project root directory: d:\Code\DemoDeepLee\

================================================================================
"""

# ============================================================================
# STEP 1: INSTALL DEPENDENCIES (First Time Only)
# ============================================================================

"""
Open Command Prompt and run:

    cd d:\Code\DemoDeepLee
    pip install -r requirements.txt

This will install:
    - tensorflow (deep learning framework)
    - keras (neural network API)
    - numpy (numerical computing)
    - pandas (data manipulation)
    - opencv-python (image processing)
    - scikit-learn (machine learning utilities)
    - matplotlib & seaborn (visualization)
    - streamlit (interactive dashboard)
    - and more...

Expected time: 5-15 minutes
Disk space: ~2-3 GB (including all dependencies)

✓ Installation is complete when you see "Successfully installed..."
"""

# ============================================================================
# STEP 2: TRAIN THE MODEL
# ============================================================================

"""
COMMAND:
    python train.py

WHAT IT DOES:
    1. Loads all images from MLDemoProj/IMG_CLASSES/
    2. Preprocesses images (resize, normalize)
    3. Computes class weights for imbalanced data
    4. Builds MobileNet model (MobileNetV2 only - optimized for speed)
    5. Trains the model with aggressive EarlyStopping (patience=3)
    6. Saves trained model to models/skin_disease_mobilenet_model.h5
    7. Creates training visualization plots

EXPECTED OUTPUT:
    ✓ Found 10 disease classes
    ✓ Data loaded: Total images, Array shape
    ✓ Class weights computed with weight ranges
    ✓ Model architecture summary
    ✓ Training starts (epoch 1/50, batch 1/X, etc.)
    ✓ Validation loss monitored
    ✓ EarlyStopping may trigger before epoch 50
    ✓ Model saved
    ✓ Training history plot generated

EXPECTED TIME:
    - First run: 30-60 minutes (downloading pre-trained weights + training)
    - Subsequent runs: Same time (weights cached)
    - Varies based on total image count

FILES CREATED:
    ✓ models/skin_disease_mobilenet_model.h5 (~45 MB - smaller than ensemble)
    ✓ models/training_history.png

TIPS:
    - Keep terminal open, don't interrupt training
    - Monitor GPU memory if using GPU
    - If out of memory, reduce BATCH_SIZE in train.py
"""

# ============================================================================
# STEP 3: EVALUATE THE MODEL
# ============================================================================

"""
COMMAND:
    python test.py

WHAT IT DOES:
    1. Loads trained model from models/
    2. Creates test set (20% of data)
    3. Evaluates model performance
    4. Computes precision, recall, F1-score
    5. Generates confusion matrix
    6. Creates performance visualization plots
    7. Prints detailed classification report

EXPECTED OUTPUT:
    ✓ Model loaded from models/skin_disease_mobilenet_model.h5
    ✓ Test data created with distribution info
    ✓ Evaluation metrics: Loss, Accuracy
    ✓ Per-class metrics for each disease
    ✓ Classification report with detailed scores

FILES CREATED:
    ✓ models/confusion_matrix.png
    ✓ models/per_class_metrics.png
    ✓ models/confidence_distribution.png

INTERPRETATION:
    - Accuracy > 0.85: Good model
    - Accuracy > 0.90: Excellent model
    - Low F1 for class X: Difficult class (need more data or samples)
"""

# ============================================================================
# STEP 4: INTERACTIVE PREDICTIONS (STREAMLIT DASHBOARD)
# ============================================================================

"""
COMMAND:
    streamlit run app.py

WHAT IT DOES:
    Opens interactive web dashboard at http://localhost:8501 with:
    
    1. 🔬 Make Prediction
       - Upload custom image
       - Select from dataset
       - Get disease prediction
       - View confidence scores
       - See prediction distribution
    
    2. 📈 Model Information
       - MobileNet architecture explanation (single backbone)
       - Why this architecture works
       - Class imbalance handling
       - Overfitting prevention techniques
    
    3. 📋 Disease Reference
       - Information about each disease
       - Descriptions and types
       - Educational reference

EXPECTED OUTPUT:
    ✓ Streamlit server starts
    ✓ Local URL: http://localhost:8501
    ✓ Interactive web interface opens
    ✓ Upload/select image → Get prediction

TO STOP:
    Press Ctrl+C in terminal

BROWSER:
    - Chrome, Firefox, Edge all supported
    - No additional installation needed
    - Runs entirely in browser
"""

# ============================================================================
# COMPLETE WORKFLOW EXAMPLE
# ============================================================================

"""
Here's a complete example using all three scripts:

1. PREPARE
   - Ensure MLDemoProj/IMG_CLASSES/ has disease folders with images

2. INSTALL DEPENDENCIES (once)
   cd d:\Code\DemoDeepLee
   pip install -r requirements.txt

3. TRAIN MODEL (first time takes 30-60 mins)
   python train.py
   → Creates: models/skin_disease_mobilenet_model.h5

4. EVALUATE MODEL
   python test.py
   → Creates: confusion_matrix.png, per_class_metrics.png, etc.

5. MAKE PREDICTIONS (interactive)
   streamlit run app.py
   → Opens: http://localhost:8501
   → Upload image → See prediction

6. REVIEW RESULTS
   Open generated plots in models/ folder
   Check classification report
   Analyze confusion matrix

TOTAL TIME: 1-2 hours (including all steps)
"""

# ============================================================================
# ADVANCED USAGE
# ============================================================================

"""
Import utilities in your own script:

    from utils.data_utils import load_dataset_from_directory, plot_training_curves
    
    # Load data
    images, labels, classes = load_dataset_from_directory('MLDemoProj/IMG_CLASSES')
    
    # Load model
    from tensorflow import keras
    model = keras.models.load_model('models/skin_disease_mobilenet_model.h5')
    
    # Make prediction
    prediction = model.predict(images[:1])

"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
ERROR: "No module named 'tensorflow'"
SOLUTION: Run: pip install tensorflow

ERROR: "No images found"
SOLUTION: Check path is: MLDemoProj/IMG_CLASSES/[disease]/[images]

ERROR: "Model not found at models/..."
SOLUTION: Run train.py first to create the model

ERROR: "CUDA out of memory"
SOLUTION: Reduce BATCH_SIZE in train.py (e.g., 32 → 16 or 8)

ERROR: "Streamlit not found"
SOLUTION: Run: pip install streamlit

ERROR: "Port 8501 already in use"
SOLUTION: Run: streamlit run app.py --server.port 8502

"""

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

"""
d:\Code\DemoDeepLee\
├── train.py                                    ← Run this first
├── test.py                                     ← Run this second
├── app.py                                      ← Run this third
├── requirements.txt                            ← Dependencies list
├── README.md                                   ← Full documentation
├── QUICKSTART.py                               ← This file
│
├── MLDemoProj/IMG_CLASSES/                     ← Dataset location
│   ├── 1. Eczema 1677/
│   ├── 2. Melanoma 3.5k/
│   ├── ... (8 more diseases)
│   └── 10. Warts Molluscum 2103/
│
├── models/                                     ← Created after training
│   ├── skin_disease_mobilenet_model.h5
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── per_class_metrics.png
│   └── confidence_distribution.png
│
└── utils/                                      ← Utility functions
    ├── __init__.py
    └── data_utils.py
"""

# ============================================================================
# KEY PARAMETERS YOU CAN MODIFY
# ============================================================================

"""
In train.py (around line 50):

    IMG_HEIGHT = 224          # ← Image height
    IMG_WIDTH = 224           # ← Image width
    BATCH_SIZE = 32           # ← Samples per batch (↓ if out of memory)
    EPOCHS = 50               # ← Max training iterations
    VALIDATION_SPLIT = 0.2    # ← 20% for validation

In train.py (line ~450):

    early_stop = EarlyStopping(
        patience=5,           # ← Stop after 5 epochs no improvement
        ...
    )

To modify these:
1. Open train.py in editor
2. Change the value
3. Run python train.py again
"""

# ============================================================================
# WHAT EACH SCRIPT DOES
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────┐
│ train.py                                                        │
├─────────────────────────────────────────────────────────────────┤
│ Purpose: Train the MobileNet model (fast & efficient)           │
│                                                                  │
│ Input:   MLDemoProj/IMG_CLASSES/*.jpg                          │
│ Output:  models/skin_disease_mobilenet_model.h5                │
│                                                                  │
│ Steps:                                                          │
│  1. Load images from disk                                       │
│  2. Normalize pixel values                                      │
│  3. Compute class weights                                       │
│  4. Build MobileNetV2 model (single backbone for speed)       │
│  5. Compile with Adam optimizer                                │
│  6. Train with EarlyStopping                                   │
│  7. Save trained model                                         │
│  8. Plot training history                                      │
│                                                                  │
│ Time:    30-60 minutes (first run)                             │
│ Output:  1 model file + 1 plot image                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ test.py                                                         │
├─────────────────────────────────────────────────────────────────┤
│ Purpose: Evaluate model on test data                            │
│                                                                  │
│ Input:   models/skin_disease_mobilenet_model.h5                │
│          MLDemoProj/IMG_CLASSES/*.jpg (test set)               │
│ Output:  Metrics + 3 visualization plots                       │
│                                                                  │
│ Steps:                                                          │
│  1. Load trained model                                         │
│  2. Split data: 80% train, 20% test                           │
│  3. Evaluate on test set                                       │
│  4. Compute per-class metrics                                  │
│  5. Generate confusion matrix                                  │
│  6. Create visualization plots                                 │
│                                                                  │
│ Time:    5-10 minutes                                          │
│ Output:  3 plot images + console report                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ app.py                                                          │
├─────────────────────────────────────────────────────────────────┤
│ Purpose: Interactive web dashboard                              │
│                                                                  │
│ Input:   models/skin_disease_mobilenet_model.h5                │
│          User uploads image in browser                          │
│ Output:  Web page with predictions                             │
│                                                                  │
│ Features:                                                       │
│  1. Make predictions on custom images                          │
│  2. View confidence for all 10 classes                         │
│  3. Model information & explanation                            │
│  4. Disease reference guide                                    │
│  5. Interactive visualizations                                 │
│                                                                  │
│ Access:  http://localhost:8501                                 │
│ Platform: Browser-based (no installation needed)               │
└─────────────────────────────────────────────────────────────────┘
"""

# ============================================================================
# COMMON QUESTIONS
# ============================================================================

"""
Q: Can I use my own images?
A: Yes! Either upload in Streamlit app OR add to MLDemoProj/IMG_CLASSES/
   Organize as: MLDemoProj/IMG_CLASSES/[disease_name]/image.jpg

Q: How do I improve model accuracy?
A: 1. Add more images (especially rare classes)
   2. Data augmentation (rotate, flip, zoom)
   3. Fine-tune backbone weights
   4. More aggressive optimization (faster learning rates, smaller models)

Q: Does it need GPU?
A: No! Works on CPU (slower but works). GPU is optional for faster training.

Q: How big is the model?
A: ~90 MB (can compress to ~20 MB with quantization)

Q: Can I deploy this?
A: Yes! Export to ONNX, TensorFlow Lite, or containerize with Docker.

Q: Is this medical diagnostic tool?
A: NO! For research/education only. Always consult dermatologist.

Q: How long does training take?
A: 30-60 minutes on CPU (depends on image count & computer speed)
   5-10 minutes on GPU (NVIDIA CUDA)
"""

# ============================================================================
# NEXT STEPS
# ============================================================================

"""
After completing the basic workflow:

1. IMPROVE MODEL
   - Collect more skin disease images
   - Implement data augmentation
   - Fine-tune hyperparameters

2. DEPLOY
   - Export model to mobile format (TensorFlow Lite)
   - Create REST API with Flask/FastAPI
   - Build mobile app (or web app)

3. EXPAND
   - Add more disease classes
   - Integrate with dermatologist feedback
   - Build clinical validation study

4. OPTIMIZE
   - Quantize model for 4× smaller size
   - Optimize inference speed
   - Deploy on edge devices

5. DOCUMENT
   - Write technical paper
   - Create tutorial blog posts
   - Share on GitHub/Kaggle
"""

# ============================================================================
# SUPPORT & RESOURCES
# ============================================================================

"""
📚 DOCUMENTATION
  - README.md: Full project documentation
  - Code comments: Detailed explanations in each file
  - Streamlit app: Built-in model information tab

🎓 LEARNING RESOURCES
  - TensorFlow documentation: https://tensorflow.org
  - Keras API: https://keras.io
  - Deep Learning course: https://cs231n.github.io

🐛 TROUBLESHOOTING
  1. Check error message carefully
  2. Verify file paths are correct
  3. Ensure all dependencies installed
  4. Check available disk space
  5. Try reducing BATCH_SIZE

💬 GETTING HELP
  - Read comments in source code
  - Check README.md troubleshooting section
  - Search TensorFlow documentation
  - Try Stack Overflow
"""

print(__doc__)
