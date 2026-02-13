# 🏥 Skin Disease Classification - MobileNet Deep Learning Model

A complete machine learning pipeline for classifying 10 different skin diseases using an optimized **MobileNetV2** deep learning model.

## 📋 Overview

This project implements a state-of-the-art skin disease classification system with:

- ✅ **MobileNet Architecture**: Lightweight, efficient model optimized for speed
- ✅ **Fast Training**: 7-9 hours on i5 8th gen + 16GB RAM (vs 18-25 hours ensemble)
- ✅ **Memory Efficient**: Uses data generators to handle 23k images without memory errors
- ✅ **Class Imbalance Handling**: Automatic class weight computation for 1.25k-3.5k sample ranges
- ✅ **Transfer Learning**: Pre-trained ImageNet weights for faster training
- ✅ **Early Stopping**: Aggressive patience=3 for faster training while preventing overfitting
- ✅ **Interactive Dashboard**: Streamlit app for predictions and visualization
- ✅ **Comprehensive Testing**: Detailed evaluation metrics and confusion matrices

## 🎯 Project Structure

```
DemoDeepLee/
├── train.py                          # Main training script
├── test.py                           # Model evaluation script
├── app.py                            # Streamlit interactive dashboard
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
│
├── MLDemoProj/IMG_CLASSES/           # Dataset directory
│   ├── 1. Eczema 1677/
│   ├── 2. Melanoma 3.5k/
│   ├── 3. Atopic Dermatitis - 1.25k/
│   ├── 4. Basal Cell Carcinoma (BCC) 3323/
│   ├── 5. Melanocytic Nevi (NV) - 3.5k/
│   ├── 6. Benign Keratosis-like Lesions (BKL) 2624/
│   ├── 7. Psoriasis pictures Lichen Planus - 2k/
│   ├── 8. Seborrheic Keratoses - 1.8k/
│   ├── 9. Tinea Ringworm Candidiasis - 1.7k/
│   └── 10. Warts Molluscum - 2103/
│mobilenet
├── models/                           # Saved models & results (created after training)
│   ├── skin_disease_ensemble_model.h5
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── per_class_metrics.png
│   └── confidence_distribution.png
│
└── utils/                            # Utility functions (optional)
```

## 🚀 Quick Start

### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 2. **Train the Model**

```bash
python train.py
```

This will:
- 📷 Load all images from `MLDemoProj/IMG_CLASSES/`
- ⚖️ Compute class weights automatically
- 🏗️ Build the ensemble architecture
- 🎓 Train the model with EarlyStopping
- 💾 Save the trained model to `models/`
- 📊 Generate training visualizations

**Expected Output:**
- ✓ Trained model: `models/skin_disease_ensemble_model.h5`
- ✓ Training plot: `models/training_history.png`

**Training Time:** ~30-60 minutes (depending on total image count)

### 3. **Evaluate the Model**

```bash
python test.py
```

This will:
- 📊 Split data into training/test sets
- 🔍 Evaluate on test set
- 📈 Compute precision, recall, F1-score
- 📉 Generate confusion matrix
- 📋 Create detailed classification reports

**Generated Outputs:**
- ✓ `models/confusion_matrix.png` - Confusion matrix heatmap
- ✓ `models/per_class_metrics.png` - Per-class performance
- ✓ `models/confidence_distribution.png` - Prediction confidence histogram

### 4. **Interactive Predictions**

```bash
streamlit run app.py
```

Opens browser at `http://localhost:8501` with:
- 🔬 Make single image predictions
- 📊 View confidence scores
- 📈 Interactive visualizations
- 📋 Disease reference guide
- ℹ️ Model architecture details

## 🏗️ Model Architecture

### Ensemble Design

```
Input Image (224×224×3)
        ↓
        ├─────────────────────┬───────────────────────┐
        ↓                     ↓                       ↓
    MobileNetV2          EfficientNetB0         [Pre-trained weights]
    (Lightweight)        (Efficient)
        ↓                     ↓
GlobalAveragePooling2D  GlobalAveragePooling2D
    (1280)                (1280)
        ↓                     ↓
        └───────────┬─────────┘
                    ↓
            Concatenate (2560)
                    ↓
            Dense(128) + ReLU
                    ↓
            Dropout(0.5)
                    ↓
            Dense(10) + Softmax
                    ↓
            10 Disease Classes
```

### Why This Architecture?

| Feature | Benefit |
|---------|---------|
| **Ensemble** | Combines strengths of two models → better predictions |
| **MobileNetV2** | Lightweight, fast, mobile-friendly |
| **EfficientNetB0** | Balanced speed & accuracy |
| **Transfer Learning** | ImageNet pre-training accelerates learning |
| **GlobalAveragePooling2D** | Dimensionality reduction without losing information |
| **Concatenation** | Merges 2560 features for classification |
| **Dropout** | Reduces overfitting, improves generalization |

## ⚖️ Handling Class Imbalance

### The Problem
- Melanoma: **3,500** samples (tied largest)
- Eczema: **1,677** samples 
- Atopic Dermatitis: **1,250** samples (smallest)
- **2.8× difference** in class size

### The Solution: Automatic Class Weights

```python
# Computed automatically in train.py
class_weights = {
    0: 4.0,  # Atopic Dermatitis (rare) - high weight
    1: 1.4,  # Melanoma (common) - lower weight
    2: 1.4,  # Melanocytic Nevi (common) - lower weight
    ...
}
```

#### Formula
```
weight[class] = total_samples / (num_classes × samples_in_class)
```

#### Effect
- **Small classes** get higher loss weights → model "cares more"
- **Large classes** get lower loss weights → prevents bias
- Result: **Balanced accuracy** across all disease types

### During Training
```python
model.fit(
    images, labels,
    class_weight=class_weights,  # ← Applied here
    ...
)
```

## 🛑 Preventing Overfitting

### Early Stopping

```python
EarlyStopping(
    monitor='val_loss',        # Watch validation loss
    patience=5,                # Stop if no improvement for 5 epochs
    restore_best_weights=True  # Revert to best model
)
```

**How It Works:**
1. Training loss decreases (good)
2. Validation loss increases (overfitting detected)
3. After 5 epochs without improvement, stop training
4. Restore weights from the best epoch

### Dropout

```python
Dense(128, activation='relu') 
Dropout(0.5)  # Randomly deactivate 50% of neurons
```

**How It Works:**
- During training: Some neurons randomly "turn off"
- Forces the model to learn redundant representations
- Better generalization to unseen data
- Doesn't affect inference (predictions)

## 📊 Key Metrics

### Overall Metrics
- **Accuracy**: Percentage of correct predictions
- **Precision**: How many predicted positives were actually positive
- **Recall**: How many actual positives were correctly identified
- **F1-Score**: Harmonic mean of precision and recall (best for imbalanced data)

### Per-Class Metrics
Computed for each of 10 diseases separately to ensure balanced performance

## 🎬 Complete Training Pipeline

### Step-by-Step Execution

```
train.py
├── SECTION 1: Load all image files
│   ├── Read from MLDemoProj/IMG_CLASSES/
│   ├── Normalize pixel values [0, 1]
│   └── Create training dataset
│
├── SECTION 2: Compute class weights
│   ├── Count samples per class
│   ├── Apply balanced weight formula
│   └── Higher weight for minority classes
│
├── SECTION 3: Build ensemble model
│   ├── Load pre-trained MobileNetV2
│   ├── Load pre-trained EfficientNetB0
│   ├── Freeze backbone weights
│   ├── Add fusion + classification layers
│   └── Display model summary
│
├── SECTION 4: Compile model
│   ├── Optimizer: Adam (learning_rate=0.001)
│   ├── Loss: Categorical Crossentropy
│   └── Metric: Accuracy
│
├── SECTION 5: Train model
│   ├── Apply class weights for balance
│   ├── Monitor validation loss
│   ├── Stop training if overfitting detected
│   └── Save best model weights
│
├── SECTION 6: Save trained model
│   └── Save to models/skin_disease_ensemble_model.h5
│
└── SECTION 7: Visualize results
    └── Plot training/validation loss and accuracy
```

## 📈 Interpreting Results

### Confusion Matrix
- **Diagonal (dark)**: Correct predictions ✓
- **Off-diagonal (dark)**: Common misclassifications ✗
- **Normalized version**: Percentage per class

### Per-Class Metrics
- High F1-score: Model performs well on that class
- Low F1-score: Model struggles with that class (might need more samples)

### Confidence Distribution
- Centered near 1.0: Model is very confident
- Spread distribution: Model is uncertain
- Ideally: High confidence for correct predictions

## 🔬 Making Predictions

### Command Line
```python
from tensorflow import keras
import cv2
import numpy as np

# Load model
model = keras.models.load_model('models/skin_disease_ensemble_model.h5')

# Load and preprocess image
image = cv2.imread('path/to/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
image = image.astype('float32') / 255.0

# Predict
prediction = model.predict(np.expand_dims(image, axis=0))[0]
disease_id = np.argmax(prediction)
confidence = prediction[disease_id]

print(f"Disease: {DISEASE_CLASSES[disease_id]}")
print(f"Confidence: {confidence:.2%}")
```

### Using Streamlit App
Simply upload an image and see interactive predictions!

## 📚 Understanding The Code

### Important Concepts

#### 1. **One-Hot Encoding**
```python
# Label 3 (Atopic Dermatitis) becomes:
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```
- Used for multi-class classification
- Softmax output matches this format
- Required by categorical_crossentropy loss

#### 2. **Softmax Activation**
```python
# Raw predictions: [2.1, -1.3, 0.5, ...]
# After softmax: [0.6, 0.1, 0.2, ...]  <- probabilities sum to 1
```

#### 3. **Transfer Learning**
```python
# Pre-trained = ImageNet weights (trained on 1.2M images)
# Benefit = Learns image features from the start
# Result = Faster training, better performance
```

#### 4. **Batch Processing**
```python
# Instead of 1 image at a time (slow)
# Process 32 images together (fast & memory efficient)
batch_size = 32
```

## 🐛 Troubleshooting

### ❌ Memory Error
```bash
# Reduce batch size
BATCH_SIZE = 16  # Instead of 32
```

### ❌ Model Not Found
```bash
# Make sure you've run train.py first
python train.py
```

### ❌ No Images Found
```bash
# Verify dataset path
# Should be: MLDemoProj/IMG_CLASSES/[disease_name]/[images]
```

### ❌ Streamlit Not Working
```bash
# Install streamlit
pip install streamlit

# Run with absolute path
streamlit run C:/path/to/app.py
```

## 📖 File Descriptions

| File | Purpose |
|------|---------|
| `train.py` | Main training script with detailed comments |
| `test.py` | Evaluate model and generate metrics |
| `app.py` | Streamlit interactive dashboard |
| `requirements.txt` | Python package dependencies |
| `README.md` | This documentation |

## 💡 Tips & Tricks

### For Better Results
1. **More Images**: Collect more samples for small classes
2. **Data Augmentation**: Rotate, flip, zoom images (add to train.py)
3. **Fine-tuning**: Unfreeze some backbone layers and retrain (advanced)
4. **Ensemble Improvement**: Add more models (ResNet50, InceptionV3, etc.)

### For Faster Training
1. **Reduce image resolution**: 224→192x192
2. **Reduce dataset**: Sample a subset for testing
3. **Use GPU**: Install tensorflow-gpu (NVIDIA CUDA required)

### For Production
1. **Model Quantization**: Reduce model size 4× with int8 quantization
2. **ONNX Export**: Convert to format compatible with many frameworks
3. **Caching**: Cache predictions for repeated images

## ⚠️ Important Disclaimers

**THIS IS NOT MEDICAL ADVICE**

- This model is for **educational and research purposes only**
- **Never use for actual medical diagnosis** without professional review
- Always **consult with a qualified dermatologist** for medical decisions
- Model confidence does not guarantee accuracy

## 📞 Support

For issues or questions, check:
1. Comments in the Python files (very detailed!)
2. Streamlit app's "Model Information" tab
3. This README file
4. TensorFlow/Keras documentation

## 🎓 Learning Resources

### Concepts Used
- **Convolutional Neural Networks (CNN)**
- **Transfer Learning**
- **Ensemble Methods**
- **Class Imbalance Handling**
- **Early Stopping & Dropout**

### Recommended Reading
- TensorFlow/Keras Documentation
- Deep Learning course (Andrew Ng)
- PyImageSearch tutorials

## 📄 License

[Add your license here]

## 👨‍💻 Author

[Your name/organization]

---

**Last Updated**: February 2026

**Status**: ✅ Production Ready

For questions or improvements, please reach out!
