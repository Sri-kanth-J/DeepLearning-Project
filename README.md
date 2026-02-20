# Improved Skin Disease Classification 🩺

An advanced deep learning system for skin disease classification using state-of-the-art models and the facial skin condition dataset from Hugging Face.

## 🚀 Quick Start

### 1. Setup Environment
```bash
python setup.py
```

### 2. Authenticate with Hugging Face
```bash
huggingface-cli login
```

### 3. Download Dataset
```bash
python download_dataset.py
```

### 4. Train Models
```bash
python train_improved.py
```

### 5. Test Models
```bash
python test_improved.py
```

### 6. Run Web App
```bash
streamlit run app.py
```

## 📊 Expected Performance

- **Previous Accuracy**: 6%
- **Expected Accuracy**: 60-85%
- **Training Time**: 1-3 hours (depending on hardware)

## 🏗️ Architecture

### Models Implemented
1. **MobileNetV3Large**: Optimized for mobile deployment
2. **EfficientNetB4**: Best accuracy performance

### Training Strategy
1. **Transfer Learning**: Pre-trained ImageNet weights
2. **Fine-tuning**: Gradual unfreezing with lower learning rates
3. **Data Augmentation**: Aggressive augmentation for better generalization
4. **Mixed Precision**: Faster training with FP16

## 📁 Project Structure

```
├── setup.py                 # Environment setup
├── download_dataset.py      # Download HuggingFace dataset
├── train_improved.py        # Advanced training script
├── test_improved.py         # Comprehensive testing
├── train_simple.py          # Original simple training
├── test.py                  # Original test script
├── app.py                   # Streamlit web application
├── requirements.txt         # Python dependencies
├── dataset/                 # Local dataset storage
│   ├── train/              # Training images
│   ├── val/                # Validation images
│   └── test/               # Test images
└── models/                 # Trained models
    ├── mobilenet_best.h5   # Best MobileNet model
    ├── efficientnet_best.h5 # Best EfficientNet model
    └── class_indices.npy   # Class mapping
```

## 🔧 Key Improvements

### 1. Better Dataset
- **Source**: UniDataPro/facial-skin-condition-dataset (Hugging Face)
- **Quality**: Higher quality, professionally labeled
- **Size**: Larger and more diverse dataset
- **Storage**: Local storage for faster training

### 2. Optimized Models
- **MobileNetV3Large**: Instead of MobileNetV2
- **EfficientNetB4**: State-of-the-art efficiency
- **Better Architecture**: Optimized classification heads
- **Regularization**: BatchNorm + Dropout for better generalization

### 3. Advanced Training
- **Two-Phase Training**: Transfer learning → Fine-tuning
- **Learning Rate Scheduling**: Adaptive learning rates
- **Data Augmentation**: More aggressive augmentation
- **Mixed Precision**: 2x faster training
- **Early Stopping**: Prevent overfitting

### 4. Hyperparameter Optimization
```python
# Previous Settings
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

# Improved Settings  
BATCH_SIZE = 16          # Better gradient estimates
INITIAL_EPOCHS = 50      # More training
FINE_TUNE_EPOCHS = 30    # Additional fine-tuning
INITIAL_LR = 0.001       # Standard transfer learning rate
FINE_TUNE_LR = 0.0001    # Lower rate for fine-tuning
```

## 📈 Training Process

### Phase 1: Transfer Learning (50 epochs)
- Frozen backbone (pre-trained weights)
- Train only classification head
- Higher learning rate (0.001)

### Phase 2: Fine-tuning (30 epochs)  
- Unfreeze top layers of backbone
- Lower learning rate (0.0001/10)
- Fine-tune entire model

## 🧪 Testing & Evaluation

The improved test script provides:

- **Overall Accuracy**: Final test accuracy
- **Per-Class Metrics**: Precision, recall, F1-score
- **Confusion Matrix**: Visual error analysis
- **Confidence Analysis**: Prediction reliability
- **Visualization**: Comprehensive plots and charts

## 💡 Usage Tips

### For Best Results:
1. **Use GPU**: Significantly faster training
2. **Monitor Training**: Watch for overfitting
3. **Compare Models**: Both MobileNet and EfficientNet are trained
4. **Test Thoroughly**: Use the comprehensive test script

### Troubleshooting:
1. **Dataset Issues**: Run `python download_dataset.py` first
2. **Memory Issues**: Reduce batch size in scripts
3. **GPU Issues**: Models automatically fall back to CPU
4. **Login Issues**: Use `huggingface-cli login` for dataset access

## 🔄 Migration from Original

If you have the original dataset structure:
1. Keep your existing `MLDemoProj/IMG_CLASSES/` folder
2. The improved test script automatically detects it
3. Use the new training script for better results
4. Original models remain compatible

## 📚 Advanced Features

### Custom Image Testing
```python
from test_improved import ModelTester

tester = ModelTester()
model, _, _ = tester.load_best_model()
class_indices = tester.load_class_indices()

# Test your own image
result = tester.test_single_image(model, "path/to/image.jpg", class_indices)
```

### Model Comparison
The training script automatically trains both models and reports which performs better on your specific dataset.

## 📊 Expected Results

- **Training Time**: 1-3 hours total
- **MobileNet Accuracy**: 60-75%
- **EfficientNet Accuracy**: 70-85%
- **Model Size**: 
  - MobileNet: ~15MB
  - EfficientNet: ~70MB

---

## 🤝 Support

If you encounter any issues:
1. Check the console output for specific error messages
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Verify Hugging Face authentication: `huggingface-cli login`
4. Check GPU memory if using GPU: reduce batch size if needed

**Happy Training! 🎉**