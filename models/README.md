# 📁 Models Directory

This folder stores:

## 🤖 **Trained Models**
- `skin_disease_ensemble_model.h5` - Main trained ensemble model
- `skin_disease_ensemble_model.keras` - Alternative format
- `model_checkpoints/` - Training checkpoints

## 📊 **Data Points & Metrics**
- `training_history.json` - Loss/accuracy curves data
- `class_weights.json` - Computed class weights
- `evaluation_results.json` - Test set performance
- `confusion_matrix.npy` - Confusion matrix data

## 📈 **Training Logs**
- `training_logs/` - TensorBoard logs
- `model_config.json` - Model architecture config

## 🔧 **Usage**
```python
# Load trained model
model = keras.models.load_model('models/skin_disease_ensemble_model.h5')

# Save training data points
import json
with open('models/training_history.json', 'w') as f:
    json.dump(history.history, f)
```