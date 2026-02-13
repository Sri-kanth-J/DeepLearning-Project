"""
================================================================================
SKIN DISEASE CLASSIFICATION - TEST AND EVALUATION SCRIPT
================================================================================

Purpose:
    Evaluate the trained MobileNet model on test data
    Generate performance metrics, confusion matrix, and classification reports

Features:
    - Load trained model
    - Evaluate on test set
    - Generate confusion matrix
    - Classification metrics per disease
    - Visualizations for model performance

================================================================================
"""

# ============================================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Image Processing
import cv2
import warnings
warnings.filterwarnings('ignore')

print("✓ All libraries imported successfully!")


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path('MLDemoProj/IMG_CLASSES')
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
BATCH_SIZE = 32

DISEASE_CLASSES = [
    "1. Eczema",
    "2. Melanoma",
    "3. Atopic Dermatitis",
    "4. Basal Cell Carcinoma",
    "5. Melanocytic Nevi",
    "6. Benign Keratosis-like Lesions",
    "7. Psoriasis/Lichen Planus",
    "8. Seborrheic Keratoses",
    "9. Tinea/Ringworm/Candidiasis",
    "10. Warts/Molluscum"
]

NUM_CLASSES = len(DISEASE_CLASSES)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_and_preprocess_images(base_path, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Load images from dataset (same as train.py).
    
    Args:
        base_path (str): Path to dataset root
        img_size (tuple): Target image size
    
    Returns:
        images (np.array): Preprocessed images
        labels (np.array): Class labels
        class_names (list): Class folder names
    """
    
    print("\n" + "="*70)
    print("LOADING TEST DATA")
    print("="*70)
    
    images = []
    labels = []
    class_names = []
    
    # Get class folders
    class_folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])
    
    print(f"Found {len(class_folders)} classes\n")
    
    for class_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(base_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"  Class {class_idx + 1}: {class_name}: {len(image_files)} images")
        class_names.append(class_name)
        
        # Load each image in the class folder
        for image_file in image_files:
            try:
                image_path = os.path.join(class_path, image_file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, img_size)
                image = image.astype('float32') / 255.0
                
                images.append(image)
                labels.append(class_idx)
                
            except Exception as e:
                continue
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"\n✓ Total images loaded: {len(images)}")
    print(f"  Shape: {images.shape}")
    
    return images, labels, class_names


def split_test_set(images, labels, test_size=0.2, random_state=42):
    """
    Split data into training and test sets.
    
    Args:
        images (np.array): All images
        labels (np.array): All labels
        test_size (float): Proportion of test set
        random_state (int): Random seed
    
    Returns:
        X_test, y_test: Test images and labels
    """
    
    _, X_test, _, y_test = train_test_split(
        images, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels  # Keep class distribution
    )
    
    return X_test, y_test


def load_model(model_path='models/skin_disease_mobilenet_model.h5'):
    """
    Load the trained MobileNet model.
    
    Args:
        model_path (str): Path to saved model
    
    Returns:
        model: Loaded Keras model
    """
    
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = keras.models.load_model(model_path)
    print(f"✓ Model loaded from: {model_path}")
    print(f"  File size: {os.path.getsize(model_path) / 1e6:.2f} MB")
    
    return model


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test (np.array): Test images
        y_test (np.array): Test labels
    
    Returns:
        test_loss (float): Loss on test set
        test_accuracy (float): Accuracy on test set
    """
    
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST SET")
    print("="*70)
    
    # Convert labels to one-hot
    y_test_encoded = keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(
        X_test, y_test_encoded,
        batch_size=BATCH_SIZE,
        verbose=0
    )
    
    print(f"\n✓ Evaluation Results:")
    print(f"  - Test Loss: {test_loss:.4f}")
    print(f"  - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    return test_loss, test_accuracy


def make_predictions(model, X_test):
    """
    Make predictions on test set.
    
    Args:
        model: Trained model
        X_test (np.array): Test images
    
    Returns:
        predictions (np.array): Probability predictions
        predicted_classes (np.array): Predicted class indices
    """
    
    print("\nMaking predictions on test set...")
    
    predictions = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    print(f"✓ Made predictions for {len(predicted_classes)} images")
    
    return predictions, predicted_classes


def compute_metrics(y_test, y_pred):
    """
    Compute classification metrics.
    
    Args:
        y_test (np.array): True labels
        y_pred (np.array): Predicted labels
    
    Returns:
        metrics (dict): Dictionary of metrics
    """
    
    print("\n" + "="*70)
    print("COMPUTING CLASSIFICATION METRICS")
    print("="*70)
    
    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print(f"\n✓ Overall Metrics:")
    print(f"  - Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")
    
    # Per-class metrics
    print(f"\n✓ Per-Class Metrics:")
    print("-" * 70)
    
    per_class_precision = precision_score(y_test, y_pred, average=None)
    per_class_recall = recall_score(y_test, y_pred, average=None)
    per_class_f1 = f1_score(y_test, y_pred, average=None)
    
    for i in range(NUM_CLASSES):
        print(f"  Class {i}: Precision={per_class_precision[i]:.3f} | "
              f"Recall={per_class_recall[i]:.3f} | F1={per_class_f1[i]:.3f}")
    
    return metrics


def generate_classification_report(y_test, y_pred):
    """
    Generate detailed classification report.
    
    Args:
        y_test (np.array): True labels
        y_pred (np.array): Predicted labels
    
    Returns:
        report (str): Formatted classification report
    """
    
    report = classification_report(
        y_test, y_pred,
        target_names=DISEASE_CLASSES,
        digits=4
    )
    
    print("\n" + "="*70)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*70)
    print(report)
    
    return report


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix(y_test, y_pred, class_names, save_path='models/confusion_matrix.png'):
    """
    Plot confusion matrix heatmap.
    
    WHAT IS A CONFUSION MATRIX?
    - Rows: True class labels
    - Columns: Predicted class labels
    - Diagonal: Correct predictions
    - Off-diagonal: Misclassifications
    
    INTERPRETATION:
    - Dark colors on diagonal = good
    - Dark colors off-diagonal = common errors
    
    Args:
        y_test (np.array): True labels
        y_pred (np.array): Predicted labels
        class_names (list): Class names
        save_path (str): Where to save plot
    """
    
    print("\n" + "="*70)
    print("GENERATING CONFUSION MATRIX")
    print("="*70)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalize for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
    plt.setp(axes[0].get_yticklabels(), rotation=0)
    
    # Plot 2: Normalized (percentages)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Percentage'})
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
    plt.setp(axes[1].get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm


def plot_per_class_metrics(y_test, y_pred, class_names, save_path='models/per_class_metrics.png'):
    """
    Plot per-class precision, recall, and F1-score.
    
    Args:
        y_test (np.array): True labels
        y_pred (np.array): Predicted labels
        class_names (list): Class names
        save_path (str): Where to save plot
    """
    
    print("\nGenerating per-class metrics plot...")
    
    # Compute per-class metrics
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    
    # Create DataFrame
    metrics_df = pd.DataFrame({
        'Disease': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }).sort_values('F1-Score', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = np.arange(len(metrics_df))
    width = 0.25
    
    ax.barh(x - width, metrics_df['Precision'], width, label='Precision', color='#3498db')
    ax.barh(x, metrics_df['Recall'], width, label='Recall', color='#2ecc71')
    ax.barh(x + width, metrics_df['F1-Score'], width, label='F1-Score', color='#e74c3c')
    
    ax.set_yticks(x)
    ax.set_yticklabels(metrics_df['Disease'], fontsize=10)
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Per-class metrics saved to: {save_path}")
    
    plt.show()


def plot_prediction_confidence_distribution(predictions, save_path='models/confidence_distribution.png'):
    """
    Plot distribution of prediction confidence scores.
    
    WHAT DOES THIS SHOW?
    - Histogram of max probability per prediction
    - Higher values = model is confident
    - Lower values = model is uncertain
    
    Args:
        predictions (np.array): Prediction probabilities [batch_size, num_classes]
        save_path (str): Where to save plot
    """
    
    print("\nGenerating confidence distribution plot...")
    
    # Get maximum confidence per prediction
    max_confidences = np.max(predictions, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(max_confidences, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(max_confidences.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {max_confidences.mean():.3f}')
    ax.axvline(np.median(max_confidences), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(max_confidences):.3f}')
    
    ax.set_xlabel('Max Prediction Confidence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Prediction Confidence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confidence distribution saved to: {save_path}")
    
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("SKIN DISEASE CLASSIFICATION - MODEL EVALUATION")
    print("="*70)
    
    # Load data
    images, labels, class_names = load_and_preprocess_images(BASE_DIR)
    
    # Split into test set
    X_test, y_test = split_test_set(images, labels)
    
    print(f"\n✓ Test set created:")
    print(f"  - Test size: {len(X_test)} images")
    
    # Load model
    model = load_model()
    
    # Evaluate
    test_loss, test_accuracy = evaluate_model(model, X_test, y_test)
    
    # Make predictions
    predictions, y_pred = make_predictions(model, X_test)
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred)
    
    # Generate report
    report = generate_classification_report(y_test, y_pred)
    
    # Create visualizations
    confusion_mat = plot_confusion_matrix(y_test, y_pred, class_names)
    plot_per_class_metrics(y_test, y_pred, class_names)
    plot_prediction_confidence_distribution(predictions)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETED!")
    print("="*70)
    print("\nGenerated files:")
    print("  ✓ models/confusion_matrix.png")
    print("  ✓ models/per_class_metrics.png")
    print("  ✓ models/confidence_distribution.png")
    print("="*70 + "\n")
