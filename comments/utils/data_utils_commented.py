"""
================================================================================
DATA UTILITIES MODULE - SKIN DISEASE CLASSIFICATION
================================================================================

Purpose:
    Reusable utility functions and classes for:
    1. Image preprocessing and loading
    2. Class weight computation for imbalanced data
    3. Data visualization and exploration
    4. Model utilities and summaries

Module Contents:
    - ImagePreprocessor: Class for image loading and normalization
    - load_dataset_from_directory(): Load entire dataset with progress
    - compute_class_weights_balanced(): Compute balanced class weights
    - print_class_distribution(): Print class distribution with visualization
    - plot_training_curves(): Plot training and validation loss/accuracy
    - plot_confusion_matrix_heatmap(): Visualize confusion matrix
    - plot_class_distribution(): Bar chart of sample counts
    - get_model_summary_dataframe(): Convert model summary to DataFrame
    - print_model_info(): Print detailed model information
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler


# ============================================================================
# IMAGE PREPROCESSOR CLASS
# ============================================================================

class ImagePreprocessor:
    """
    Class for consistent image preprocessing throughout the project.
    
    Handles:
    - Loading images from file paths
    - Resizing to target dimensions
    - Normalizing pixel values
    - Denormalizing for visualization
    
    Attributes:
        target_size (tuple): Target image dimensions
        mean (float/array): Mean for normalization
        std (float/array): Std dev for normalization
    """
    
    def __init__(self, target_size=(224, 224)):
        """
        Initialize the preprocessor.
        
        Args:
            target_size (tuple): Target dimensions (height, width)
        """
        self.target_size = target_size
        self.mean = 0.5  # Default mean for [0,1] normalization
        self.std = 0.5   # Default std for [0,1] normalization
    
    def read_and_preprocess(self, image_path):
        """
        Read image from file and preprocess it.
        
        Steps:
        1. Read image using PIL
        2. Convert to RGB if needed
        3. Resize to target_size
        4. Convert to numpy array
        5. Normalize to [0, 1]
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            image (np.array): Preprocessed image (224, 224, 3)
        """
        try:
            # Open image with PIL
            image = Image.open(image_path).convert('RGB')
            
            # Resize to target size
            image = image.resize(self.target_size, Image.BILINEAR)
            
            # Convert to numpy array
            image_array = np.array(image, dtype='float32')
            
            # Normalize to [0, 1]
            image_normalized = image_array / 255.0
            
            return image_normalized
        
        except Exception as e:
            print(f"Error loading {image_path}: {str(e)}")
            return None
    
    def normalize(self, image):
        """
        Normalize image to [0, 1] range.
        
        Args:
            image (np.array): Image with pixel values [0, 255]
        
        Returns:
            normalized (np.array): Image with values [0, 1]
        """
        return image.astype('float32') / 255.0
    
    def denormalize(self, image):
        """
        Denormalize image from [0, 1] back to [0, 255].
        
        Args:
            image (np.array): Image with values [0, 1]
        
        Returns:
            denormalized (np.array): Image with pixel values [0, 255]
        """
        return (image * 255.0).astype('uint8')


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_dataset_from_directory(base_path, img_size=(224, 224)):
    """
    Load entire dataset from directory structure.
    
    Expected structure:
        base_path/
        ├── class_1/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── class_2/
        │   └── ...
        └── ...
    
    Progress:
    - Uses tqdm to show loading progress
    - Prints summary when complete
    
    Args:
        base_path (str): Path to dataset root directory
        img_size (tuple): Target image size
    
    Returns:
        images (list): Loaded images
        labels (list): Class labels
        class_names (list): Mapping of class indices to names
    """
    preprocessor = ImagePreprocessor(img_size)
    images = []
    labels = []
    class_names = []
    
    # Get sorted list of class directories
    class_dirs = sorted([d for d in os.listdir(base_path) 
                        if os.path.isdir(os.path.join(base_path, d))])
    
    print(f"Found {len(class_dirs)} classes: {class_dirs}")
    print("\nLoading dataset...")
    
    # Iterate through each class
    for class_idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(base_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        class_names.append(class_name)
        
        # Load images from this class with progress bar
        pbar = tqdm(image_files, desc=f"Class {class_idx}: {class_name}")
        
        for img_file in pbar:
            img_path = os.path.join(class_path, img_file)
            image = preprocessor.read_and_preprocess(img_path)
            
            if image is not None:
                images.append(image)
                labels.append(class_idx)
        
        print(f"  ✓ Loaded {len([l for l in labels if l == class_idx])} images")
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Total images: {len(images)}")
    print(f"  Image shape: {images[0].shape}")
    print(f"  Labels shape: {labels.shape}")
    
    return images, labels, class_names


# ============================================================================
# CLASS WEIGHT COMPUTATION
# ============================================================================

def compute_class_weights_balanced(labels):
    """
    Compute balanced class weights to handle imbalanced dataset.
    
    Formula:
        weight[class_i] = total_samples / (num_classes × samples_in_class_i)
    
    Why this works:
    - Classes with fewer samples get higher weights
    - Loss contribution is amplified for rare classes
    - Model focuses more on learning rare classes
    
    Example:
        Total 10k samples, 10 classes
        Class A: 2k samples → weight = 10000 / (10 × 2000) = 0.5
        Class B: 100 samples → weight = 10000 / (10 × 100) = 10.0
        Class B's loss is weighted 20x higher than Class A
    
    Args:
        labels (np.array): Array of class labels
    
    Returns:
        class_weights_dict (dict): Mapping {class_idx: weight}
        class_weights_array (np.array): Array of weights (sorted by class)
    """
    
    # Count samples per class
    classes, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    num_classes = len(classes)
    
    # Compute balanced weights
    class_weights_dict = {}
    class_weights_array = np.zeros(num_classes)
    
    for cls_idx, count in zip(classes, counts):
        # Balanced weight formula
        weight = total_samples / (num_classes * count)
        class_weights_dict[int(cls_idx)] = float(weight)
        class_weights_array[int(cls_idx)] = float(weight)
    
    return class_weights_dict, class_weights_array


# ============================================================================
# DATA EXPLORATION VISUALIZATION
# ============================================================================

def print_class_distribution(labels, class_names=None):
    """
    Print class distribution with ASCII bar chart.
    
    Shows:
    - Count of samples per class
    - Percentage of total dataset
    - Visual representation with █ symbols
    
    Args:
        labels (np.array): Array of class labels
        class_names (list): Optional class names
    """
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION")
    print("="*60)
    
    max_count = counts.max()
    
    for cls_idx, count in zip(classes, counts):
        percentage = (count / total) * 100
        bar_length = int((count / max_count) * 40)
        bar = "█" * bar_length
        
        class_label = class_names[cls_idx] if class_names else f"Class {cls_idx}"
        
        print(f"{class_label:30} | {bar:40} | {count:5} ({percentage:5.1f}%)")
    
    print("="*60)


def plot_training_curves(history):
    """
    Plot training and validation loss/accuracy curves.
    
    Creates:
    - Left plot: Loss over epochs
    - Right plot: Accuracy over epochs
    
    Args:
        history (keras.callbacks.History): Training history object
    
    Returns:
        fig (matplotlib.figure.Figure): Figure object
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix_heatmap(cm, class_names, normalize=False):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm (np.array): Confusion matrix
        class_names (list): Class names
        normalize (bool): Whether to normalize rows (percentages)
    
    Returns:
        fig (matplotlib.figure.Figure): Figure object
    """
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if normalize:
        # Normalize each row by dividing by row sum
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        threshold = cm_normalized.max() / 2.0
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        threshold = cm.max() / 2.0
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set labels
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Set ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                value = cm_normalized[i, j]
                text = f'{value:.2%}'
            else:
                value = cm[i, j]
                text = str(value)
            
            color = 'white' if value > threshold else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_class_distribution(labels, class_names=None):
    """
    Create bar chart of sample counts per class.
    
    Args:
        labels (np.array): Array of class labels
        class_names (list): Optional class names
    
    Returns:
        fig (matplotlib.figure.Figure): Figure object
    """
    
    classes, counts = np.unique(labels, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar chart
    if class_names:
        class_labels = [class_names[i] for i in classes]
    else:
        class_labels = [f"Class {i}" for i in classes]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    bars = ax.bar(class_labels, counts, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    return fig


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def get_model_summary_dataframe(model):
    """
    Convert Keras model summary to pandas DataFrame.
    
    Returns:
    - Layer name
    - Output shape
    - Parameter count
    - Layer type
    
    Args:
        model (keras.Model): Trained Keras model
    
    Returns:
        df (pd.DataFrame): Model summary as DataFrame
    """
    
    layer_data = []
    
    for layer in model.layers:
        layer_data.append({
            'Layer': layer.name,
            'Type': layer.__class__.__name__,
            'Output Shape': str(layer.output_shape),
            'Parameters': layer.count_params()
        })
    
    df = pd.DataFrame(layer_data)
    return df


def print_model_info(model):
    """
    Print detailed model information.
    
    Shows:
    - Total parameters
    - Trainable parameters
    - Non-trainable parameters
    - Model depth
    - Per-layer information
    
    Args:
        model (keras.Model): Trained Keras model
    """
    
    print("\n" + "="*80)
    print("MODEL INFORMATION")
    print("="*80)
    
    # Get model summary
    model.summary()
    
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS")
    print("-"*80)
    
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) 
                           for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"Total Parameters:        {total_params:,}")
    print(f"Trainable Parameters:    {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    print(f"Model Depth:             {len(model.layers)} layers")
    
    print("\n" + "="*80)
