import numpy as np
import os
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from datetime import datetime


class ImagePreprocessor:
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def read_and_preprocess(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.target_size)
            image = image.astype('float32') / 255.0
            return image
        except Exception as e:
            print(f"Error preprocessing {image_path}: {str(e)}")
            return None
    
    @staticmethod
    def normalize(image):
        return image.astype('float32') / 255.0
    
    @staticmethod
    def denormalize(image):
        return (image * 255).astype('uint8')


def load_dataset_from_directory(base_path, target_size=(224, 224), verbose=True):
    
    preprocessor = ImagePreprocessor(target_size=target_size)
    images = []
    labels = []
    class_names = []
    
    class_folders = sorted([f for f in os.listdir(base_path) 
                           if os.path.isdir(os.path.join(base_path, f))])
    
    if verbose:
        print(f"\nLoading dataset from: {base_path}")
        print(f"Found {len(class_folders)} classes\n")
    
    for class_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(base_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if verbose:
            print(f"  [{class_idx+1}] {class_name}: {len(image_files)} images")
        
        class_names.append(class_name)
        
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            image = preprocessor.read_and_preprocess(image_path)
            
            if image is not None:
                images.append(image)
                labels.append(class_idx)
    
    images = np.array(images)
    labels = np.array(labels)
    
    if verbose:
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  - Total images: {len(images)}")
        print(f"  - Shape: {images.shape}")
        print(f"  - Label range: {labels.min()} to {labels.max()}")
    
    return images, labels, class_names


def compute_class_weights_balanced(labels, num_classes):
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(num_classes),
        y=labels
    )
    
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    return class_weights_dict, class_weights


def print_class_distribution(labels, class_names):
    
    print("\nClass Distribution:")
    print("-" * 60)
    
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    for class_idx, count in zip(unique, counts):
        percentage = (count / total) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {class_names[class_idx]:40s} {count:6d} ({percentage:5.1f}%) {bar}")
    
    print("-" * 60)


def plot_training_curves(history, save_path=None):
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to: {save_path}")
    
    return fig


def plot_confusion_matrix_heatmap(y_true, y_pred, class_names, save_path=None, normalize=False):
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    return fig


def plot_class_distribution(labels, class_names, save_path=None):
    
    unique, counts = np.unique(labels, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    bars = ax.bar(range(len(unique)), counts, color='steelblue', edgecolor='black')
    
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([class_names[i] for i in unique], rotation=45, ha='right')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Distribution plot saved to: {save_path}")
    
    return fig


def get_model_summary_dataframe(model):
    
    layers_info = []
    
    for layer in model.layers:
        config = layer.get_config()
        weights = layer.get_weights()
        
        num_params = sum([tf.size(w).numpy() for w in weights])
        
        layers_info.append({
            'Layer Name': layer.name,
            'Type': layer.__class__.__name__,
            'Output Shape': str(layer.output_shape),
            'Parameters': int(num_params),
            'Trainable': layer.trainable
        })
    
    df = pd.DataFrame(layers_info)
    return df


def print_model_info(model):
    
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)
    
    model.summary()
    
    df = get_model_summary_dataframe(model)
    print("\nLayerwise Breakdown:")
    print(df.to_string(index=False))
    
    total_params = sum([tf.size(w).numpy() for layer in model.layers for w in layer.get_weights()])
    trainable_params = sum([tf.size(w).numpy() for layer in model.layers 
                           if layer.trainable for w in layer.get_weights()])
    
    print("\n" + "-"*70)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    print("-"*70)


if __name__ == "__main__":
    print("Utility module loaded successfully!")


# ============================================================================
# DATA POINT SAVING FUNCTIONS
# ============================================================================

def save_training_data_points(history, class_weights, class_names, save_dir="models/data_points"):
    """
    Save training data points to models/data_points/ folder.
    
    Args:
        history: Keras training history
        class_weights: Computed class weights dict
        class_names: List of disease class names
        save_dir: Directory to save data points
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save training history
    history_data = {
        'loss': history.history.get('loss', []),
        'accuracy': history.history.get('accuracy', []),
        'val_loss': history.history.get('val_loss', []),
        'val_accuracy': history.history.get('val_accuracy', []),
        'epochs': len(history.history.get('loss', [])),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{save_dir}/training_history.json", 'w') as f:
        json.dump(history_data, f, indent=2)
    
    # Save class weights
    weights_data = {
        'class_weights': class_weights,
        'class_names': class_names,
        'num_classes': len(class_names),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{save_dir}/class_weights.json", 'w') as f:
        json.dump(weights_data, f, indent=2)
    
    print(f"✓ Training data points saved to: {save_dir}")


def save_evaluation_data_points(y_true, y_pred, predictions, class_names, save_dir="models/data_points"):
    """
    Save evaluation results to models/data_points/ folder.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        predictions: Raw prediction probabilities
        class_names: List of disease class names
        save_dir: Directory to save data points
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    np.save(f"{save_dir}/confusion_matrix.npy", cm)
    
    # Save predictions
    np.save(f"{save_dir}/test_predictions.npy", predictions)
    np.save(f"{save_dir}/test_true_labels.npy", y_true)
    np.save(f"{save_dir}/test_pred_labels.npy", y_pred)
    
    # Save evaluation metrics
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    eval_data = {
        'classification_report': report,
        'accuracy': report['accuracy'],
        'macro_avg_f1': report['macro avg']['f1-score'],
        'weighted_avg_f1': report['weighted avg']['f1-score'],
        'class_names': class_names,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{save_dir}/evaluation_results.json", 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    print(f"✓ Evaluation data points saved to: {save_dir}")


def load_training_data_points(save_dir="models/data_points"):
    """Load saved training data points."""
    try:
        with open(f"{save_dir}/training_history.json", 'r') as f:
            history = json.load(f)
        
        with open(f"{save_dir}/class_weights.json", 'r') as f:
            weights = json.load(f)
        
        return history, weights
    except FileNotFoundError:
        print(f"No training data points found in: {save_dir}")
        return None, None


def load_evaluation_data_points(save_dir="models/data_points"):
    """Load saved evaluation data points."""
    try:
        with open(f"{save_dir}/evaluation_results.json", 'r') as f:
            eval_data = json.load(f)
        
        predictions = np.load(f"{save_dir}/test_predictions.npy")
        y_true = np.load(f"{save_dir}/test_true_labels.npy") 
        y_pred = np.load(f"{save_dir}/test_pred_labels.npy")
        cm = np.load(f"{save_dir}/confusion_matrix.npy")
        
        return eval_data, predictions, y_true, y_pred, cm
    except FileNotFoundError:
        print(f"No evaluation data points found in: {save_dir}")
        return None, None, None, None, None


# ============================================================================
# ENHANCED PREPROCESSING FUNCTIONS
# ============================================================================

def batch_preprocess_images(image_paths, target_size=(224, 224), batch_size=32):
    """
    Preprocess images in batches for memory efficiency.
    
    Args:
        image_paths: List of image file paths
        target_size: Target image dimensions
        batch_size: Number of images to process at once
    
    Returns:
        generator: Yields batches of preprocessed images
    """
    preprocessor = ImagePreprocessor(target_size=target_size)
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for path in batch_paths:
            img = preprocessor.read_and_preprocess(path)
            if img is not None:
                batch_images.append(img)
        
        if batch_images:
            yield np.array(batch_images)


def augment_image(image, augmentation_type='random'):
    """
    Apply data augmentation to single image.
    
    Args:
        image: Input image array (224, 224, 3)
        augmentation_type: 'rotation', 'flip', 'brightness', 'random'
    
    Returns:
        augmented_image: Augmented image array
    """
    if augmentation_type == 'rotation':
        angle = np.random.uniform(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    
    elif augmentation_type == 'flip':
        return cv2.flip(image, np.random.choice([0, 1, -1]))
    
    elif augmentation_type == 'brightness':
        factor = np.random.uniform(0.8, 1.2)
        return np.clip(image * factor, 0, 1)
    
    elif augmentation_type == 'random':
        aug_type = np.random.choice(['rotation', 'flip', 'brightness'])
        return augment_image(image, aug_type)
    
    return image


def create_data_summary(base_path="MLDemoProj/IMG_CLASSES", save_to_models=True):
    """
    Create comprehensive data summary and optionally save to models folder.
    
    Args:
        base_path: Path to dataset
        save_to_models: Whether to save summary to models/data_points/
    """
    if not os.path.exists(base_path):
        print(f"Dataset not found at: {base_path}")
        return None
    
    class_folders = sorted([f for f in os.listdir(base_path) 
                           if os.path.isdir(os.path.join(base_path, f))])
    
    summary = {
        'dataset_path': base_path,
        'num_classes': len(class_folders),
        'class_distribution': {},
        'total_images': 0,
        'timestamp': datetime.now().isoformat()
    }
    
    for folder in class_folders:
        folder_path = os.path.join(base_path, folder)
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        count = len(image_files)
        summary['class_distribution'][folder] = count
        summary['total_images'] += count
    
    # Add statistics
    counts = list(summary['class_distribution'].values())
    summary['statistics'] = {
        'min_samples': min(counts),
        'max_samples': max(counts),
        'mean_samples': np.mean(counts),
        'std_samples': np.std(counts),
        'imbalance_ratio': max(counts) / min(counts)
    }
    
    if save_to_models:
        save_dir = "models/data_points"
        os.makedirs(save_dir, exist_ok=True)
        
        with open(f"{save_dir}/dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Dataset summary saved to: {save_dir}/dataset_summary.json")
    
    return summary
