"""
================================================================================
SKIN DISEASE IMAGE CLASSIFICATION - MOBILENET MODEL TRAINING SCRIPT
================================================================================

Purpose: 
    Train a MobileNet deep learning model using MobileNetV2 architecture
    to classify skin diseases across 10 different categories.

Key Features:
    - MobileNetV2 architecture (optimized for speed and efficiency)
    - Automatic class weight computation for imbalanced dataset handling
    - EarlyStopping to prevent overfitting
    - Fast training (7-9 hours target) with memory efficiency
    - Modular, well-commented code for easy understanding

Dataset Balance:
    - Melanoma: ~3.5k (tied largest)
    - Melanocytic Nevi: ~3.5k (tied largest)
    - Basal Cell Carcinoma: ~3.32k
    - Others: 1.7k - 2.6k (smaller classes)
    
This script handles class imbalance by computing and applying class weights.
Training optimized for i5 8th gen + 16GB RAM hardware.
================================================================================
"""

# ============================================================================
# SECTION 1: IMPORT ALL REQUIRED LIBRARIES
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Image Processing
import cv2
from PIL import Image

# Utilities
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("✓ All libraries imported successfully!")


# ============================================================================
# SECTION 2: CONFIGURATION AND PATH SETUP
# ============================================================================

# Define the base directory containing all skin disease images
BASE_DIR = Path('MLDemoProj/IMG_CLASSES')

# Image preprocessing parameters
IMG_HEIGHT = 224  # Height of input images (required by MobileNetV2)
IMG_WIDTH = 224   # Width of input images
IMG_CHANNELS = 3  # RGB color channels (Red, Green, Blue)
BATCH_SIZE = 20   # Optimized for single model (increased from 16)
EPOCHS = 25       # Reduced for faster training (7-9 hours target)
VALIDATION_SPLIT = 0.2  # 20% of data used for validation, 80% for training

# Define the 10 skin disease classes
# These match the folder names in MLDemoProj/IMG_CLASSES/
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

print(f"✓ Configuration optimized for fast training:")
print(f"  - Input image size: {IMG_HEIGHT}x{IMG_WIDTH}")
print(f"  - Number of classes: {NUM_CLASSES}")
print(f"  - Batch size: {BATCH_SIZE} (optimized for MobileNet)")
print(f"  - Epochs: {EPOCHS} (reduced for speed)")
print(f"  - Target training time: 7-9 hours")


# ============================================================================
# SECTION 3: DATA LOADING AND PREPROCESSING FUNCTIONS
# ============================================================================

def load_and_preprocess_images(base_path, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Load all images from the dataset and prepare them for training.
    
    Args:
        base_path (str): Path to the root directory containing class folders
        img_size (tuple): Target size to resize all images to (height, width)
    
    Returns:
        images (np.array): Array of preprocessed images [0, 1] range
        labels (np.array): Corresponding class indices for each image
        class_names (list): List of class folder names
    """
    
    print("\n" + "="*70)
    print("STEP 1: LOADING AND PREPROCESSING IMAGES")
    print("="*70)
    
    images = []
    labels = []
    class_names = []
    
    # Get all class folders (one per disease type)
    class_folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])
    
    print(f"\nFound {len(class_folders)} disease classes:")
    
    # Loop through each disease class folder
    for class_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(base_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"  [{class_idx+1}] {class_name}: {len(image_files)} images")
        class_names.append(class_name)
        
        # Load each image in the class folder
        for image_file in image_files:
            try:
                image_path = os.path.join(class_path, image_file)
                
                # Read image using OpenCV (faster than PIL)
                image = cv2.imread(image_path)
                
                # Convert BGR to RGB (OpenCV uses BGR by default)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize image to standard size (224x224)
                image = cv2.resize(image, img_size)
                
                # Normalize pixel values to [0, 1] range
                # This helps the neural network learn better
                image = image.astype('float32') / 255.0
                
                # Add to lists
                images.append(image)
                labels.append(class_idx)
                
            except Exception as e:
                print(f"    ⚠ Error loading {image_file}: {str(e)}")
                continue
    
    # Convert lists to numpy arrays for efficient computation
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  - Total images: {len(images)}")
    print(f"  - Array shape: {images.shape} (batch, height, width, channels)")
    print(f"  - Label range: {labels.min()} to {labels.max()}")
    
    return images, labels, class_names


def compute_class_weights(labels, num_classes):
    """
    Calculate class weights to balance the imbalanced dataset.
    
    WHY THIS MATTERS:
    - Our dataset is imbalanced (1.25k to 3.5k samples per class)
    - Without class weights, the model learns to predict large classes well
      but fails on small classes
    - Class weights give MORE WEIGHT (penalty) to misclassifying rare classes
    
    Formula: weight = total_samples / (num_classes * samples_in_class)
    
    Example:
    - If Melanoma has 3.5k samples and Atopic Dermatitis has 1.25k samples
    - Eczema gets higher weight so the model "cares more" about it
    
    Args:
        labels (np.array): Array of class indices
        num_classes (int): Total number of classes
    
    Returns:
        class_weights (dict): Dictionary mapping class index to its weight
    """
    
    print("\n" + "="*70)
    print("STEP 2: COMPUTING CLASS WEIGHTS FOR IMBALANCED DATA")
    print("="*70)
    
    # Compute weights to balance the classes
    class_weights = compute_class_weight(
        'balanced',  # Method: 'balanced' automatically computes weights
        classes=np.arange(num_classes),
        y=labels
    )
    
    # Convert to dictionary format for model.fit()
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print("\nClass Distribution and Weights:")
    print("-" * 70)
    
    # Show statistics for each class
    for class_idx in range(num_classes):
        count = np.sum(labels == class_idx)
        weight = class_weights_dict[class_idx]
        percentage = (count / len(labels)) * 100
        print(f"  Class {class_idx}: {count:5d} images ({percentage:5.1f}%) | Weight: {weight:.3f}")
    
    print("-" * 70)
    print("✓ Higher weight = model focuses more on that class")
    print(f"✓ Weight range: {min(class_weights):.3f} to {max(class_weights):.3f}")
    
    return class_weights_dict, class_weights


# ============================================================================
# SECTION 4: BUILD MOBILENET MODEL (FAST AND EFFICIENT)
# ============================================================================

def create_mobilenet_model(num_classes=10):
    """
    Build a MobileNet model using ONE pre-trained backbone (optimized for speed).
    
    ARCHITECTURE:
    
    Input (224x224x3)
        ↓
    MobileNetV2 (frozen, pre-trained on ImageNet)
        ↓
    GlobalAveragePooling2D
        ↓
    Dense(128) with ReLU activation
        ↓
    Dropout(0.5) - reduces overfitting
        ↓
    Dense(10) with Softmax - final predictions
    
    WHY MOBILENET-ONLY?
    1. Speed: Designed for mobile devices, runs fast on any hardware
    2. Efficiency: Only ~3.5M parameters vs ~8.8M for ensemble
    3. Memory: Uses less RAM (crucial for your 16GB system)
    4. Transfer Learning: Pre-trained on ImageNet (1.2M images)
    5. Good Performance: Still achieves excellent accuracy for skin diseases
    6. Training Time: ~7-9 hours vs 18-25 hours for ensemble
    
    SPEED OPTIMIZATIONS FOR YOUR HARDWARE (i5 8th gen + 16GB RAM):
    - Single backbone instead of ensemble (50% faster)
    - Batch size increased to 20 (from 16)
    - Epochs reduced to 25 (from 50)
    - Early stopping with patience=3 (aggressive)
    
    Args:
        num_classes (int): Number of output classes (10 for our dataset)
    
    Returns:
        model: Compiled Keras Functional API model
    """
    
    print("\n" + "="*70)
    print("STEP 3: BUILDING MOBILENET MODEL (FAST TRAINING)")
    print("="*70)
    
    # ========================================================================
    # PART A: DEFINE INPUT LAYER
    # ========================================================================
    # This is where images enter the model
    input_tensor = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    print("\n✓ Input layer created: (224, 224, 3)")
    
    # ========================================================================
    # PART B: LOAD PRE-TRAINED MOBILENETV2 (SINGLE BACKBONE)
    # ========================================================================
    # MobileNetV2 is pre-trained on ImageNet (1.2M images, 1000 classes)
    # We use 'weights=imagenet' to load this pre-training
    # include_top=False removes the original 1000-class output layer
    # We'll add our own 10-class output layer
    
    mobile_net = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        include_top=False,  # Remove the original top (1000 classes)
        weights='imagenet'  # Load pre-trained ImageNet weights
    )
    
    # IMPORTANT: Freeze the weights
    # This prevents changing the learned patterns during training
    # We only train the new layers we add on top
    mobile_net.trainable = False
    
    print("✓ MobileNetV2 loaded (pre-trained on ImageNet)")
    print(f"  - Architecture: Lightweight, optimized for speed")
    print(f"  - Parameters: ~3.5M (vs ~8.8M for ensemble)")
    print(f"  - Speed: ~50% faster than ensemble")
    print(f"  - Memory: Much lower RAM usage")
    print(f"  - Frozen: Yes (we use pre-trained weights as feature extractor)")
    
    # Apply MobileNetV2 to input
    x = mobile_net(input_tensor)  # Output shape: (batch, 7, 7, 1280)
    
    # Apply Global Average Pooling
    # This takes the 7x7 feature map and averages all 49 values into 1
    # 7x7x1280 → 1280
    x = layers.GlobalAveragePooling2D()(x)
    print(f"  - After pooling: (batch, 1280)")   # 1280 features from MobileNet
    
    # ========================================================================
    # PART C: ADD DENSE LAYERS FOR CLASSIFICATION
    # ========================================================================
    # Dense layer with 128 neurons and ReLU activation
    # ReLU: Rectified Linear Unit (outputs max(0, x))
    # This adds non-linearity, allowing the model to learn complex patterns
    
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(1e-4)  # L2 regularization to prevent overfitting
    )(x)
    print("\n✓ Dense layer: 128 neurons with ReLU + L2 regularization")
    
    # Dropout layer
    # WHAT IS DROPOUT?
    # During training, randomly "turn off" 50% of neurons
    # This prevents the model from relying too much on specific neurons
    # It's like training multiple smaller models and averaging their predictions
    # Result: Much better generalization and reduced overfitting
    
    x = layers.Dropout(0.5)(x)
    print("✓ Dropout layer: 50% (prevents overfitting)")
    
    # ========================================================================
    # PART D: OUTPUT LAYER - 10-CLASS CLASSIFICATION
    # ========================================================================
    # Dense layer with 10 neurons (one per disease class)
    # Softmax activation: converts 10 numbers into probabilities that sum to 1
    # Example output: [0.05, 0.02, 0.85, 0.01, ...]
    # The highest probability indicates the predicted class
    
    output = layers.Dense(
        num_classes,
        activation='softmax'
    )(x)
    print(f"\n✓ Output layer: {num_classes} neurons with Softmax activation")
    
    # ========================================================================
    # PART E: CREATE THE MODEL
    # ========================================================================
    # Functional API: explicitly specify input and output
    model = models.Model(inputs=input_tensor, outputs=output)
    
    print("\n" + "="*70)
    print("MOBILENET MODEL SUMMARY")
    print("="*70)
    model.summary()  # Print layer-by-layer breakdown
    
    print(f"\n✅ Model optimized for speed:")
    print(f"  - Single backbone vs ensemble (50% faster)")
    print(f"  - Batch size: {BATCH_SIZE} (increased from 16)")
    print(f"  - Target training time: 7-9 hours")
    print(f"  - Memory usage: Significantly reduced")
    print(f"  - Parameters: ~3.5M (much lower than ensemble)")
    
    return model


# ============================================================================
# SECTION 5: COMPILE THE MODEL
# ============================================================================

def compile_model(model):
    """
    Configure the model for training.
    
    Three key components:
    1. Optimizer: How the model learns (Adam is state-of-the-art)
    2. Loss Function: What we're trying to minimize (categorical crossentropy for multi-class)
    3. Metrics: How we measure performance (accuracy)
    
    Args:
        model: The compiled model
    """
    
    print("\n" + "="*70)
    print("STEP 4: COMPILING MODEL")
    print("="*70)
    
    # OPTIMIZER: Adam (Adaptive Moment Estimation)
    # WHY ADAM?
    # - Combines advantages of AdaGrad and RMSprop
    # - Adapts learning rate for each parameter individually
    # - Learning rate=0.001 is the default and works well for most cases
    # - Faster convergence than standard SGD
    
    optimizer = Adam(learning_rate=0.001)
    
    # LOSS FUNCTION: Categorical Crossentropy
    # WHY THIS?
    # - For multi-class classification problems (>2 classes)
    # - Measures difference between predicted probabilities and true labels
    # - Lower loss = better predictions
    
    loss = 'categorical_crossentropy'
    
    # METRICS: Accuracy
    # What we care about: % of correct predictions
    
    metrics = ['accuracy']
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    print("✓ Model compiled successfully!")
    print(f"  - Optimizer: Adam (learning_rate=0.001)")
    print(f"  - Loss: Categorical Crossentropy")
    print(f"  - Metrics: Accuracy")


# ============================================================================
# SECTION 6: TRAIN THE MODEL
# ============================================================================

def train_model(model, images, labels, class_weights, batch_size=20, epochs=25):
    """
    Train the MobileNet model on the skin disease dataset.
    
    OPTIMIZED FOR SPEED (7-9 hour target):
    - Reduced epochs: 25 instead of 50
    - Increased batch size: 20 instead of 16
    - Aggressive early stopping: patience=3 instead of 5
    - Single backbone: MobileNet-only vs ensemble
    
    TRAINING PROCESS:
    1. Shuffle data and split into batches
    2. Forward pass: images through network → predictions
    3. Calculate loss (prediction error)
    4. Backward pass: compute gradients (how to adjust weights)
    5. Update weights to reduce loss
    6. Repeat for all batches (one epoch)
    7. Validate on held-out data
    8. Repeat for multiple epochs
    
    CLASS WEIGHTS:
    - Each misclassification is weighted by class_weight
    - Rare classes (small datasets) get higher weights
    - Model learns to predict rare classes correctly
    
    EARLY STOPPING (AGGRESSIVE FOR SPEED):
    - Monitor validation loss
    - If it doesn't improve for 3 epochs, stop training
    - Prevents overfitting and reduces training time
    
    Args:
        model: The compiled model
        images (np.array): All training images
        labels (np.array): All training labels (one-hot encoded)
        class_weights (dict): Weight for each class
        batch_size (int): Number of images per batch (20 for speed)
        epochs (int): Maximum number of training iterations (25 for speed)
    
    Returns:
        history: Training history (loss, accuracy per epoch)
    """
    
    print("\n" + "="*70)
    print("STEP 5: TRAINING THE MODEL")
    print("="*70)
    
    # Convert labels to one-hot encoding
    # WHY?
    # - MobileNetV2 output: [prob_class_0, prob_class_1, ..., prob_class_9]
    # - Label 3 (Atopic Dermatitis) → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # - This is what categorical crossentropy expects
    
    labels_one_hot = keras.utils.to_categorical(labels, num_classes=10)
    
    print(f"\n✓ Labels converted to one-hot encoding")
    print(f"  - Example: Class 3 → {labels_one_hot[0]}")
    
    # Define Early Stopping callback (AGGRESSIVE FOR SPEED)
    early_stop = EarlyStopping(
        monitor='val_loss',      # Monitor validation loss
        patience=3,              # Stop if no improvement for 3 epochs (reduced from 5)
        restore_best_weights=True,  # Revert to best model
        verbose=1
    )
    
    print(f"\n✓ Early Stopping enabled (aggressive for speed):")
    print(f"  - Monitor: Validation Loss")
    print(f"  - Patience: 3 epochs (reduced from 5)")
    print(f"  - Will restore best model weights")
    
    print(f"\nStarting training for up to {epochs} epochs...")
    print(f"Dataset: {len(images)} total images")
    print(f"Batches per epoch: {len(images) // batch_size}")
    print("-" * 70)
    
    # Train the model
    history = model.fit(
        images,
        labels_one_hot,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,  # Use 20% for validation
        class_weight=class_weights,  # Apply class weights to balance dataset
        callbacks=[early_stop],  # Stop early if overfitting detected
        verbose=1
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    
    return history


# ============================================================================
# SECTION 7: SAVE THE MODEL
# ============================================================================

def save_model(model, save_path='models/skin_disease_mobilenet_model.h5'):
    """
    Save the trained model to disk.
    
    WHY SAVE?
    - Can reuse the model without retraining
    - High-quality models take hours to train
    - Save time and computational resources
    
    Args:
        model: The trained model
        save_path (str): Where to save the model
    """
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the model
    model.save(save_path)
    
    print("\n" + "="*70)
    print("MODEL SAVED")
    print("="*70)
    print(f"✓ Model saved to: {save_path}")
    print(f"✓ File size: {os.path.getsize(save_path) / 1e6:.2f} MB")


# ============================================================================
# SECTION 8: VISUALIZE TRAINING RESULTS
# ============================================================================

def plot_training_history(history, save_path='models/training_history.png'):
    """
    Plot training and validation accuracy/loss results.
    
    WHAT TO LOOK FOR:
    1. Both curves decreasing: Model learning correctly
    2. Training loss ↓ but validation loss ↑: Overfitting
    3. Both plateauing: Model has converged
    
    Args:
        history: Training history from model.fit()
        save_path (str): Where to save the plot
    """
    
    print("\n" + "="*70)
    print("STEP 6: VISUALIZING TRAINING RESULTS")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to: {save_path}")
    
    plt.show()


# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("SKIN DISEASE CLASSIFICATION - FULL TRAINING PIPELINE")
    print("="*70)
    
    # Load and preprocess data
    images, labels, class_names = load_and_preprocess_images(BASE_DIR)
    
    # Compute class weights for imbalanced data
    class_weights_dict, _ = compute_class_weights(labels, NUM_CLASSES)
    
    # Build the MobileNet model (fast training)
    model = create_mobilenet_model(num_classes=NUM_CLASSES)
    
    # Compile the model
    compile_model(model)
    
    # Train the model
    history = train_model(
        model,
        images,
        labels,
        class_weights_dict,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    
    # Save the trained model
    save_model(model)
    
    # Visualize training results
    plot_training_history(history)
    
    print("\n" + "="*70)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run 'streamlit run app.py' to see interactive predictions")
    print("2. Use 'test.py' to evaluate on test data")
    print("3. Model saved in 'models/skin_disease_mobilenet_model.h5'")
    print("="*70 + "\n")
