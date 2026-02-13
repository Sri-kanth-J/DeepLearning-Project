import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2
from PIL import Image

from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print("✓ All libraries imported successfully!")

BASE_DIR = Path('MLDemoProj/IMG_CLASSES')

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
BATCH_SIZE = 20
EPOCHS = 50  # Increased for better training
VALIDATION_SPLIT = 0.2

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

print(f"✓ IMPROVED CONFIGURATION:")
print(f"  - Epochs: {EPOCHS} (increased from 25)")
print(f"  - Progressive training: Frozen → Fine-tuned")
print(f"  - Better early stopping: Patience=10")
print(f"  - Learning rate scheduling enabled")
print(f"  - Enhanced data augmentation")


def create_enhanced_data_generators(base_path):
    """Enhanced data generators with stronger augmentation."""
    
    print("\n" + "="*70)
    print("STEP 1: ENHANCED DATA GENERATORS")
    print("="*70)
    
    # Stronger data augmentation for better generalization
    train_datagen = ImageDataGenerator(
        rescale=1./255.0,
        rotation_range=25,         # Increased rotation
        width_shift_range=0.15,    # More translation
        height_shift_range=0.15,
        shear_range=0.1,           # Added shear
        zoom_range=0.1,            # Added zoom
        horizontal_flip=True,
        vertical_flip=False,       # Medical images shouldn't be flipped vertically
        brightness_range=[0.8, 1.2], # Brightness variation
        fill_mode='nearest',
        validation_split=VALIDATION_SPLIT
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255.0,
        validation_split=VALIDATION_SPLIT
    )
    
    train_generator = train_datagen.flow_from_directory(
        base_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    validation_generator = val_datagen.flow_from_directory(
        base_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    class_names = list(train_generator.class_indices.keys())
    class_names.sort()
    
    print(f"✓ Enhanced data augmentation enabled!")
    print(f"  - Rotation: ±25° (vs ±15°)")
    print(f"  - Translation: ±15% (vs ±10%)")  
    print(f"  - Added: shear, zoom, brightness variation")
    print(f"  - Training samples: {train_generator.samples}")
    print(f"  - Validation samples: {validation_generator.samples}")
    
    return train_generator, validation_generator, class_names


def compute_class_weights_from_generators(train_generator, validation_generator, num_classes):
    """Compute class weights for imbalanced dataset."""
    
    print("\n" + "="*70)
    print("STEP 2: CLASS WEIGHT COMPUTATION")
    print("="*70)
    
    all_labels = np.concatenate([train_generator.classes, validation_generator.classes])
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(num_classes),
        y=all_labels
    )
    
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print("\nClass Distribution and Weights:")
    for class_idx in range(num_classes):
        count = np.sum(all_labels == class_idx)
        weight = class_weights_dict[class_idx]
        percentage = (count / len(all_labels)) * 100
        print(f"  Class {class_idx}: {count:5d} images ({percentage:5.1f}%) | Weight: {weight:.3f}")
    
    return class_weights_dict, class_weights


def create_improved_mobilenet_model(num_classes=10):
    """Create improved MobileNet model with better activations."""
    
    print("\n" + "="*70)
    print("STEP 3: IMPROVED MOBILENET MODEL")
    print("="*70)
    
    input_tensor = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    # MobileNetV2 backbone
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        include_top=False,
        weights='imagenet'
    )
    
    # Start with frozen backbone (will unfreeze later)
    base_model.trainable = False
    
    print("✓ MobileNetV2 backbone loaded")
    print(f"  - Initial state: FROZEN (feature extraction)")
    print(f"  - Will be unfrozen after initial training")
    
    x = base_model(input_tensor)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Improved classification head with Swish activation
    x = layers.Dense(
        256,  # Increased from 128
        activation='swish',  # Better than ReLU for deep networks
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name='dense_1'
    )(x)
    print("✓ Dense layer 1: 256 neurons with Swish activation")
    
    x = layers.BatchNormalization(name='bn_1')(x)
    print("✓ Batch normalization added")
    
    x = layers.Dropout(0.3, name='dropout_1')(x)  # Reduced dropout
    
    # Second dense layer for better representation
    x = layers.Dense(
        128,
        activation='swish',
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name='dense_2'
    )(x)
    print("✓ Dense layer 2: 128 neurons with Swish activation")
    
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(0.2, name='dropout_2')(x)
    
    # Output layer
    output = layers.Dense(
        num_classes,
        activation='softmax',
        name='predictions'
    )(x)
    print(f"✓ Output layer: {num_classes} classes with Softmax")
    
    model = models.Model(inputs=input_tensor, outputs=output)
    
    print("\n" + "="*70)
    print("IMPROVED MODEL SUMMARY")
    print("="*70)
    model.summary()
    
    print(f"\n✅ Model improvements:")
    print(f"  - Deeper classification head (256 → 128 → {num_classes})")
    print(f"  - Swish activation (better than ReLU)")
    print(f"  - Batch normalization for stable training")
    print(f"  - Progressive training strategy")
    
    return model, base_model


def compile_model_progressive(model, phase="initial"):
    """Compile model with different settings for different training phases."""
    
    print(f"\n✓ Compiling model for {phase} phase")
    
    if phase == "initial":
        # Higher learning rate for initial training
        optimizer = Adam(learning_rate=0.001)
        print("  - Learning rate: 0.001 (initial training)")
    else:
        # Lower learning rate for fine-tuning
        optimizer = Adam(learning_rate=0.0001)
        print("  - Learning rate: 0.0001 (fine-tuning)")
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']  # Using standard accuracy metric for compatibility
    )


def create_callbacks():
    """Create improved callbacks for training."""
    
    print("\n" + "="*70)
    print("STEP 4: SETTING UP TRAINING CALLBACKS")
    print("="*70)
    
    # Less aggressive early stopping
    early_stop = EarlyStopping(
        monitor='val_accuracy',  # Monitor accuracy instead of loss
        patience=10,             # Increased from 3
        restore_best_weights=True,
        verbose=1,
        mode='max'
    )
    
    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Model checkpointing
    checkpoint = ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
    
    print("✓ Callbacks configured:")
    print("  - Early Stopping: patience=10 (vs 3), monitor=val_accuracy")
    print("  - Learning Rate Reduction: factor=0.2, patience=5")
    print("  - Model Checkpoint: saves best model automatically")
    
    return [early_stop, reduce_lr, checkpoint]


def train_progressive(model, base_model, train_generator, validation_generator, class_weights_dict):
    """Progressive training: frozen backbone → fine-tuning."""
    
    print("\n" + "="*70)
    print("STEP 5: PROGRESSIVE TRAINING STRATEGY")
    print("="*70)
    
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = validation_generator.samples // BATCH_SIZE
    callbacks = create_callbacks()
    
    # Phase 1: Train with frozen backbone
    print("\n🔥 PHASE 1: TRAINING WITH FROZEN BACKBONE")
    print("-" * 50)
    
    compile_model_progressive(model, "initial")
    
    history_1 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=20,  # Initial training
        validation_data=validation_generator,
        validation_steps=validation_steps,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ Phase 1 completed!")
    
    # Phase 2: Unfreeze and fine-tune
    print("\n🔥 PHASE 2: FINE-TUNING WITH UNFROZEN BACKBONE")
    print("-" * 50)
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Fine-tune from the top layers only
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    print(f"✓ Unfrozen backbone: {sum(1 for layer in base_model.layers if layer.trainable)} trainable layers")
    
    # Recompile with lower learning rate
    compile_model_progressive(model, "fine_tuning")
    
    # Continue training
    history_2 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=30,  # Additional fine-tuning
        validation_data=validation_generator,
        validation_steps=validation_steps,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ Phase 2 completed!")
    
    # Combine histories
    history = {
        'loss': history_1.history['loss'] + history_2.history['loss'],
        'accuracy': history_1.history['accuracy'] + history_2.history['accuracy'],
        'val_loss': history_1.history['val_loss'] + history_2.history['val_loss'],
        'val_accuracy': history_1.history['val_accuracy'] + history_2.history['val_accuracy']
    }
    
    return history


def plot_enhanced_training_history(history):
    """Plot comprehensive training history."""
    
    print("\n" + "="*70)
    print("PLOTTING TRAINING RESULTS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0,0].plot(history['accuracy'], 'b-', label='Training Accuracy')
    axes[0,0].plot(history['val_accuracy'], 'r-', label='Validation Accuracy')
    axes[0,0].set_title('Model Accuracy')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Loss
    axes[0,1].plot(history['loss'], 'b-', label='Training Loss')
    axes[0,1].plot(history['val_loss'], 'r-', label='Validation Loss')
    axes[0,1].set_title('Model Loss')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Learning curves
    axes[1,0].plot(history['accuracy'], 'b-', alpha=0.7)
    axes[1,0].plot(history['val_accuracy'], 'r-', alpha=0.7)
    axes[1,0].axvline(x=20, color='g', linestyle='--', label='Fine-tuning starts')
    axes[1,0].set_title('Training Phases')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Final metrics
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    best_val_acc = max(history['val_accuracy'])
    
    axes[1,1].bar(['Training', 'Validation', 'Best Validation'], 
                  [final_train_acc, final_val_acc, best_val_acc],
                  color=['blue', 'red', 'green'])
    axes[1,1].set_title('Final Performance')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].set_ylim(0, 1)
    
    for i, v in enumerate([final_train_acc, final_val_acc, best_val_acc]):
        axes[1,1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('training_history_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Training completed with improved strategy!")
    print(f"  - Final training accuracy: {final_train_acc:.3f}")
    print(f"  - Final validation accuracy: {final_val_acc:.3f}")
    print(f"  - Best validation accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 IMPROVED SKIN DISEASE CLASSIFICATION TRAINING")
    print("=" * 70)
    print("Improvements:")
    print("✓ Progressive training (frozen → fine-tuned)")
    print("✓ Better early stopping (patience=10)")
    print("✓ Swish activation functions")
    print("✓ Enhanced data augmentation")
    print("✓ Learning rate scheduling")
    print("✓ Deeper classification head")
    print("=" * 70)
    
    # Main training pipeline
    train_generator, validation_generator, class_names = create_enhanced_data_generators(BASE_DIR)
    
    class_weights_dict, _ = compute_class_weights_from_generators(
        train_generator, validation_generator, NUM_CLASSES
    )
    
    model, base_model = create_improved_mobilenet_model(num_classes=NUM_CLASSES)
    
    history = train_progressive(
        model, base_model, train_generator, validation_generator, class_weights_dict
    )
    
    # Save final model
    os.makedirs('models', exist_ok=True)
    model.save('models/skin_disease_improved_model.h5')
    
    plot_enhanced_training_history(history)
    
    print("\n" + "="*70)
    print("🎉 IMPROVED TRAINING COMPLETED!")
    print("="*70)
    print("Key improvements implemented:")
    print("✓ 50 total epochs (20 frozen + 30 fine-tuned)")
    print("✓ Progressive training strategy")
    print("✓ Better activation functions (Swish)")
    print("✓ Enhanced data augmentation")
    print("✓ Smarter early stopping")
    print("✓ Learning rate scheduling")
    print("\nExpected improvements:")
    print("📈 Higher accuracy (target: 75-85%)")
    print("🎯 Better generalization")
    print("⚡ More stable training")
    print("="*70)