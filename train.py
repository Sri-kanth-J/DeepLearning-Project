"""
Optimized ResNet50 Training Script for Medical Image Classification
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

class ResNetSkinClassifier:
    def __init__(self, data_dir="balanced_dataset", input_shape=(224, 224, 3)):
        self.data_dir = Path(data_dir)
        self.input_shape = input_shape
        self.batch_size = 32  # Larger batch size for 3000 samples per class
        self.initial_epochs = 30  # More epochs for larger dataset
        self.finetune_epochs = 20
        
    def load_data(self):
        print("Loading dataset and applying augmentations...")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Balanced dataset directory '{self.data_dir}' not found. Run simple_rebalance.py first to create the balanced dataset.")

        # Since dataset already has extensive augmentation, use lighter training augmentation
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,  # Reduced since dataset is already augmented
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # Validation generator (add rescaling)
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        train_gen = train_datagen.flow_from_directory(
            self.data_dir / "train",
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        val_gen = val_datagen.flow_from_directory(
            self.data_dir / "val",
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"Classes found: {train_gen.num_classes}")
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Samples per class (approx): {train_gen.samples // train_gen.num_classes}")
        return train_gen, val_gen

    def build_model(self, num_classes):
        print("Building ResNet50 model...")
        
        base_model = ResNet50(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        inputs = Input(shape=self.input_shape)
        
        # Since we're using rescale=1./255 in data generator, don't use preprocess_input
        x = base_model(inputs, training=False)
        
        # Enhanced architecture for larger balanced dataset
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Larger dense layers for more complex patterns
        x = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model, base_model

    def train(self):
        train_gen, val_gen = self.load_data()
        num_classes = train_gen.num_classes
        model, base_model = self.build_model(num_classes)

        # For balanced dataset with 3000 samples per class, analyze balance and use minimal weights
        class_counts = {}
        for i in range(train_gen.num_classes):
            class_counts[i] = np.sum(train_gen.classes == i)
        
        print(f"\n📊 Class distribution:")
        for class_id, count in class_counts.items():
            print(f"   Class {class_id}: {count} samples")
        
        # Calculate balance ratio
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        balance_ratio = max_count / min_count if min_count > 0 else 1
        
        print(f"\n⚖️ Balance ratio: {balance_ratio:.2f}:1")
        
        if balance_ratio < 1.5:
            print("✅ Excellent balance - skipping class weights for faster training")
            class_weight_dict = None  # No class weights needed for perfectly balanced data
        else:
            print("🟡 Some imbalance detected - using light class weights")
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_gen.classes),
                y=train_gen.classes
            )
            # Cap weights for balanced dataset
            class_weights = np.clip(class_weights, 0.5, 2.0)
            class_weight_dict = dict(enumerate(class_weights))
            print(f"   Applied weights: {class_weight_dict}")

        # --- Phase 1: Train Top Layers ---
        print("\nPhase 1: Transfer Learning...")
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Optimized callbacks for balanced dataset - Phase 1 (Transfer Learning)
        callbacks_phase1 = [
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=6, min_lr=1e-6, verbose=1),  # More patience for 3000-sample classes
            EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1),  # Increased patience for balanced data
            ModelCheckpoint('models/checkpoints/balanced_resnet50_transfer.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
        ]

        # Create directories for balanced dataset training
        os.makedirs('models/checkpoints', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        history1 = model.fit(
            train_gen,
            epochs=self.initial_epochs,
            validation_data=val_gen,
            callbacks=callbacks_phase1,
            class_weight=class_weight_dict,
            verbose=1
        )

        # --- Phase 2: Fine-Tuning ---
        print("\nPhase 2: Fine-tuning top convolutional blocks...")
        base_model.trainable = True
        
        # Freeze all layers except the last 30
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=Adam(learning_rate=5e-5), # Lower LR for fine-tuning
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Optimized callbacks for balanced dataset - Phase 2 (Fine-tuning)
        callbacks_phase2 = [
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-7, verbose=1),  # Conservative reduction for fine-tuning
            EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),  # Sufficient patience for convergence
            ModelCheckpoint('models/checkpoints/balanced_resnet50_finetuned.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
        ]

        history2 = model.fit(
            train_gen,
            epochs=self.finetune_epochs,
            validation_data=val_gen,
            callbacks=callbacks_phase2,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        print("\nTraining completed. Best model saved to 'models/balanced_resnet50_finetuned.keras'")
        print("\n📊 Training used balanced dataset - should see much better performance!")
        return model

if __name__ == "__main__":
    # Check if balanced dataset exists
    from pathlib import Path
    if not Path("balanced_dataset").exists():
        print("❌ Balanced dataset not found!")
        print("📋 Please run simple_rebalance.py first to create balanced dataset")
        print("   python simple_rebalance.py")
        exit(1)
    
    print("🚀 Training with balanced dataset...")
    classifier = ResNetSkinClassifier()
    classifier.train()