"""
Optimized Test Script for Balanced Dataset
Updated to work with balanced dataset and balanced model
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ModelTester:
    def __init__(self, model_path="models/balanced_resnet50_finetuned.keras", dataset_path="balanced_dataset/test"):
        self.model_path = model_path
        self.dataset_path = Path(dataset_path)
        self.input_shape = (224, 224)
        self.batch_size = 32  # Larger batch size for bigger test set
        
    def load_model_and_data(self):
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found at {self.model_path}")
            print("💡 Available models:")
            models_dir = Path("models")
            if models_dir.exists():
                for model_file in models_dir.glob("*.keras"):
                    print(f"   - {model_file.name}")
            raise FileNotFoundError(f"Run train.py first to create the balanced model.")
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Balanced test data not found at {self.dataset_path}. Run simple_rebalance.py first.")

        print(f"Loading model from {self.model_path}...")
        model = tf.keras.models.load_model(self.model_path, compile=False)

        print("Creating test generator with rescaling...")
        # Add rescaling to match training preprocessing
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.input_shape,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False # CRITICAL: Must be False for evaluation
        )
        
        return model, test_generator

    def evaluate(self):
        model, test_generator = self.load_model_and_data()
        
        # Analyze test set balance
        print("\n📊 BALANCED TEST SET ANALYSIS:")
        print("-" * 40)
        class_counts = {}
        for i, class_name in enumerate(test_generator.class_indices.keys()):
            count = np.sum(test_generator.classes == i)
            class_counts[class_name] = count
            print(f"   Class {class_name}: {count} samples")
        
        # Calculate balance ratio
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        balance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"\n   Balance ratio: {balance_ratio:.1f}:1")
        
        if balance_ratio < 2:
            print("   ✅ Excellent balance in test set")
        elif balance_ratio < 3:
            print("   🟢 Good balance in test set") 
        else:
            print("   🟡 Moderate balance in test set")
        
        print("\nGenerating predictions...")
        predictions = model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())

        accuracy = accuracy_score(true_classes, predicted_classes)
        
        print("\n" + "="*60)
        print(f"🎯 BALANCED DATASET TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print("="*60)
        
        # Per-class accuracy analysis
        print("\n📋 PER-CLASS PERFORMANCE (Balanced Dataset):")
        print("-" * 50)
        for i, class_name in enumerate(class_labels):
            class_mask = (true_classes == i)
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(true_classes[class_mask], predicted_classes[class_mask])
                class_count = np.sum(class_mask)
                status = "🟢" if class_acc > 0.7 else "🟡" if class_acc > 0.5 else "🔴"
                print(f"   {status} Class {class_name}: {class_acc:.3f} ({class_count} samples)")
        
        print("\nDETAILED CLASSIFICATION REPORT:")
        print(classification_report(true_classes, predicted_classes, target_names=class_labels))

        self.plot_confusion_matrix(true_classes, predicted_classes, class_labels)
        
        # Summary for balanced dataset
        print(f"\n💡 BALANCED DATASET RESULTS SUMMARY:")
        print(f"   • Total accuracy: {accuracy*100:.1f}%")
        print(f"   • Test set balance: {balance_ratio:.1f}:1")
        print(f"   • Classes tested: {len(class_labels)}")
        print(f"   • Total test samples: {len(true_classes)}")
        
        if accuracy > 0.8:
            print("   🏆 Excellent performance! Balanced dataset worked well.")
        elif accuracy > 0.6:
            print("   ✅ Good improvement with balanced dataset!")
        elif accuracy > 0.4:
            print("   📈 Better than original imbalanced dataset.")
        else:
            print("   ⚠️ Performance still low - check data quality.")

    def plot_confusion_matrix(self, true_classes, predicted_classes, class_labels):
        cm = confusion_matrix(true_classes, predicted_classes)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels)
        
        plt.title('Balanced Dataset - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Calculate and display accuracy on plot
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.3f}', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        os.makedirs('models', exist_ok=True)
        plt.savefig('models/balanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\n📊 Confusion matrix saved to: models/balanced_confusion_matrix.png")

if __name__ == "__main__":
    print("🧪 BALANCED DATASET MODEL TESTING")
    print("=" * 60)
    
    # Check if balanced dataset exists
    from pathlib import Path
    
    if not Path("balanced_dataset").exists():
        print("❌ Balanced dataset not found!")
        print("📋 Please run simple_rebalance.py first to create balanced dataset")
        print("   python simple_rebalance.py")
        exit(1)
    
    if not Path("balanced_dataset/test").exists():
        print("❌ Balanced test set not found!")
        print("📋 Please ensure simple_rebalance.py created the test set")
        exit(1)
    
    print("🚀 Testing model trained on balanced dataset...")
    print("📊 This will analyze performance on balanced test data")
    
    try:
        tester = ModelTester()
        tester.evaluate()
        
        print(f"\n🎉 Balanced dataset testing completed successfully!")
        print(f"📈 Check the confusion matrix for detailed class performance")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure you've trained the balanced model: python train.py")
        print("   2. Verify balanced dataset exists: python simple_rebalance.py")
        print("   3. Check if model file exists in models/ directory")