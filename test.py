import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import warnings
warnings.filterwarnings('ignore')

print("✓ Libraries imported successfully!")

# Configuration
BASE_DIR = Path('MLDemoProj/IMG_CLASSES')
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 20

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


def load_improved_model():
    """Load the specified model for testing."""
    
    print("\n" + "="*70)
    print("LOADING SPECIFIED MODEL")
    print("="*70)
    
    # Test specific model path as requested
    model_path = 'models/best_model.h5'
    
    if os.path.exists(model_path):
        print(f"✓ Loading model from: {model_path}")
        try:
            model = keras.models.load_model(model_path)
            print(f"✓ Model loaded successfully!")
            
            # Print model info
            total_params = model.count_params()
            trainable_params = sum([keras.backend.count_params(w) for w in model.trainable_weights])
            
            print(f"  - Total parameters: {total_params:,}")
            print(f"  - Trainable parameters: {trainable_params:,}")
            print(f"  - Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
            
            return model, model_path
            
        except Exception as e:
            print(f"✗ Failed to load {model_path}: {e}")
            raise FileNotFoundError(f"Failed to load specified model: {model_path}")
    else:
        print(f"✗ Model file not found: {model_path}")
        # Also try alternative model locations if primary not found
        fallback_paths = [
            'models/skin_disease_improved_model.h5',
            'models/skin_disease_mobilenet_model.h5',
            'models/old1/best_model.h5'
        ]
        
        print("\n🔍 Trying fallback locations:")
        for fallback_path in fallback_paths:
            if os.path.exists(fallback_path):
                print(f"✓ Found fallback model: {fallback_path}")
                try:
                    model = keras.models.load_model(fallback_path)
                    print(f"✓ Fallback model loaded successfully!")
                    
                    total_params = model.count_params()
                    trainable_params = sum([keras.backend.count_params(w) for w in model.trainable_weights])
                    
                    print(f"  - Total parameters: {total_params:,}")
                    print(f"  - Trainable parameters: {trainable_params:,}")
                    print(f"  - Model size: {os.path.getsize(fallback_path) / (1024*1024):.2f} MB")
                    
                    return model, fallback_path
                    
                except Exception as e:
                    print(f"✗ Failed to load {fallback_path}: {e}")
                    continue
        
        raise FileNotFoundError(f"No model found at {model_path} or fallback locations!")


def create_test_data_generator(base_path):
    """Create test data generator for memory-efficient evaluation."""
    
    print("\n" + "="*70)
    print("CREATING TEST DATA GENERATOR")
    print("="*70)
    
    # No augmentation for testing - just rescaling
    test_datagen = ImageDataGenerator(rescale=1./255.0)
    
    test_generator = test_datagen.flow_from_directory(
        base_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,  # Important: don't shuffle for consistent results
        seed=42
    )
    
    class_names = list(test_generator.class_indices.keys())
    class_names.sort()
    
    print(f"✓ Test generator created!")
    print(f"  - Total test samples: {test_generator.samples:,}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Steps: {test_generator.samples // BATCH_SIZE}")
    print(f"  - Classes: {len(class_names)}")
    
    return test_generator, class_names


def comprehensive_model_evaluation(model, test_generator, class_names):
    """Comprehensive evaluation of the improved model."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Calculate steps needed
    steps = test_generator.samples // BATCH_SIZE
    
    print(f"📊 Evaluating model on {test_generator.samples:,} images...")
    print(f"⚡ Processing in {steps} batches of {BATCH_SIZE} images each")
    
    # Get predictions
    print("\n🔮 Generating predictions...")
    predictions = model.predict(test_generator, steps=steps, verbose=1)
    
    # Get true labels
    true_labels = test_generator.classes[:len(predictions)]
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Basic metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None, labels=range(NUM_CLASSES)
    )
    
    print(f"\n📈 OVERALL PERFORMANCE:")
    print(f"  - Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  - Average Precision: {np.mean(precision):.4f}")
    print(f"  - Average Recall: {np.mean(recall):.4f}")
    print(f"  - Average F1-Score: {np.mean(f1):.4f}")
    
    return predictions, true_labels, predicted_labels, accuracy, precision, recall, f1, support


def detailed_class_analysis(true_labels, predicted_labels, class_names, precision, recall, f1, support):
    """Detailed per-class performance analysis."""
    
    print("\n" + "="*70)
    print("DETAILED CLASS-WISE PERFORMANCE")
    print("="*70)
    
    print(f"\n{'Class':<35} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 75)
    
    class_performance = []
    
    for i in range(len(class_names)):
        class_name = class_names[i]
        
        # Truncate long class names for display
        display_name = class_name if len(class_name) <= 30 else class_name[:27] + "..."
        
        print(f"{display_name:<35} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1[i]:<10.3f} {support[i]:<8}")
        
        class_performance.append({
            'class': class_name,
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        })
    
    print("-" * 75)
    
    # Find best and worst performing classes
    best_class_idx = np.argmax(f1)
    worst_class_idx = np.argmin(f1)
    
    print(f"\n🏆 BEST PERFORMING CLASS:")
    print(f"  - {class_names[best_class_idx]}")
    print(f"  - F1-Score: {f1[best_class_idx]:.3f}")
    
    print(f"\n⚠️ WORST PERFORMING CLASS:")
    print(f"  - {class_names[worst_class_idx]}")
    print(f"  - F1-Score: {f1[worst_class_idx]:.3f}")
    
    return class_performance


def create_confusion_matrix_analysis(true_labels, predicted_labels, class_names):
    """Create and analyze confusion matrix with improved visualization."""
    
    print("\n" + "="*70)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*70)
    
    # Create shorter, cleaner labels for better visualization
    short_labels = []
    label_mapping = {}
    
    for i, class_name in enumerate(class_names):
        if "Eczema" in class_name:
            short_label = "Eczema"
        elif "Melanoma" in class_name:
            short_label = "Melanoma"
        elif "Atopic" in class_name:
            short_label = "Atopic Derm."
        elif "Basal Cell" in class_name:
            short_label = "BCC"
        elif "Melanocytic" in class_name:
            short_label = "Nevi"
        elif "Benign Keratosis" in class_name:
            short_label = "Ben. Kerat."
        elif "Psoriasis" in class_name:
            short_label = "Psoriasis"
        elif "Seborrheic" in class_name:
            short_label = "Seb. Kerat."
        elif "Tinea" in class_name or "Ringworm" in class_name:
            short_label = "Fungal Inf."
        elif "Warts" in class_name or "Molluscum" in class_name:
            short_label = "Warts/Mol."
        else:
            # Fallback: use first 10 characters
            short_label = class_name.split('.')[1].strip()[:10] if '.' in class_name else class_name[:10]
        
        short_labels.append(short_label)
        label_mapping[short_label] = class_name
        
    print(f"\nLabel Mapping:")
    for short, full in label_mapping.items():
        print(f"  {short:<12} → {full}")
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Calculate normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create improved visualization
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Raw confusion matrix
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                xticklabels=short_labels,
                yticklabels=short_labels,
                ax=axes[0],
                cbar_kws={'shrink': 0.8},
                square=True)
    
    axes[0].set_title('Confusion Matrix - Raw Counts', fontsize=16, fontweight='bold', pad=20)
    axes[0].set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    axes[0].set_xticklabels(short_labels, rotation=45, ha='right')
    axes[0].set_yticklabels(short_labels, rotation=0)
    
    # Normalized confusion matrix
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.2f', 
                cmap='Oranges',
                xticklabels=short_labels,
                yticklabels=short_labels,
                ax=axes[1],
                cbar_kws={'shrink': 0.8},
                square=True,
                vmin=0.0,
                vmax=1.0)
    
    axes[1].set_title('Confusion Matrix - Normalized (%)', fontsize=16, fontweight='bold', pad=20)
    axes[1].set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    axes[1].set_xticklabels(short_labels, rotation=45, ha='right')
    axes[1].set_yticklabels(short_labels, rotation=0)
    
    # Adjust layout and spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.95, wspace=0.3)
    
    # Save with high quality
    plt.savefig('confusion_matrix_improved.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # Create a separate detailed confusion matrix as a table
    print(f"\n📋 DETAILED CONFUSION MATRIX:")
    print("-" * 100)
    
    # Create a pandas DataFrame for better formatting
    cm_df = pd.DataFrame(cm, index=short_labels, columns=short_labels)
    cm_norm_df = pd.DataFrame(cm_normalized, index=short_labels, columns=short_labels)
    
    print("Raw Counts:")
    print(cm_df.to_string())
    
    print("\n\nNormalized (%):")
    print((cm_norm_df * 100).round(1).to_string())
    
    # Print accuracy per class
    print(f"\n📊 PER-CLASS ACCURACY:")
    print("-" * 50)
    diagonal = np.diag(cm)
    row_sums = cm.sum(axis=1)
    class_accuracies = diagonal / row_sums
    
    for i, (short_label, accuracy) in enumerate(zip(short_labels, class_accuracies)):
        print(f"  {short_label:<12}: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    overall_accuracy = diagonal.sum() / cm.sum()
    print(f"\n  Overall Acc.: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
    print("-" * 50)
    
    # Analyze common misclassifications
    print("\n🔍 COMMON MISCLASSIFICATIONS:")
    
    # Find most confused pairs
    misclassification_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 0:
                misclassification_pairs.append((i, j, cm[i][j]))
    
    # Sort by frequency of misclassification
    misclassification_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 5 misclassification patterns:")
    for idx, (true_idx, pred_idx, count) in enumerate(misclassification_pairs[:5]):
        true_class = short_labels[true_idx]
        pred_class = short_labels[pred_idx]
        percentage = (count / cm[true_idx].sum()) * 100
        print(f"  {idx+1}. {true_class:<12} → {pred_class:<12}: {count:3d} cases ({percentage:.1f}%)")
    
    return cm, cm_normalized, short_labels
    
    # Analyze common misclassifications
    print("\n🔍 COMMON MISCLASSIFICATIONS:")
    
    # Find most confused pairs
    misclassification_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 0:
                misclassification_pairs.append((i, j, cm[i][j]))
    
    # Sort by frequency of misclassification
    misclassification_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 5 misclassification patterns:")
    for i, (true_idx, pred_idx, count) in enumerate(misclassification_pairs[:5]):
        true_class = class_names[true_idx].split('.')[1].strip() if '.' in class_names[true_idx] else class_names[true_idx]
        pred_class = class_names[pred_idx].split('.')[1].strip() if '.' in class_names[pred_idx] else class_names[pred_idx]
        percentage = (count / cm[true_idx].sum()) * 100
        print(f"  {i+1}. {true_class} → {pred_class}: {count} cases ({percentage:.1f}%)")


def confidence_analysis(predictions, true_labels, predicted_labels):
    """Analyze prediction confidence and reliability."""
    
    print("\n" + "="*70)
    print("PREDICTION CONFIDENCE ANALYSIS")
    print("="*70)
    
    # Get confidence scores (max probability for each prediction)
    confidence_scores = np.max(predictions, axis=1)
    
    # Separate correct and incorrect predictions
    correct_mask = (true_labels == predicted_labels)
    
    correct_confidences = confidence_scores[correct_mask]
    incorrect_confidences = confidence_scores[~correct_mask]
    
    print(f"\n📊 CONFIDENCE STATISTICS:")
    print(f"  - Average confidence (correct predictions): {np.mean(correct_confidences):.3f}")
    print(f"  - Average confidence (incorrect predictions): {np.mean(incorrect_confidences):.3f}")
    print(f"  - Overall average confidence: {np.mean(confidence_scores):.3f}")
    
    # Confidence distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green')
    plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', color='red')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confidence vs Accuracy
    confidence_thresholds = np.arange(0.1, 1.0, 0.1)
    accuracies_at_threshold = []
    coverage_at_threshold = []
    
    for threshold in confidence_thresholds:
        high_confidence_mask = confidence_scores >= threshold
        if np.sum(high_confidence_mask) > 0:
            accuracy_at_threshold = np.mean(correct_mask[high_confidence_mask])
            coverage_at_threshold = np.mean(high_confidence_mask)
        else:
            accuracy_at_threshold = 0
            coverage_at_threshold = 0
        
        accuracies_at_threshold.append(accuracy_at_threshold)
        coverage_at_threshold.append(coverage_at_threshold)
    
    plt.subplot(1, 2, 2)
    plt.plot(confidence_thresholds, accuracies_at_threshold, 'b-', label='Accuracy', marker='o')
    plt.plot(confidence_thresholds, coverage_at_threshold, 'r--', label='Coverage', marker='s')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    plt.title('Accuracy vs Coverage at Different Confidence Thresholds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('confidence_analysis_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find high-confidence mistakes
    high_conf_wrong = confidence_scores[(~correct_mask) & (confidence_scores > 0.8)]
    print(f"\n⚠️ HIGH-CONFIDENCE MISTAKES: {len(high_conf_wrong)} predictions")
    if len(high_conf_wrong) > 0:
        print(f"   - These are cases where model was very confident (>80%) but wrong")
        print(f"   - Average confidence of mistakes: {np.mean(high_conf_wrong):.3f}")


def generate_performance_report(model_path, accuracy, class_performance):
    """Generate comprehensive performance report."""
    
    print("\n" + "="*70)
    print("📋 COMPREHENSIVE PERFORMANCE REPORT")
    print("="*70)
    
    report = f"""
IMPROVED SKIN DISEASE CLASSIFICATION MODEL - EVALUATION REPORT
============================================================

Model Information:
- Model file: {model_path}
- Architecture: Improved MobileNetV2 with progressive training
- Evaluation date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Overall Performance:
- Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)

Class-wise Performance:
"""
    
    for class_info in class_performance:
        report += f"- {class_info['class']}: F1={class_info['f1']:.3f}, "
        report += f"Precision={class_info['precision']:.3f}, "
        report += f"Recall={class_info['recall']:.3f}, "
        report += f"Support={class_info['support']}\n"
    
    # Calculate weighted averages
    total_support = sum([c['support'] for c in class_performance])
    weighted_f1 = sum([c['f1'] * c['support'] for c in class_performance]) / total_support
    weighted_precision = sum([c['precision'] * c['support'] for c in class_performance]) / total_support
    weighted_recall = sum([c['recall'] * c['support'] for c in class_performance]) / total_support
    
    report += f"""
Weighted Averages:
- Weighted F1-Score: {weighted_f1:.4f}
- Weighted Precision: {weighted_precision:.4f}
- Weighted Recall: {weighted_recall:.4f}

Performance Analysis:
- This model shows {'significant' if accuracy > 0.75 else 'moderate' if accuracy > 0.60 else 'limited'} improvement
- Best performing classes: {sorted(class_performance, key=lambda x: x['f1'], reverse=True)[:3]}
- Areas for improvement: {sorted(class_performance, key=lambda x: x['f1'])[:3]}

Recommendations:
{'✅ Model is production-ready' if accuracy > 0.75 else 
 '⚠️ Model needs further tuning' if accuracy > 0.60 else 
 '❌ Model needs significant improvement'}
"""
    
    # Save report
    with open('improved_model_evaluation_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\n✅ Full report saved to: improved_model_evaluation_report.txt")


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 IMPROVED MODEL EVALUATION")
    print("=" * 70)
    print("Features:")
    print("✓ Comprehensive accuracy metrics")
    print("✓ Detailed class-wise performance")
    print("✓ Confusion matrix analysis")
    print("✓ Confidence score evaluation")
    print("✓ Performance report generation")
    print("=" * 70)
    
    try:
        # Load model
        model, model_path = load_improved_model()
        
        # Create test generator
        test_generator, class_names = create_test_data_generator(BASE_DIR)
        
        # Evaluate model
        predictions, true_labels, predicted_labels, accuracy, precision, recall, f1, support = comprehensive_model_evaluation(
            model, test_generator, class_names
        )
        
        # Detailed analysis
        class_performance = detailed_class_analysis(
            true_labels, predicted_labels, class_names, precision, recall, f1, support
        )
        
        # Confusion matrix
        cm, cm_normalized, short_labels = create_confusion_matrix_analysis(true_labels, predicted_labels, class_names)
        
        # Confidence analysis
        confidence_analysis(predictions, true_labels, predicted_labels)
        
        # Generate report
        generate_performance_report(model_path, accuracy, class_performance)
        
        print("\n" + "="*70)
        print("🎉 EVALUATION COMPLETED!")
        print("="*70)
        print(f"📊 Final Accuracy: {accuracy*100:.2f}%")
        print("📁 Generated files:")
        print("  - confusion_matrix_improved.png")
        print("  - confidence_analysis_improved.png") 
        print("  - improved_model_evaluation_report.txt")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print("\n💡 Solutions:")
        print("  1. Make sure you've run train_improved.py first")
        print("  2. Check that the model file exists in the models/ folder")
        print("  3. Verify that the dataset is in MLDemoProj/IMG_CLASSES/")