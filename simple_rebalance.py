"""
Simple Dataset Rebalancing Script
Optimized for stronger real-world accuracy:
- removes weak classes automatically
- balances only train split
- keeps val/test clean (no augmentation)
"""

import cv2
import numpy as np
import shutil
from pathlib import Path
import random


class SimpleDatasetRebalancer:
    def __init__(
        self,
        source_dataset="D:\Code\Skin-disease-prediction\dataset",
        target_dataset="balanced_dataset",
        target_train_samples_per_class=2500,
        min_total_samples=1000,
        min_train_samples=600,
        split_ratios=(0.7, 0.15, 0.15),
        random_seed=42,
    ):
        self.source_path = Path(source_dataset)
        self.target_path = Path(target_dataset)
        self.target_train_samples = target_train_samples_per_class
        self.min_total_samples = min_total_samples
        self.min_train_samples = min_train_samples
        self.train_ratio, self.val_ratio, self.test_ratio = split_ratios
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
    def analyze_dataset(self):
        """Analyze current dataset distribution"""
        print("Analyzing current dataset...")
        print("=" * 60)
        
        class_analysis = {}
        
        for split in ['train', 'val', 'test']:
            split_path = self.source_path / split
            if not split_path.exists():
                continue
                
            print(f"\n{split.upper()} SET:")
            
            for class_dir in sorted(split_path.iterdir()):
                if not class_dir.is_dir():
                    continue
                    
                class_name = class_dir.name
                # Count image files
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(list(class_dir.glob(ext)))
                count = len(image_files)
                
                if class_name not in class_analysis:
                    class_analysis[class_name] = {'train': 0, 'val': 0, 'test': 0, 'total': 0}
                
                class_analysis[class_name][split] = count
                
                # Status indicators
                if count == 0:
                    status = "EMPTY"
                elif count < self.min_train_samples:
                    status = "LOW"
                elif count > self.target_train_samples:
                    status = "HIGH"
                else:
                    status = "OK"
                
                print(f"   Class {class_name}: {count:4d} images ({status})")
        
        # Calculate totals
        for class_name in class_analysis:
            class_analysis[class_name]['total'] = sum([
                class_analysis[class_name]['train'],
                class_analysis[class_name]['val'], 
                class_analysis[class_name]['test']
            ])
        
        return class_analysis

    def create_rebalancing_plan(self, class_analysis):
        """Create a plan for class removal and balancing"""
        print(f"\nREBALANCING PLAN:")
        print("=" * 60)
        
        plan = {
            'remove_classes': [],
            'reduce_classes': [],
            'augment_classes': [],
            'keep_classes': []
        }
        
        for class_name, counts in class_analysis.items():
            total = counts['total']
            train_count = counts['train']

            if total < self.min_total_samples or train_count < self.min_train_samples:
                plan['remove_classes'].append((class_name, total))
                print(
                    f"REMOVE Class {class_name}: total={total}, train={train_count}"
                    f" (min_total={self.min_total_samples}, min_train={self.min_train_samples})"
                )

            elif train_count > self.target_train_samples:
                plan['reduce_classes'].append((class_name, train_count, self.target_train_samples))
                print(f"REDUCE Class {class_name}: train {train_count} -> {self.target_train_samples}")

            elif train_count < self.target_train_samples:
                needed = self.target_train_samples - train_count
                plan['augment_classes'].append((class_name, train_count, needed))
                print(f"AUGMENT Class {class_name}: train {train_count} -> {self.target_train_samples} (+{needed})")
                
            else:
                plan['keep_classes'].append((class_name, train_count))
                print(f"KEEP Class {class_name}: {train_count} (good size)")
        
        return plan

    def advanced_augment_image(self, image):
        """Apply moderate augmentation that preserves lesion realism"""
        h, w = image.shape[:2]
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random rotation (moderate)
        if random.random() > 0.3:
            angle = random.randint(-20, 20)
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random cropping and resize
        if random.random() > 0.4:
            crop_factor = random.uniform(0.85, 0.97)
            crop_h, crop_w = int(h * crop_factor), int(w * crop_factor)
            start_h = random.randint(0, h - crop_h)
            start_w = random.randint(0, w - crop_w)
            cropped = image[start_h:start_h+crop_h, start_w:start_w+crop_w]
            image = cv2.resize(cropped, (w, h))
        
        # Random brightness and contrast (mild)
        if random.random() > 0.3:
            brightness = random.randint(-20, 20)
            contrast = random.uniform(0.85, 1.15)
            image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        
        # Random blur (light only)
        if random.random() > 0.6:
            blur_type = random.choice(['gaussian', 'median'])
            if blur_type == 'gaussian':
                kernel_size = random.choice([3, 5])
                image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            else:  # median
                kernel_size = random.choice([3, 5])
                image = cv2.medianBlur(image, kernel_size)
        
        # Random noise
        if random.random() > 0.3:
            noise = np.random.randint(-8, 8, image.shape, dtype=np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Random zoom
        if random.random() > 0.4:
            zoom_factor = random.uniform(0.9, 1.1)
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, 0, zoom_factor)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random shear (very mild)
        if random.random() > 0.7:
            shear_factor = random.uniform(-0.08, 0.08)
            M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random HSV adjustments
        if random.random() > 0.4:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv = hsv.astype(np.float32)
            
            # Hue shift
            hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-10, 10)) % 180
            
            # Saturation adjust
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
            
            # Value adjust  
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.8, 1.2), 0, 255)
            
            hsv = hsv.astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return image

    def _gather_class_images(self, class_name):
        """Gather all image paths for a class across train/val/test."""
        all_images = []
        for split in ['train', 'val', 'test']:
            source_dir = self.source_path / split / class_name
            if not source_dir.exists():
                continue
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                all_images.extend(list(source_dir.glob(ext)))
        random.shuffle(all_images)
        return all_images

    def _split_images(self, image_files):
        """Split class images into train/val/test using configured ratios."""
        total = len(image_files)
        train_n = int(total * self.train_ratio)
        val_n = int(total * self.val_ratio)
        test_n = total - train_n - val_n

        train_files = image_files[:train_n]
        val_files = image_files[train_n:train_n + val_n]
        test_files = image_files[train_n + val_n:train_n + val_n + test_n]
        return train_files, val_files, test_files

    def _copy_files(self, image_files, target_dir, prefix):
        """Copy files while avoiding filename collisions."""
        target_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for idx, img_file in enumerate(image_files):
            try:
                new_name = f"{prefix}_{idx:06d}_{img_file.name}"
                shutil.copy2(img_file, target_dir / new_name)
                copied += 1
            except Exception as e:
                print(f"   Error copying {img_file}: {e}")
        return copied

    def copy_and_reduce_class(self, class_name, split, max_count):
        """Copy original images with reduction if needed"""
        source_dir = self.source_path / split / class_name
        target_dir = self.target_path / split / class_name
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(source_dir.glob(ext)))
        
        # If reducing, randomly sample
        if max_count and len(image_files) > max_count:
            image_files = random.sample(image_files, max_count)
        
        copied = 0
        for img_file in image_files:
            try:
                shutil.copy2(img_file, target_dir / img_file.name)
                copied += 1
            except Exception as e:
                print(f"   Error copying {img_file}: {e}")
        
        return copied

    def augment_class(self, class_name, source_images, num_needed):
        """Generate augmented images for TRAIN split only."""
        target_dir = self.target_path / 'train' / class_name
        image_files = list(source_images)
        
        if not image_files:
            print(f"   No images found for class {class_name}")
            return 0
        
        print(f"   Generating {num_needed} train augmentations for class {class_name}...")
        print(f"   Source images available: {len(image_files)}")
        
        generated = 0
        failed_attempts = 0
        max_failed_attempts = 100  # Prevent infinite loops
        
        for i in range(num_needed):
            if failed_attempts > max_failed_attempts:
                print(f"   ⚠️ Stopping after {max_failed_attempts} failed attempts")
                break
                
            try:
                # Select random source image
                source_img_path = random.choice(image_files)
                
                # Load image
                image = cv2.imread(str(source_img_path))
                if image is None:
                    failed_attempts += 1
                    continue
                
                # Apply one or two rounds only (less artifact risk)
                for _ in range(random.randint(1, 2)):
                    image = self.advanced_augment_image(image)
                
                # Ensure image is still valid
                if image.shape[0] < 32 or image.shape[1] < 32:
                    failed_attempts += 1
                    continue
                
                # Save augmented image with detailed naming
                output_filename = f"aug_{generated:06d}_{source_img_path.stem}.jpg"
                output_path = target_dir / output_filename
                
                # Save with high quality
                success = cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                
                if success:
                    generated += 1
                    failed_attempts = 0  # Reset on success
                    
                    # Progress indicator for large augmentation
                    if generated % 250 == 0:
                        progress = (generated / num_needed) * 100
                        print(f"      Progress: {generated}/{num_needed} ({progress:.1f}%)")
                else:
                    failed_attempts += 1
                    
            except Exception as e:
                print(f"   ⚠️ Error augmenting image: {e}")
                failed_attempts += 1
                continue
        
        print(f"   ✅ Successfully generated {generated} augmented images")
        if generated < num_needed:
            print(f"   ⚠️ Could only generate {generated}/{num_needed} due to technical issues")
            
        return generated

    def rebalance_dataset(self):
        """Main rebalancing function"""
        print("STARTING DATASET REBALANCING")
        print("=" * 60)
        
        # Clear target directory
        if self.target_path.exists():
            shutil.rmtree(self.target_path)
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (self.target_path / split).mkdir(parents=True, exist_ok=True)
        
        # Analyze current dataset
        class_analysis = self.analyze_dataset()
        
        # Create rebalancing plan
        plan = self.create_rebalancing_plan(class_analysis)
        
        print(f"\nEXECUTING REBALANCING...")
        print("=" * 60)
        
        final_stats = {'train': {}, 'val': {}, 'test': {}}
        
        # Process all classes except those to be removed
        all_classes = set(class_analysis.keys())
        classes_to_remove = {cls[0] for cls in plan['remove_classes']}
        classes_to_keep = all_classes - classes_to_remove
        
        print(f"Classes to remove: {sorted(classes_to_remove)}")
        print(f"Classes to keep: {sorted(classes_to_keep)}")
        
        for class_name in sorted(classes_to_keep):
            print(f"\nProcessing Class {class_name}...")
            all_images = self._gather_class_images(class_name)
            if len(all_images) < self.min_total_samples:
                print(f"   SKIP: only {len(all_images)} images after scan")
                final_stats['train'][class_name] = 0
                final_stats['val'][class_name] = 0
                final_stats['test'][class_name] = 0
                continue

            # Re-split from full class pool for consistent distribution
            train_files, val_files, test_files = self._split_images(all_images)

            # Keep val/test fully real (no augmentation)
            copied_val = self._copy_files(
                val_files,
                self.target_path / 'val' / class_name,
                prefix='real_val'
            )
            copied_test = self._copy_files(
                test_files,
                self.target_path / 'test' / class_name,
                prefix='real_test'
            )
            final_stats['val'][class_name] = copied_val
            final_stats['test'][class_name] = copied_test

            # For train: reduce or augment to exact target
            max_real_train = min(len(train_files), self.target_train_samples)
            selected_train_files = train_files[:max_real_train]
            copied_train = self._copy_files(
                selected_train_files,
                self.target_path / 'train' / class_name,
                prefix='real_train'
            )

            needed = self.target_train_samples - copied_train
            augmented = 0
            if needed > 0 and copied_train > 0:
                augmented = self.augment_class(class_name, selected_train_files, needed)

            final_stats['train'][class_name] = copied_train + augmented

            print(
                f"   TRAIN: {copied_train} real + {augmented} aug = {final_stats['train'][class_name]}"
            )
            print(f"   VAL:   {copied_val} real (no augmentation)")
            print(f"   TEST:  {copied_test} real (no augmentation)")

        return final_stats

    def print_final_report(self, final_stats):
        """Print final rebalancing report"""
        print(f"\nFINAL DATASET STATISTICS")
        print("=" * 70)
        
        for split in ['train', 'val', 'test']:
            print(f"\n{split.upper()} SET:")
            
            if not final_stats[split]:
                print("   No data")
                continue
            
            total = 0
            for class_name in sorted(final_stats[split].keys()):
                count = final_stats[split][class_name]
                total += count
                print(f"   Class {class_name}: {count:4d} images")
            
            print(f"   TOTAL: {total:,} images")
        
        # Balance analysis for training set
        train_classes = len(final_stats['train'])
        if train_classes > 0:
            train_samples = [count for count in final_stats['train'].values() if count > 0]
            if train_samples:
                min_samples = min(train_samples)
                max_samples = max(train_samples)
                avg_samples = sum(train_samples) // len(train_samples)
                balance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
                
                print(f"\nTRAINING SET BALANCE ANALYSIS:")
                print(f"   Classes: {train_classes}")
                print(f"   Min samples: {min_samples}")
                print(f"   Max samples: {max_samples}")
                print(f"   Average: {avg_samples}")
                print(f"   Balance ratio: {balance_ratio:.1f}:1")
                
                if balance_ratio < 2:
                    print("   EXCELLENT BALANCE!")
                elif balance_ratio < 3:
                    print("   GOOD BALANCE")
                else:
                    print("   MODERATE BALANCE")

if __name__ == "__main__":
    print("DATASET REBALANCER - ACCURACY-FOCUSED VERSION")
    print("Creates balanced train data while keeping val/test fully real")
    print("=" * 70)

    # Resolve paths relative to the script's own location so the script
    # works regardless of the current working directory.
    SCRIPT_DIR = Path(__file__).resolve().parent
    # dataset/ is now inside the project folder
    SOURCE_DATASET = SCRIPT_DIR / "dataset"
    TARGET_DATASET = SCRIPT_DIR / "balanced_dataset"

    if not SOURCE_DATASET.exists():
        print(f"Error: dataset folder not found at {SOURCE_DATASET}")
        print("Run download_dataset.py first to download the dataset.")
        exit(1)

    # Configuration - Better generalization and stable validation metrics
    rebalancer = SimpleDatasetRebalancer(
        source_dataset=str(SOURCE_DATASET),
        target_dataset=str(TARGET_DATASET),
        target_train_samples_per_class=2500,
        min_total_samples=1000,
        min_train_samples=600,
        split_ratios=(0.7, 0.15, 0.15),
        random_seed=42,
    )
    
    print(f"\nSettings (Accuracy-Focused):")
    print(f"   Source: {rebalancer.source_path}")
    print(f"   Target: {rebalancer.target_path}")
    print(f"   Target train samples/class: {rebalancer.target_train_samples}")
    print(f"   Minimum total samples/class: {rebalancer.min_total_samples}")
    print(f"   Minimum train samples/class: {rebalancer.min_train_samples}")
    print(f"   Split ratios (train/val/test): {rebalancer.train_ratio}/{rebalancer.val_ratio}/{rebalancer.test_ratio}")
    print(f"   Classes below thresholds will be REMOVED")
    
    print(f"\nPlanned behavior:")
    print(f"   - REMOVE low-sample classes")
    print(f"   - TRAIN balanced to fixed target with moderate augmentation")
    print(f"   - VAL/TEST use only real images (no augmentation)")
    
    print(f"\n🎯 Goal:")
    print(f"   - Higher real validation accuracy with cleaner evaluation")
    print(f"   - Better generalization by avoiding augmented val/test")
    print(f"   - More stable training through class pruning + balanced train")
    
    response = input(f"\nProceed with rebalancing? (y/N): ").strip().lower()
    
    if response == 'y':
        try:
            final_stats = rebalancer.rebalance_dataset()
            rebalancer.print_final_report(final_stats)
            
            print(f"\n🎉 BALANCED DATASET CREATED!")
            print(f"📁 Balanced dataset in: {rebalancer.target_path}")
            print(f"\n🚀 NEXT STEPS:")
            print(f"   1. Train on new balanced data: python train.py")
            print(f"   2. Evaluate model: python test.py")
            print(f"\n💡 Improvements in this version:")
            print(f"   - Auto-removes low-sample classes")
            print(f"   - Balances only train split")
            print(f"   - Keeps val/test unaugmented for honest metrics")
            print(f"   - Uses moderate augmentation to reduce artifacts")
            
        except Exception as e:
            print(f"Error during rebalancing: {str(e)}")
            import traceback
            traceback.print_exc()
            
    else:
        print("Rebalancing cancelled")