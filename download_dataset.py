"""
Enhanced script to download and prepare the Hugging Face skin lesions dataset locally
Optimized for limited resources and crash prevention
"""
import os
import shutil
import gc
import json
import time
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def setup_robust_session():
    """Setup a robust requests session with retries"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def safe_image_save(image, image_path, max_retries=3):
    """Safely save an image with error handling"""
    for attempt in range(max_retries):
        try:
            # Validate image
            if image is None:
                return False
                
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Create directory if needed
            image_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image with optimization
            image.save(image_path, 'JPEG', quality=85, optimize=True)
            
            # Verify the saved file
            if image_path.exists() and image_path.stat().st_size > 0:
                return True
            else:
                if image_path.exists():
                    image_path.unlink()  # Remove corrupted file
                return False
                
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed to save {image_path}: {e}")
            if image_path.exists():
                try:
                    image_path.unlink()  # Remove corrupted file
                except:
                    pass
            
            if attempt < max_retries - 1:
                time.sleep(0.5)  # Brief pause before retry
            
    return False

def process_split_efficiently(dataset_split, split_name, class_names, output_dirs, batch_size=50):
    """Process a dataset split efficiently with batching"""
    print(f"\n🔄 Processing {split_name} split...")
    
    # Get split info
    total_samples = len(dataset_split)
    print(f"📊 Found {total_samples} samples in {split_name}")
    
    if total_samples == 0:
        return {}
    
    # Initialize counters
    from collections import defaultdict
    class_counts = defaultdict(int)
    processed_counts = defaultdict(int)
    failed_count = 0
    
    # Process in smaller batches to manage memory
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        
        print(f"🔄 Processing batch {batch_start//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size} ({batch_start+1}-{batch_end}/{total_samples})")
        
        # Process samples one by one to avoid batching issues
        for i in range(batch_start, batch_end):
            try:
                # Access individual sample directly from dataset
                sample = dataset_split[i]
                
                # Debug: Check sample structure
                if i == batch_start and batch_start == 0:
                    print(f"🔍 Sample structure debug: {type(sample)}, keys: {sample.keys() if hasattr(sample, 'keys') else 'No keys'}")
                
                # Extract data with error handling
                if isinstance(sample, dict):
                    image = sample.get('image')
                    label = sample.get('label')
                else:
                    print(f"⚠️ Unexpected sample type: {type(sample)}")
                    failed_count += 1
                    continue
                
                if image is None or label is None:
                    print(f"⚠️ Missing data - image: {image is not None}, label: {label is not None}")
                    failed_count += 1
                    continue
                
                # Validate label
                if not isinstance(label, (int, float)) or label >= len(class_names) or label < 0:
                    print(f"⚠️ Skipping invalid label {label} (type: {type(label)})")
                    failed_count += 1
                    continue
                
                label = int(label)  # Ensure it's an integer
                
                # Count for distribution
                class_counts[label] += 1
                current_idx = class_counts[label] - 1
                
                # Simplified split distribution (avoid counting all samples)
                # Use modulo for balanced distribution
                if current_idx % 10 < 7:  # 70% train
                    target_dir = output_dirs['train']
                    target_split = "train"
                elif current_idx % 10 < 8:  # 10% val (70-80%)
                    target_dir = output_dirs['val']
                    target_split = "val"
                else:  # 20% test (80-100%)
                    target_dir = output_dirs['test']
                    target_split = "test"
                
                # Create image path
                image_path = target_dir / str(label) / f"{target_split}_{label}_{current_idx:06d}.jpg"
                
                # Save image safely
                if safe_image_save(image, image_path):
                    processed_counts[label] += 1
                else:
                    failed_count += 1
                    print(f"⚠️ Failed to save image for class {label}")
                
                # Memory cleanup and progress update
                if (i + 1) % 10 == 0:
                    gc.collect()
                    
                # Show progress every 100 images
                if (i + 1) % 100 == 0:
                    print(f"✅ Processed {i + 1}/{total_samples} samples...")
                    
            except Exception as e:
                failed_count += 1
                print(f"⚠️ Error processing sample {i}: {e}")
                continue
        
        # Cleanup after batch
        gc.collect()
        time.sleep(0.1)  # Brief pause for system recovery
    
    if failed_count > 0:
        print(f"⚠️ Failed to process {failed_count} images in {split_name}")
    
    return dict(class_counts)

def download_dataset():
    """Download and organize the skin lesions dataset with crash prevention"""
    print("🔄 Downloading Skin Lesions Dataset (14 classes)...")
    print("This will be saved locally in the 'dataset' folder")
    print("💡 Optimized for limited resources...")
    
    try:
        # Setup robust session
        session = setup_robust_session()
        
        # Create dataset directory
        dataset_path = Path("dataset")
        if dataset_path.exists():
            print("⚠️ Dataset directory exists. Removing old data...")
            shutil.rmtree(dataset_path)
        
        # Create directory structure
        train_dir = dataset_path / "train"
        val_dir = dataset_path / "val"
        test_dir = dataset_path / "test"
        
        output_dirs = {
            'train': train_dir,
            'val': val_dir,
            'test': test_dir
        }
        
        for dir_path in output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Class mapping
        class_names = [
            "Actinic_keratoses",
            "Basal_cell_carcinoma", 
            "Benign_keratosis",
            "Chickenpox",
            "Cowpox",
            "Dermatofibroma",
            "Healthy",
            "HFMD",
            "Measles",
            "Melanocytic_nevi",
            "Melanoma",
            "Monkeypox",
            "Squamous_cell_carcinoma",
            "Vascular_lesions"
        ]
        
        # save name list so training/testing scripts can label outputs
        try:
            (dataset_path / "class_names.json").write_text(json.dumps(class_names, indent=2))
            print("✅ Saved class_names.json in dataset directory")
        except Exception as e:
            print(f"⚠️ Could not write class_names.json: {e}")
        
        print(f"✅ Dataset has {len(class_names)} classes:")
        for i, class_name in enumerate(class_names):
            print(f"  {i}. {class_name}")
        
        # Create class directories
        for i in range(len(class_names)):
            for dir_path in output_dirs.values():
                (dir_path / str(i)).mkdir(exist_ok=True)
        
        # Download dataset with timeout and retries
        print("📥 Loading dataset from Hugging Face (with retries)...")
        max_retries = 3
        dataset = None
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Download attempt {attempt + 1}/{max_retries}")
                # Remove trust_remote_code as it's deprecated
                dataset = load_dataset("ahmed-ai/skin-lesions-classification-dataset")
                print("✅ Dataset loaded successfully!")
                break
            except Exception as e:
                print(f"❌ Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print("⏳ Waiting before retry...")
                    time.sleep(5)
                    gc.collect()
                else:
                    raise e
        
        if dataset is None:
            raise Exception("Failed to download dataset after all retries")
        
        # Process each split separately to manage memory
        all_class_counts = {}
        available_splits = list(dataset.keys())
        print(f"📂 Available splits: {available_splits}")
        
        for split_name in available_splits:
            print(f"\n{'='*50}")
            print(f"Processing {split_name} split")
            print(f"{'='*50}")
            
            try:
                split_data = dataset[split_name]
                split_counts = process_split_efficiently(
                    split_data, split_name, class_names, output_dirs, batch_size=25
                )
                
                # Merge counts
                for label, count in split_counts.items():
                    all_class_counts[label] = all_class_counts.get(label, 0) + count
                
                # Cleanup
                del split_data
                gc.collect()
                time.sleep(1)  # Brief system recovery
                
            except Exception as e:
                print(f"❌ Error processing {split_name}: {e}")
                continue
        
        # Cleanup dataset from memory
        del dataset
        gc.collect()
        
        print("\n✅ Dataset downloaded and organized successfully!")
        print(f"📁 Dataset saved to: {dataset_path.absolute()}")
        
        # Print final statistics
        print("\n📊 Final Dataset Statistics:")
        total_train = total_val = total_test = 0
        
        for split_name, split_dir in [("Train", train_dir), ("Validation", val_dir), ("Test", test_dir)]:
            split_total = 0
            for i in range(len(class_names)):
                class_dir = split_dir / str(i)
                if class_dir.exists():
                    class_images = len(list(class_dir.glob("*.jpg")))
                    split_total += class_images
            
            print(f"  {split_name}: {split_total} images")
            if split_name == "Train":
                total_train = split_total
            elif split_name == "Validation":
                total_val = split_total
            else:
                total_test = split_total
        
        total_images = total_train + total_val + total_test
        if total_images > 0:
            print(f"\n🎯 Total: {total_images} images")
            print(f"   Train: {total_train} ({100*total_train/total_images:.1f}%)")
            print(f"   Val:   {total_val} ({100*total_val/total_images:.1f}%)")
            print(f"   Test:  {total_test} ({100*total_test/total_images:.1f}%)")
        else:
            print("⚠️ No images were successfully processed!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on failure
        if 'dataset_path' in locals() and dataset_path.exists():
            try:
                shutil.rmtree(dataset_path)
                print("🧹 Cleaned up partial download")
            except:
                pass
        
        return False

if __name__ == "__main__":
    print("🚀 Skin Lesions Dataset Downloader (Optimized)")
    print("-" * 50)
    print("💡 System Requirements: 8GB+ RAM, Stable Internet")
    print("⚡ Optimized for crash prevention and memory efficiency")
    print("-" * 50)
    
    try:
        start_time = time.time()
        success = download_dataset()
        end_time = time.time()
        duration = end_time - start_time
        
        if success:
            print(f"\n🎉 Dataset preparation completed successfully!")
            print(f"⏱️ Total time: {duration:.1f} seconds")
            print("✅ Ready to start training!")
            print("Run: python train_improved.py")
        else:
            print(f"\n❌ Dataset download failed after {duration:.1f} seconds")
            print("💡 Tips for troubleshooting:")
            print("  - Check internet connection")
            print("  - Ensure sufficient disk space (>5GB)")
            print("  - Close other memory-intensive applications")
            print("  - Try running again (temporary network issues)")
            
    except KeyboardInterrupt:
        print("\n⛔ Download interrupted by user")
        # Cleanup
        dataset_path = Path("dataset")
        if dataset_path.exists():
            try:
                shutil.rmtree(dataset_path)
                print("🧹 Cleaned up partial download")
            except:
                pass
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()