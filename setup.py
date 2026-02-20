"""
Setup script for improved skin disease classification
Installs all required packages and prepares the environment
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def install_packages():
    """Install required Python packages"""
    print("📦 Installing required packages...")
    
    packages = [
        "tensorflow==2.13.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "opencv-python>=4.8.0",  
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "streamlit>=1.25.0",
        "datasets>=2.14.0",
        "huggingface_hub>=0.16.0",
        "transformers>=4.33.0",
        "tqdm>=4.65.0"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️ Failed to install {package}, continuing...")
    
    print("✅ Package installation completed")

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = ["models", "dataset", "models/checkpoints"]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def check_gpu():
    """Check GPU availability"""
    print("🖥️ Checking GPU availability...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            
            # Configure GPU memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth configured")
        else:
            print("ℹ️ No GPU found, will use CPU (training will be slower)")
            
    except ImportError:
        print("⚠️ TensorFlow not installed yet")

def setup_huggingface():
    """Setup Hugging Face authentication"""
    print("🤗 Setting up Hugging Face...")
    
    print("To download the facial skin condition dataset, you need to:")
    print("1. Create a Hugging Face account at https://huggingface.co")
    print("2. Run: huggingface-cli login")
    print("3. Or set HF_TOKEN environment variable")
    print("4. This is required for the dataset download")

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED!")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    print("1. Authenticate with Hugging Face:")
    print("   huggingface-cli login")
    print()
    print("2. Download the dataset:")
    print("   python download_dataset.py")
    print()
    print("3. Train the improved models:")
    print("   python train_improved.py")
    print()
    print("4. Test the trained models:")
    print("   python test_improved.py")
    print()
    print("5. Run the Streamlit app:")
    print("   streamlit run app.py")
    
    print("\n💡 TIPS:")
    print("- Training will take 1-3 hours depending on your hardware")
    print("- GPU is recommended for faster training")
    print("- The new models should achieve >60% accuracy (vs your current 6%)")
    print("- Both MobileNet and EfficientNet will be trained for comparison")

def main():
    """Main setup function"""
    print("🚀 IMPROVED SKIN DISEASE CLASSIFIER SETUP")
    print("="*50)
    
    # Install packages
    install_packages()
    
    # Setup directories
    setup_directories()
    
    # Check GPU
    check_gpu()
    
    # Setup Hugging Face
    setup_huggingface()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()