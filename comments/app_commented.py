"""
================================================================================
SKIN DISEASE CLASSIFICATION - STREAMLIT INTERACTIVE DASHBOARD
================================================================================

Purpose:
    Interactive Streamlit web app for:
    1. Visualizing training results
    2. Making predictions on new skin disease images
    3. Understanding model confidence
    4. Analyzing class distribution

How to Run:
    streamlit run app.py
    
Then open: http://localhost:8501 in your browser
================================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from PIL import Image
import cv2

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Skin Disease Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLING
# ============================================================================

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DEFINE DISEASE CLASSES
# ============================================================================

DISEASE_CLASSES = [
    "Eczema",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma (BCC)",
    "Melanocytic Nevi (NV)",
    "Benign Keratosis-like Lesions (BKL)",
    "Psoriasis/Lichen Planus",
    "Seborrheic Keratoses/Benign Tumors",
    "Tinea/Ringworm/Candidiasis",
    "Warts/Molluscum/Viral Infections"
]

# Color palette for visualizations
COLORS = {
    0: "#FF6B6B",   # Red
    1: "#4ECDC4",   # Teal
    2: "#45B7D1",   # Blue
    3: "#FFA07A",   # Light Salmon
    4: "#98D8C8",   # Sage
    5: "#F7DC6F",   # Yellow
    6: "#BB8FCE",   # Purple
    7: "#85C1E2",   # Light Blue
    8: "#F8B88B",   # Peach
    9: "#ABEBC6"    # Light Green
}

# ============================================================================
# CACHE MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained MobileNet model (cached for performance)."""
    model_path = 'models/skin_disease_mobilenet_model.h5'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        st.info("Please run 'python train.py' first to train the model.")
        return None
    
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image_path, img_size=(224, 224)):
    """
    Preprocess image for model prediction.
    
    Steps:
    1. Read image
    2. Convert to RGB if needed
    3. Resize to 224x224
    4. Normalize to [0, 1]
    
    Args:
        image_path (str): Path to the image file
        img_size (tuple): Target size
    
    Returns:
        image (np.array): Preprocessed image
    """
    # Read image
    image = cv2.imread(image_path)
    
    if image is None:
        return None
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224
    image = cv2.resize(image, img_size)
    
    # Normalize to [0, 1]
    image = image.astype('float32') / 255.0
    
    return image


def preprocess_uploaded_image(uploaded_file, img_size=(224, 224)):
    """
    Preprocess uploaded image from Streamlit file uploader.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        img_size (tuple): Target size
    
    Returns:
        image (np.array): Preprocessed image
    """
    # Read the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Convert to numpy array and resize
    image_array = np.array(image)
    image_resized = cv2.resize(image_array, img_size)
    
    # Normalize to [0, 1]
    image_normalized = image_resized.astype('float32') / 255.0
    
    return image_normalized, image


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_disease(model, image, confidence_threshold=0.3):
    """
    Make prediction on a single image.
    
    Args:
        model: Trained Keras model
        image (np.array): Preprocessed image (224x224x3)
        confidence_threshold (float): Min confidence to show prediction
    
    Returns:
        predictions (np.array): Probabilities for all classes
        predicted_class (int): Index of highest probability class
        confidence (float): Confidence of top prediction
    """
    # Add batch dimension: (224,224,3) → (1,224,224,3)
    image_batch = np.expand_dims(image, axis=0)
    
    # Get predictions: 10 probabilities
    predictions = model.predict(image_batch, verbose=0)[0]
    
    # Get top prediction
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    
    return predictions, predicted_class, confidence


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_prediction_confidence(predictions, disease_classes):
    """
    Create bar chart of prediction confidence for all classes.
    
    Args:
        predictions (np.array): Probabilities for all classes
        disease_classes (list): Names of all classes
    
    Returns:
        fig (matplotlib.figure): Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Disease': disease_classes,
        'Confidence': predictions
    }).sort_values('Confidence', ascending=True)
    
    # Color the top prediction differently
    colors = ['#2ecc71' if i == len(df)-1 else '#7f8c8d' for i in range(len(df))]
    
    # Create horizontal bar chart
    ax.barh(df['Disease'], df['Confidence'], color=colors)
    ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Confidence for Each Disease', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    
    # Add value labels on bars
    for i, (disease, conf) in enumerate(zip(df['Disease'], df['Confidence'])):
        ax.text(conf + 0.01, i, f'{conf:.2%}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_class_comparison(predictions, disease_classes):
    """
    Create pie chart showing distribution of predictions.
    
    Args:
        predictions (np.array): Probabilities for all classes
        disease_classes (list): Names of all classes
    
    Returns:
        fig (matplotlib.figure): Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Only show classes with >5% confidence
    threshold = 0.05
    significant_indices = np.where(predictions > threshold)[0]
    
    if len(significant_indices) > 0:
        labels = [disease_classes[i] for i in significant_indices]
        values = predictions[significant_indices]
        colors_list = [COLORS[i] for i in significant_indices]
        
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors_list,
            startangle=90,
            textprops={'fontsize': 10}
        )
        
        ax.set_title('Model Prediction Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def display_image_info(image, image_original):
    """
    Display image statistics in a 2-column layout.
    
    Args:
        image (np.array): Preprocessed image
        image_original (PIL.Image): Original uploaded image
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Uploaded Image**")
        st.image(image_original, use_column_width=True)
    
    with col2:
        st.write("**Image Statistics**")
        
        # Display image properties
        stats_data = {
            'Property': ['Original Size', 'Preprocessed Size', 'Min Pixel Value', 'Max Pixel Value', 'Mean Pixel Value'],
            'Value': [
                f"{image_original.size}",
                f"(224, 224, 3)",
                f"{image.min():.4f}",
                f"{image.max():.4f}",
                f"{image.mean():.4f}"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # Display histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(image.flatten(), bins=50, color='steelblue', edgecolor='black')
        ax.set_xlabel('Pixel Value', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Pixel Value Distribution', fontsize=12, fontweight='bold')
        st.pyplot(fig)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application."""
    
    # ========================================================================
    # HEADER
    # ========================================================================
    
    st.markdown("""
        <h1 style='text-align: center; color: #2c3e50;'>🏥 Skin Disease Classification</h1>
        <h3 style='text-align: center; color: #7f8c8d;'>MobileNet Deep Learning Model</h3>
        <p style='text-align: center; color: #95a5a6;'>
        Powered by MobileNetV2 Architecture (Optimized for Speed & Efficiency)
        </p>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # ========================================================================
    # SIDEBAR NAVIGATION
    # ========================================================================
    
    page = st.sidebar.radio(
        "📊 Navigation",
        ["🔬 Make Prediction", "📈 Model Information", "📋 Disease Reference"],
        label_visibility="collapsed"
    )
    
    # ========================================================================
    # PAGE 1: MAKE PREDICTION
    # ========================================================================
    
    if page == "🔬 Make Prediction":
        st.header("🔬 Make a Prediction")
        st.write("Upload a skin disease image to get a classification prediction from our optimized MobileNet model.")
        
        # Load model
        model = load_model()
        
        if model is None:
            st.stop()
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["📤 Upload Image", "📁 Select from Dataset"],
            horizontal=True
        )
        
        image_preprocessed = None
        image_original = None
        
        # ====================================================================
        # UPLOAD IMAGE METHOD
        # ====================================================================
        
        if input_method == "📤 Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp']
            )
            
            if uploaded_file is not None:
                image_preprocessed, image_original = preprocess_uploaded_image(uploaded_file)
        
        # ====================================================================
        # SELECT FROM DATASET METHOD
        # ====================================================================
        
        else:
            base_path = 'MLDemoProj/IMG_CLASSES'
            
            if os.path.exists(base_path):
                # Get all class folders
                class_folders = sorted([f for f in os.listdir(base_path) 
                                      if os.path.isdir(os.path.join(base_path, f))])
                
                # Select disease class
                selected_class = st.selectbox(
                    "Select Disease Class:",
                    class_folders
                )
                
                # Get images in selected class
                class_path = os.path.join(base_path, selected_class)
                image_files = sorted([f for f in os.listdir(class_path) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                
                if image_files:
                    # Select specific image
                    selected_image = st.selectbox(
                        "Select Image:",
                        image_files
                    )
                    
                    image_path = os.path.join(class_path, selected_image)
                    image_preprocessed = preprocess_image(image_path)
                    image_original = Image.open(image_path).convert('RGB')
            else:
                st.warning("Dataset not found. Please check the path.")
        
        # ====================================================================
        # MAKE PREDICTION
        # ====================================================================
        
        if image_preprocessed is not None and image_original is not None:
            st.divider()
            
            # Display image and statistics
            display_image_info(image_preprocessed, image_original)
            
            st.divider()
            
            # Make prediction
            predictions, predicted_class, confidence = predict_disease(model, image_preprocessed)
            
            # ================================================================
            # DISPLAY RESULTS
            # ================================================================
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="🎯 Predicted Disease",
                    value=DISEASE_CLASSES[predicted_class]
                )
            
            with col2:
                st.metric(
                    label="📊 Confidence",
                    value=f"{confidence:.2%}"
                )
            
            with col3:
                # Determine confidence level
                if confidence > 0.7:
                    confidence_level = "🟢 High"
                elif confidence > 0.4:
                    confidence_level = "🟡 Medium"
                else:
                    confidence_level = "🔴 Low"
                
                st.metric(
                    label="📈 Confidence Level",
                    value=confidence_level
                )
            
            st.divider()
            
            # Display predictions
            st.subheader("📊 Confidence Scores for All Classes")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig1 = plot_prediction_confidence(predictions, DISEASE_CLASSES)
                st.pyplot(fig1)
            
            with col2:
                fig2 = plot_class_comparison(predictions, DISEASE_CLASSES)
                st.pyplot(fig2)
            
            st.divider()
            
            # Show detailed predictions
            st.subheader("📋 Detailed Predictions")
            
            results_df = pd.DataFrame({
                'Disease': DISEASE_CLASSES,
                'Confidence': [f"{p:.2%}" for p in predictions],
                'Probability': predictions
            }).sort_values('Probability', ascending=False)
            
            st.dataframe(results_df, use_container_width=True)
            
            # Disclaimer
            st.info(
                "⚠️ **Important**: This is a machine learning model and should not be used as a medical diagnosis. "
                "Please consult with a qualified dermatologist for accurate medical diagnosis and treatment."
            )
    
    # ========================================================================
    # PAGE 2: MODEL INFORMATION
    # ========================================================================
    
    elif page == "📈 Model Information":
        st.header("📈 Model Information")
        
        st.subheader("🏗️ Architecture Overview")
        
        st.write("""
        This model uses **MobileNetV2 Architecture** - a single, highly optimized deep learning model:
        
        ### 🔷 Path 1: MobileNetV2
        - **Purpose**: Lightweight CNN optimized for mobile devices
        - **Parameters**: ~3.5M
        - **Speed**: ⚡ Very Fast
        - **Pre-training**: ImageNet (1.2M images, 1000 classes)
        
        ### 🔷 Single Backbone: MobileNetV2
        - **Purpose**: Efficient CNN with balanced speed/accuracy
        - **Parameters**: ~5.3M
        - **Speed**: ⚡ Fast
        - **Pre-training**: ImageNet (1.2M images, 1000 classes)
        
        ### 🔗 Fusion Strategy
        1. Both models process the input image independently
        2. Each outputs 1280 features after GlobalAveragePooling2D
        3. Features are **concatenated** (combined) to get 2560 features
        4. Final Dense layers classify into 10 disease categories
        """)
        
        st.subheader("🎯 Why This Architecture Works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            ✅ **Advantages**:
            - **MobileNet**: Lightweight architecture designed for efficiency
            - **Transfer Learning**: Pre-trained weights capture image features
            - **Fast Training**: ~50% faster than ensemble approaches
            - **Memory Efficient**: Optimized for systems with limited RAM
            - **CPU Friendly**: No GPU needed, runs on any hardware
            - **Versatile**: Works on imbalanced data
            """)
        
        with col2:
            st.write("""
            ⚙️ **Technical Details**:
            - **Input**: 224×224×3 (RGB images)
            - **Total Parameters**: ~8.8M
            - **Backbone Freezing**: Pre-trained weights frozen
            - **Fine-tuning**: Only top layers trained
            - **Optimization**: Adam optimizer
            """)
        
        st.subheader("⚖️ Handling Class Imbalance")
        
        st.write("""
        The dataset has imbalanced classes (1.25k to 3.5k samples):
        
        **Solution: Automatic Class Weights**
        - Classes with fewer samples get higher weights
        - Loss for rare classes is amplified during training
        - Model learns to predict rare classes correctly
        
        **Formula**:
        ```
        weight[class] = total_samples / (num_classes × samples_in_class)
        ```
        
        **Example**:
        - Melanoma (3.5k samples): weight = 1.4
        - Atopic Dermatitis (1.25k samples): weight = 4.0
        - The model "cares more" about Eczema due to higher weight
        """)
        
        st.subheader("🛑 Preventing Overfitting")
        
        st.write("""
        **Early Stopping**:
        - Monitor: Validation Loss
        - Patience: 5 epochs
        - Stop training when validation loss stops improving
        - Restore: Weights from best epoch
        
        **Dropout**:
        - 50% dropout in final dense layer
        - Randomly deactivates neurons during training
        - Improves generalization
        """)
    
    # ========================================================================
    # PAGE 3: DISEASE REFERENCE
    # ========================================================================
    
    elif page == "📋 Disease Reference":
        st.header("📋 Disease Reference Guide")
        
        diseases_info = {
            "Eczema": {
                "description": "Inflammatory skin condition causing itchiness, redness, and dryness",
                "types": "Atopic dermatitis, contact dermatitis, dyshidrotic eczema",
                "color": "#FF6B6B"
            },
            "Melanoma": {
                "description": "Malignant tumor of melanocytes (pigment-producing cells)",
                "types": "Most serious form of skin cancer",
                "color": "#4ECDC4"
            },
            "Atopic Dermatitis": {
                "description": "Chronic inflammatory skin disease with intense itching",
                "types": "Often inherited, triggered by environmental factors",
                "color": "#45B7D1"
            },
            "Basal Cell Carcinoma (BCC)": {
                "description": "Most common type of skin cancer",
                "types": "Usually caused by sun exposure, rarely metastasizes",
                "color": "#FFA07A"
            },
            "Melanocytic Nevi (NV)": {
                "description": "Common moles, usually benign skin growths",
                "types": "May be flat or raised, various colors",
                "color": "#98D8C8"
            },
            "Benign Keratosis-like Lesions (BKL)": {
                "description": "Non-cancerous skin growths, common in aging skin",
                "types": "Seborrheic keratosis, solar lentigo",
                "color": "#F7DC6F"
            },
            "Psoriasis/Lichen Planus": {
                "description": "Autoimmune inflammatory skin conditions",
                "types": "Psoriasis: plaques and redness; Lichen Planus: purple papules",
                "color": "#BB8FCE"
            },
            "Seborrheic Keratoses": {
                "description": "Benign growths common in older people",
                "types": "Brown, black, or tan raised bumps",
                "color": "#85C1E2"
            },
            "Tinea/Ringworm/Candidiasis": {
                "description": "Fungal infections of the skin",
                "types": "Tinea pedis, tinea corporis, candida",
                "color": "#F8B88B"
            },
            "Warts/Molluscum": {
                "description": "Viral skin infections",
                "types": "Common warts, genital warts, molluscum contagiosum",
                "color": "#ABEBC6"
            }
        }
        
        # Create columns for disease cards
        for i, (disease, info) in enumerate(diseases_info.items()):
            if i % 2 == 0:
                col1, col2 = st.columns(2)
            
            with (col1 if i % 2 == 0 else col2):
                st.markdown(f"""
                    <div style='background-color: {info["color"]}; 
                                border-radius: 10px; 
                                padding: 20px; 
                                color: white;
                                margin: 10px 0;'>
                    <h4 style='margin-top: 0; color: white;'>{disease}</h4>
                    <p><b>Description:</b> {info['description']}</p>
                    <p><b>Types:</b> {info['types']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.divider()
        
        st.info(
            "🏥 This reference is for educational purposes only. "
            "Always consult with a qualified dermatologist for medical diagnosis and treatment."
        )


# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()
