import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io

# Page configuration
st.set_page_config(
    page_title="Advanced Skin Lesion Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495E;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .confidence-high {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    }
    .confidence-medium {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
    }
    .confidence-low {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .disease-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .improvement-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Disease information for 14 skin lesion classes
DISEASE_INFO = {
    "Actinic_keratoses": {
        "description": "Rough, scaly patches on sun-exposed skin that may become cancerous.",
        "symptoms": "Rough, scaly patches, reddish, pink, or brown spots",
        "urgency": "MEDIUM - Should be monitored and treated by dermatologist",
        "color": "#F39C12"
    },
    "Basal_cell_carcinoma": {
        "description": "The most common type of skin cancer, typically appears as a small, shiny bump.",
        "symptoms": "Pearl-like bumps, open sores, red patches, pink growths",
        "urgency": "MEDIUM - Should be examined by a dermatologist",
        "color": "#E74C3C"
    },
    "Benign_keratosis": {
        "description": "Non-cancerous skin growths that appear as scaly, rough patches.",
        "symptoms": "Rough, scaly patches, varying colors",
        "urgency": "LOW - Generally harmless",
        "color": "#16A085"
    },
    "Chickenpox": {
        "description": "Viral infection causing itchy, blister-like rash all over the body.",
        "symptoms": "Red, itchy spots that develop into fluid-filled blisters",
        "urgency": "MEDIUM - Contagious, requires medical care",
        "color": "#E67E22"
    },
    "Cowpox": {
        "description": "Viral infection typically contracted from cattle, causes localized lesions.",
        "symptoms": "Red, inflamed lesions that may ulcerate",
        "urgency": "MEDIUM - Requires medical attention",
        "color": "#8E44AD"
    },
    "Dermatofibroma": {
        "description": "Common benign skin tumor that feels like a hard bump.",
        "symptoms": "Firm, raised nodules, brown or reddish color",
        "urgency": "LOW - Benign, cosmetic concern only",
        "color": "#27AE60"
    },
    "Healthy": {
        "description": "Normal, healthy skin with no signs of disease or infection.",
        "symptoms": "Clear, normal-appearing skin",
        "urgency": "NONE - No treatment needed",
        "color": "#27AE60"
    },
    "HFMD": {
        "description": "Hand, foot, and mouth disease - viral infection causing rash and fever.",
        "symptoms": "Fever, sore throat, rash on hands, feet, and mouth",
        "urgency": "MEDIUM - Contagious, requires medical care",
        "color": "#3498DB"
    },
    "Measles": {
        "description": "Highly contagious viral infection causing fever and characteristic rash.",
        "symptoms": "High fever, cough, runny nose, red, blotchy rash",
        "urgency": "HIGH - Highly contagious, requires immediate medical attention",
        "color": "#E74C3C"
    },
    "Melanocytic_nevi": {
        "description": "Common benign moles composed of melanocytes (pigment cells).",
        "symptoms": "Brown or black spots, uniform color and shape",
        "urgency": "LOW - Monitor for changes",
        "color": "#9B59B6"
    },
    "Melanoma": {
        "description": "A serious form of skin cancer that develops in melanocytes (pigment-producing cells).",
        "symptoms": "Asymmetric moles, irregular borders, color changes, diameter >6mm",
        "urgency": "HIGH - Requires immediate medical attention",
        "color": "#C0392B"
    },
    "Monkeypox": {
        "description": "Viral infection causing fever and distinctive pustular rash.",
        "symptoms": "Fever, headache, pustular lesions, lymph node swelling",
        "urgency": "HIGH - Contagious, requires immediate medical attention",
        "color": "#8E44AD"
    },
    "Squamous_cell_carcinoma": {
        "description": "Second most common type of skin cancer, appears as scaly red patches.",
        "symptoms": "Scaly red patches, open sores, elevated growths with central depression",
        "urgency": "HIGH - Requires immediate medical attention",
        "color": "#E74C3C"
    },
    "Vascular_lesions": {
        "description": "Abnormalities in blood vessels causing red or purple skin marks.",
        "symptoms": "Red or purple marks, may be raised or flat",
        "urgency": "LOW - Usually benign, consult dermatologist if changing",
        "color": "#2980B9"
    }
}

DISEASE_CLASSES = [
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

@st.cache_resource
def load_model():
    """Load the improved trained model with caching."""
    model_paths = [
        'best_mobilenet_skin_classifier_finetuned.h5',
        'mobilenet_skin_lesion_classifier_final.h5',
        'best_mobilenet_skin_classifier.h5',
        'efficientnet_skin_lesion_classifier.h5',
        'mobilenet_skin_lesion_classifier.h5',
        'best_efficientnet_model.h5',
        'best_mobilenet_model.h5',
        'models/efficientnet_finetuned_best.h5',
        'models/mobilenet_finetuned_best.h5',
        'models/efficientnet_best.h5',
        'models/mobilenet_best.h5',
        'models/best_model.h5',
        'models/skin_disease_improved_model.h5',
        'models/skin_disease_mobilenet_model.h5'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = keras.models.load_model(model_path, compile=False)
                # Recompile for consistency
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                return model, model_path
            except Exception as e:
                st.error(f"Error loading {model_path}: {e}")
                continue
    
    st.error("No trained model found! Please run 'python train_improved.py' first.")
    return None, None

def preprocess_image(image):
    """Preprocess uploaded image for prediction."""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Ensure RGB format
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = img_array[:,:,:3]  # Remove alpha channel
        
        # Resize to model input size (384x384 for optimized MobileNet)
        img_resized = cv2.resize(img_array, (384, 384))
        
        # Normalize pixel values
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def create_prediction_visualization(predictions, class_names):
    """Create interactive visualization of predictions."""
    
    # Sort predictions for better visualization
    pred_data = list(zip(class_names, predictions[0]))
    pred_data.sort(key=lambda x: x[1], reverse=True)
    
    classes_sorted, probs_sorted = zip(*pred_data)
    
    # Create color mapping based on confidence
    colors = []
    for prob in probs_sorted:
        if prob > 0.7:
            colors.append('#2ECC71')  # High confidence - Green
        elif prob > 0.3:
            colors.append('#F39C12')  # Medium confidence - Orange
        else:
            colors.append('#E74C3C')  # Low confidence - Red
    
    # Create interactive bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(probs_sorted),
            y=list(classes_sorted),
            orientation='h',
            marker=dict(color=colors),
            text=[f'{p:.1%}' for p in probs_sorted],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence Scores",
        xaxis_title="Confidence Score",
        yaxis_title="Disease Class",
        height=500,
        showlegend=False,
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 1], tickformat='.0%')
    )
    
    return fig

def create_confidence_gauge(confidence):
    """Create a gauge chart for prediction confidence."""
    
    # Determine color based on confidence level
    if confidence > 0.8:
        color = "#2ECC71"
        interpretation = "Very High"
    elif confidence > 0.6:
        color = "#F39C12"
        interpretation = "High"
    elif confidence > 0.4:
        color = "#FF7043"
        interpretation = "Medium"
    else:
        color = "#E74C3C"
        interpretation = "Low"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence Level: {interpretation}"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 60], 'color': "gray"},
                {'range': [60, 80], 'color': "lightgreen"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig

def display_model_info(model_path):
    """Display information about the loaded model."""
    
    st.markdown("### 🤖 Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🏗️ Architecture</h4>
            <p>Improved MobileNetV2</p>
            <p>Progressive Training</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path) / (1024*1024)
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Model Stats</h4>
                <p>Size: {model_size:.2f} MB</p>
                <p>Classes: {len(DISEASE_CLASSES)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>✨ Improvements</h4>
            <p>Swish Activation</p>
            <p>Enhanced Augmentation</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display improvement badges
    st.markdown("""
    <div style="text-align: center; margin: 1rem 0;">
        <span class="improvement-badge">🚀 Progressive Training</span>
        <span class="improvement-badge">🧠 Swish Activations</span>
        <span class="improvement-badge">💪 Enhanced Augmentation</span>
        <span class="improvement-badge">⚡ Better Early Stopping</span>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🩺 Advanced Skin Lesion Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced AI-Powered Medical Screening Tool</div>', unsafe_allow_html=True)
    
    # Load model
    model, model_path = load_model()
    
    if model is None:
        st.error("❌ Model could not be loaded. Please check the model files.")
        return
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown("## 🔧 Controls")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Skin Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of the skin condition"
        )
        
        # Analysis options
        st.markdown("### 📋 Analysis Options")
        show_detailed_analysis = st.checkbox("Show Detailed Analysis", True)
        show_confidence_gauge = st.checkbox("Show Confidence Gauge", True)
        show_disease_info = st.checkbox("Show Disease Information", True)
        
        # Model information
        display_model_info(model_path)
    
    # Main content area
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Image properties
            st.markdown(f"""
            <div class="disease-info">
                <strong>Image Properties:</strong><br>
                Size: {image.size[0]} × {image.size[1]} pixels<br>
                Format: {image.format}<br>
                Mode: {image.mode}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🔮 Prediction Results")
            
            # Preprocess and predict
            processed_image = preprocess_image(image)
            
            if processed_image is not None:
                with st.spinner("🤔 Analyzing skin condition..."):
                    predictions = model.predict(processed_image, verbose=0)
                    predicted_class_index = np.argmax(predictions[0])
                    predicted_class = DISEASE_CLASSES[predicted_class_index]
                    confidence = predictions[0][predicted_class_index]
                    
                    # Main prediction card
                    if confidence > 0.6:
                        card_class = "confidence-high"
                    elif confidence > 0.4:
                        card_class = "confidence-medium"
                    else:
                        card_class = "confidence-low"
                    
                    st.markdown(f"""
                    <div class="prediction-card {card_class}">
                        <h3>🎯 Primary Diagnosis</h3>
                        <h2>{predicted_class}</h2>
                        <h4>Confidence: {confidence:.1%}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence gauge
                    if show_confidence_gauge:
                        st.plotly_chart(
                            create_confidence_gauge(confidence),
                            use_container_width=True
                        )
        
        # Detailed analysis section
        if show_detailed_analysis:
            st.markdown("### 📊 Detailed Analysis")
            
            # Prediction visualization
            fig = create_prediction_visualization(predictions, DISEASE_CLASSES)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            
            st.markdown("#### 🏆 Top 3 Predictions")
            cols = st.columns(3)
            
            for i, idx in enumerate(top_3_indices):
                with cols[i]:
                    class_name = DISEASE_CLASSES[idx]
                    prob = predictions[0][idx]
                    
                    st.markdown(f"""
                    <div class="disease-info">
                        <strong>#{i+1}: {class_name}</strong><br>
                        Confidence: {prob:.1%}<br>
                        {'🎯 Primary' if i == 0 else '🔍 Alternative'}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Disease information
        if show_disease_info:
            st.markdown("### 🩺 Medical Information")
            
            predicted_disease_key = None
            for key in DISEASE_INFO.keys():
                if key.lower() in predicted_class.lower() or predicted_class.lower() in key.lower():
                    predicted_disease_key = key
                    break
            
            if predicted_disease_key:
                disease_info = DISEASE_INFO[predicted_disease_key]
                
                st.markdown(f"""
                <div class="disease-info">
                    <h4 style="color: {disease_info['color']};">🏥 {predicted_disease_key}</h4>
                    <p><strong>Description:</strong> {disease_info['description']}</p>
                    <p><strong>Common Symptoms:</strong> {disease_info['symptoms']}</p>
                    <p><strong>Medical Urgency:</strong> <span style="color: {disease_info['color']}; font-weight: bold;">{disease_info['urgency']}</span></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Medical disclaimer
            st.markdown("""
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; padding: 1rem; margin: 1rem 0;">
                <h4 style="color: #856404;">⚠️ Medical Disclaimer</h4>
                <p style="color: #856404; margin-bottom: 0;">This AI tool is for educational purposes only and should not replace professional medical diagnosis. 
                Please consult a qualified dermatologist or healthcare provider for proper medical evaluation and treatment.</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Landing page content
        st.markdown("""
        ### 👋 Welcome to the Advanced Skin Lesion Classifier!
        
        This advanced AI model uses progressive training and state-of-the-art techniques to analyze skin conditions.
        
        #### 🚀 New Features:
        - **Progressive Training**: Better accuracy through two-phase training
        - **Swish Activations**: Modern activation functions for improved performance  
        - **Enhanced Augmentation**: Better generalization with advanced data augmentation
        - **Smarter Early Stopping**: More patient training for better convergence
        - **Confidence Analysis**: Detailed prediction reliability metrics
        
        #### 📋 Instructions:
        1. Upload a clear image of the skin condition using the sidebar
        2. Wait for the AI analysis to complete
        3. Review the prediction results and confidence scores
        4. Read the medical information and recommendations
        
        #### ⚠️ Important Notes:
        - This tool is for educational purposes only
        - Always consult a medical professional for proper diagnosis
        - High-quality, well-lit images provide better results
        """)
        
        # Sample predictions visualization
        st.markdown("### 📊 Sample Analysis")
        
        # Create a sample prediction visualization
        sample_predictions = np.random.dirichlet(np.ones(len(DISEASE_CLASSES)), 1)
        sample_fig = create_prediction_visualization(sample_predictions, DISEASE_CLASSES)
        st.plotly_chart(sample_fig, use_container_width=True)

if __name__ == "__main__":
    main()