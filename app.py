import streamlit as st
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import torch

from utils.image_processing import preprocess_image, is_valid_image
from utils.model import load_model, predict
from utils.visualization import generate_heatmap, plot_prediction_results
from utils.dicom_handler import read_dicom
from data.conditions import get_condition_info

# Set page config
st.set_page_config(
    page_title="AI X-Ray Analysis",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'current_heatmap' not in st.session_state:
    st.session_state.current_heatmap = None
if 'model' not in st.session_state:
    st.session_state.model = None

# App title and introduction
st.title("🩻 AI-Powered X-Ray Analysis")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select a page:", ["Home", "Upload & Analyze", "History", "About"])
    
    st.header("Sample X-rays")
    sample_images = {
        "Chest X-ray 1": "https://images.unsplash.com/photo-1504813184591-01572f98c85f",
        "Chest X-ray 2": "https://images.unsplash.com/photo-1519494080410-f9aa76cb4283",
        "Bone X-ray": "https://images.unsplash.com/photo-1514416432279-50fac261c7dd",
        "Dental X-ray": "https://images.unsplash.com/photo-1460672985063-6764ac8b9c74",
        "Spine X-ray": "https://images.unsplash.com/photo-1516841273335-e39b37888115",
        "Abdominal X-ray": "https://images.unsplash.com/photo-1471864190281-a93a3070b6de"
    }
    
    selected_sample = st.selectbox("Load a sample X-ray:", ["None"] + list(sample_images.keys()))
    
    if selected_sample != "None":
        st.image(sample_images[selected_sample], caption=selected_sample, use_column_width=True)
        if st.button("Use this sample"):
            image_url = sample_images[selected_sample]
            try:
                import requests
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                st.session_state.current_image = img
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample image: {e}")

# Load model on app startup for faster analysis
@st.cache_resource
def get_model():
    return load_model()

# Initialize model
if st.session_state.model is None:
    with st.spinner("Loading AI model..."):
        st.session_state.model = get_model()
        
# Home page
if page == "Home":
    st.header("Welcome to AI-Powered X-Ray Analysis")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        This application uses deep learning to analyze X-ray images and detect abnormalities. 
        Our AI model is trained on thousands of X-ray images and can identify common conditions.
        
        ### Features:
        - Upload and analyze X-ray images in common formats (PNG, JPEG, DICOM)
        - AI-powered detection of abnormalities
        - Visualization of detection areas with heatmaps
        - Educational information about detected conditions
        - History tracking for uploaded images
                
        ### How to Use:
        1. Navigate to the "Upload & Analyze" page
        2. Upload your X-ray image or use one of our samples
        3. View the AI analysis results and heatmap visualization
        4. Learn more about detected conditions
        5. Review your analysis history anytime
        
        *Disclaimer: This tool is for educational purposes and should not replace professional medical advice.*
        """)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1499951360447-b19be8fe80f5", 
                caption="Modern Radiology Workspace", use_column_width=True)
        st.image("https://images.unsplash.com/photo-1496664444929-8c75efb9546f", 
                caption="AI in Medical Imaging", use_column_width=True)

# Upload & Analyze page
elif page == "Upload & Analyze":
    st.header("Upload & Analyze X-Ray Images")
    
    # Image upload section
    upload_col, preview_col = st.columns([1, 1])
    
    with upload_col:
        uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png", "dcm"])
        
        if uploaded_file is not None:
            try:
                # Handle different file types
                file_extension = uploaded_file.name.split(".")[-1].lower()
                
                if file_extension == "dcm":
                    # DICOM file
                    img_array = read_dicom(uploaded_file)
                    img = Image.fromarray(img_array).convert("RGB")
                else:
                    # Regular image file
                    img = Image.open(uploaded_file).convert("RGB")
                
                # Check if image is valid
                if not is_valid_image(img):
                    st.error("The uploaded file doesn't appear to be a valid X-ray image. Please upload a proper X-ray image.")
                else:
                    st.session_state.current_image = img
            except Exception as e:
                st.error(f"Error processing the uploaded file: {e}")
    
    with preview_col:
        if st.session_state.current_image is not None:
            st.image(st.session_state.current_image, caption="Uploaded X-ray", use_column_width=True)
    
    # Analysis section
    if st.session_state.current_image is not None:
        if st.button("Analyze X-ray"):
            with st.spinner("Analyzing image..."):
                # Preprocess the image
                preprocessed_img = preprocess_image(st.session_state.current_image)
                
                # Make prediction
                prediction, features = predict(st.session_state.model, preprocessed_img)
                st.session_state.current_prediction = prediction
                
                # Generate heatmap
                heatmap = generate_heatmap(st.session_state.model, preprocessed_img)
                st.session_state.current_heatmap = heatmap
                
                # Add to history
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                history_item = {
                    "timestamp": timestamp,
                    "image": st.session_state.current_image,
                    "prediction": prediction,
                    "heatmap": heatmap
                }
                st.session_state.history.append(history_item)
    
    # Display results
    if st.session_state.current_prediction is not None:
        st.subheader("Analysis Results")
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            # Plot prediction results
            fig = plot_prediction_results(st.session_state.current_prediction)
            st.pyplot(fig)
            
            # Find top condition
            top_condition = max(st.session_state.current_prediction.items(), key=lambda x: x[1])[0]
            if top_condition != "No Finding":
                st.warning(f"Potential finding: {top_condition} with {st.session_state.current_prediction[top_condition]:.1%} confidence")
            else:
                st.success("No significant abnormalities detected")
        
        with result_col2:
            # Display heatmap
            if st.session_state.current_heatmap is not None:
                plt.figure(figsize=(10, 8))
                plt.imshow(np.array(st.session_state.current_image))
                plt.imshow(st.session_state.current_heatmap, cmap='jet', alpha=0.4)
                plt.axis('off')
                plt.title("Areas of Interest")
                st.pyplot(plt)
                st.caption("Heatmap shows areas the AI focused on for analysis")
        
        # Educational information about conditions
        st.subheader("Educational Information")
        
        # Sort conditions by confidence (descending)
        sorted_conditions = sorted(
            st.session_state.current_prediction.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Display information for top 3 conditions with confidence > 5%
        displayed_count = 0
        for condition, confidence in sorted_conditions:
            if confidence > 0.05 and displayed_count < 3:
                with st.expander(f"{condition} ({confidence:.1%} confidence)"):
                    condition_info = get_condition_info(condition)
                    st.markdown(f"### {condition}")
                    st.markdown(f"**Description:** {condition_info['description']}")
                    st.markdown(f"**Common symptoms:** {condition_info['symptoms']}")
                    st.markdown(f"**Treatment options:** {condition_info['treatment']}")
                displayed_count += 1

# History page
elif page == "History":
    st.header("Analysis History")
    
    if not st.session_state.history:
        st.info("No analysis history yet. Upload and analyze an X-ray to see it here.")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Analysis {i+1} - {item['timestamp']}"):
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    st.image(item['image'], caption="X-ray Image", use_column_width=True)
                
                with col2:
                    # Display top conditions
                    st.subheader("Detected Conditions")
                    for condition, confidence in sorted(
                        item['prediction'].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:3]:
                        st.metric(condition, f"{confidence:.1%}")
                
                with col3:
                    # Display heatmap
                    if item['heatmap'] is not None:
                        plt.figure(figsize=(8, 6))
                        plt.imshow(np.array(item['image']))
                        plt.imshow(item['heatmap'], cmap='jet', alpha=0.4)
                        plt.axis('off')
                        plt.title("Areas of Interest")
                        st.pyplot(plt)

# About page
elif page == "About":
    st.header("About this Application")
    
    st.markdown("""
    ## AI-powered X-Ray Analysis
    
    This application uses deep learning techniques, specifically Convolutional Neural Networks (CNNs), 
    to analyze X-ray images and detect common medical conditions.
    
    ### Technology Stack
    - **Streamlit**: Web interface framework
    - **PyTorch**: Deep learning framework for the AI model
    - **OpenCV & Pillow**: Image processing
    - **Matplotlib**: Visualization
    - **SimpleITK/pydicom**: DICOM file handling
    - **Grad-CAM**: Heatmap generation for AI explainability
    
    ### Model Information
    The core of this application is a deep learning model based on a pre-trained DenseNet121 architecture, 
    fine-tuned on large datasets of X-ray images including:
    - NIH ChestX-ray14 (112,000+ chest X-rays)
    - CheXpert (Stanford; 200,000+ chest X-rays)
    - MIMIC-CXR (Beth Israel Hospital)
    
    ### Limitations
    - This tool is for educational purposes only and should not replace professional medical advice
    - The AI model performs best on chest X-rays and may have lower accuracy on other X-ray types
    - False positives and false negatives can occur
    - Performance varies based on image quality and positioning
    
    ### References
    - [NIH Chest X-ray Dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)
    - [CheXpert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert/)
    - [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/)
    """)
    
    st.image("https://images.unsplash.com/photo-1504813184591-01572f98c85f", 
            caption="AI in Radiology", use_column_width=True)

st.sidebar.info("""
### Disclaimer
This application is for educational purposes only. The AI analysis should not replace professional medical diagnosis.
Always consult with qualified healthcare providers for medical advice.
""")
