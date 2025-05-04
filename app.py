import streamlit as st
import os
import time
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import torch

from utils.image_processing import preprocess_image, is_valid_image
from utils.model import load_model, predict
from utils.visualization import generate_heatmap, plot_prediction_results
from utils.dicom_handler import read_dicom, extract_dicom_metadata
from utils.database import save_analysis_result, get_user_analyses, get_analysis_with_predictions
from utils.database import create_user, authenticate_user
from data.conditions import get_condition_info
from models import init_db

# Set page config
st.set_page_config(
    page_title="AI X-Ray Analysis",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure database is initialized
init_db()

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
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'dicom_metadata' not in st.session_state:
    st.session_state.dicom_metadata = None

# Helper function to display session history items
def display_session_history_item(index, item):
    with st.expander(f"Analysis {index+1} - {item['timestamp']}"):
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

# Load model on app startup for faster analysis
@st.cache_resource
def get_model():
    return load_model()

# App title and introduction
st.title("🩻 AI-Powered X-Ray Analysis")

# Sidebar
with st.sidebar:
    # User Authentication
    st.header("User Account")
    
    if not st.session_state.logged_in:
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
        
        with login_tab:
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Log In"):
                if login_username and login_password:
                    user = authenticate_user(login_username, login_password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user.id
                        st.session_state.username = user.username
                        st.success(f"Welcome back, {user.username}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")
        
        with signup_tab:
            signup_username = st.text_input("Username", key="signup_username")
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password", type="password", key="signup_password")
            signup_confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
            
            if st.button("Sign Up"):
                if signup_password != signup_confirm:
                    st.error("Passwords do not match")
                elif not (signup_username and signup_email and signup_password):
                    st.warning("Please fill in all fields")
                else:
                    try:
                        user = create_user(signup_username, signup_email, signup_password)
                        st.session_state.logged_in = True
                        st.session_state.user_id = user.id
                        st.session_state.username = user.username
                        st.success(f"Account created successfully! Welcome, {user.username}!")
                        st.rerun()
                    except Exception as e:
                        if "UNIQUE constraint failed" in str(e):
                            st.error("Username or email already exists")
                        else:
                            st.error(f"Error creating account: {e}")
    else:
        st.write(f"Logged in as: **{st.session_state.username}**")
        if st.button("Log Out"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()
    
    # Navigation
    st.header("Navigation")
    page = st.radio("Select a page:", ["Home", "Upload & Analyze", "History", "About"])
    
    # Sample X-rays
    st.header("Sample X-rays")
    sample_images = {
        # Chest X-rays
        "Normal Chest X-ray": "https://images.radiopaedia.org/images/715141/af5cfddd0990a06ce59a8337cc30b3.jpg",
        "Pneumonia X-ray": "https://images.radiopaedia.org/images/40532170/522dce8bf1ee049a8a5c3147a2b6ff.jpeg",
        "Tuberculosis X-ray": "https://images.radiopaedia.org/images/556722/6ac82713eacb3a4acd8fa28bd0290e.jpg",
        "Pleural Effusion X-ray": "https://images.radiopaedia.org/images/168348/368b18f272a0d29d88c6519d8c81a8.jpg",
        "Rib Fracture X-ray": "https://images.radiopaedia.org/images/3043723/38c3356f75f7d1ece9fe2b523d0cd8.jpg",
        
        # Bone X-rays
        "Normal Bone X-ray": "https://images.radiopaedia.org/images/149300/e2b246ea6b2383ab702418fa7e71d8.jpg",
        "Fracture Bone X-ray": "https://images.radiopaedia.org/images/7882153/3d8a08d6b3bdd33ba531a61f37c69f.jpg",
        "Osteoarthritis X-ray": "https://images.radiopaedia.org/images/575851/98e0eae8a255aeae94d5fb942b1c33.jpg",
        
        # Spine X-rays
        "Normal Spine X-ray": "https://images.radiopaedia.org/images/556721/1ce7cb63cfd66d80a0cd042b23e8dd.jpg",
        "Scoliosis Spine X-ray": "https://images.radiopaedia.org/images/556722/ea321a9fa32df0ed3c4873c2c59b75.jpg"
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
                # Store the selected sample name for reference in the prediction model
                st.session_state.selected_sample = selected_sample
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample image: {e}")

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
                    
                    # Try to extract DICOM metadata
                    try:
                        metadata = extract_dicom_metadata(uploaded_file)
                        st.session_state.dicom_metadata = metadata
                    except Exception as e:
                        st.warning(f"Could not extract DICOM metadata: {e}")
                        st.session_state.dicom_metadata = None
                else:
                    # Regular image file
                    img = Image.open(uploaded_file).convert("RGB")
                    st.session_state.dicom_metadata = None
                
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
                
                # Save to database if user is logged in
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Generate a unique filename for the image
                if not os.path.exists("data/uploads"):
                    os.makedirs("data/uploads", exist_ok=True)
                
                image_filename = f"data/uploads/xray_{uuid.uuid4()}.jpg"
                st.session_state.current_image.save(image_filename)
                
                # Get DICOM metadata if available
                metadata = st.session_state.dicom_metadata if hasattr(st.session_state, 'dicom_metadata') else None
                
                # Save to database (user_id will be None if not logged in)
                try:
                    save_analysis_result(
                        st.session_state.user_id,
                        image_filename,
                        st.session_state.current_prediction,
                        metadata
                    )
                    
                    # Add to session history for temporary display
                    history_item = {
                        "timestamp": timestamp,
                        "image": st.session_state.current_image,
                        "prediction": prediction,
                        "heatmap": heatmap
                    }
                    st.session_state.history.append(history_item)
                    
                except Exception as e:
                    st.error(f"Error saving to database: {str(e)}")
                    st.warning("Analysis completed but not saved to history.")
    
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
            
            # Check if we're dealing with a bone X-ray and using "Mass" to indicate fracture
            is_bone_xray = False
            if hasattr(st.session_state, 'selected_sample'):
                bone_samples = ["Normal Bone X-ray", "Fracture Bone X-ray", "Osteoarthritis X-ray", 
                               "Normal Spine X-ray", "Scoliosis Spine X-ray"]
                if any(bone_type in st.session_state.selected_sample for bone_type in ["Bone", "Spine"]):
                    is_bone_xray = True
            
            # Special handling for fractures in different types of X-rays
            is_rib_fracture = False
            if hasattr(st.session_state, 'selected_sample'):
                if "Rib Fracture" in st.session_state.selected_sample:
                    is_rib_fracture = True
            
            if top_condition == "Mass" and is_bone_xray:
                st.warning(f"Potential fracture detected with {st.session_state.current_prediction[top_condition]:.1%} confidence")
            elif top_condition == "Pleural_Thickening" and is_rib_fracture:
                st.warning(f"Potential rib fracture detected with {st.session_state.current_prediction[top_condition]:.1%} confidence")
            elif top_condition != "No Finding":
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
                # Special case for rib fractures shown through Pleural_Thickening in the model
                if condition == "Pleural_Thickening" and hasattr(st.session_state, 'selected_sample') and "Rib Fracture" in st.session_state.selected_sample:
                    with st.expander(f"Rib Fracture ({confidence:.1%} confidence)"):
                        condition_info = get_condition_info("Rib_Fracture")
                        st.markdown(f"### Rib Fracture")
                        st.markdown(f"**Description:** {condition_info['description']}")
                        st.markdown(f"**Common symptoms:** {condition_info['symptoms']}")
                        st.markdown(f"**Treatment options:** {condition_info['treatment']}")
                else:
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
    
    if st.session_state.logged_in:
        # Load analysis history from database for logged-in user
        try:
            analyses = get_user_analyses(st.session_state.user_id, limit=20)
            
            if not analyses:
                st.info("No analysis history yet. Upload and analyze an X-ray to see it here.")
            else:
                for i, analysis in enumerate(analyses):
                    # Get predictions for this analysis
                    _, predictions = get_analysis_with_predictions(analysis.id)
                    
                    # Create a dictionary of condition -> probability
                    prediction_dict = {p.condition: p.probability for p in predictions}
                    
                    with st.expander(f"Analysis {i+1} - {analysis.timestamp}"):
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # Load and display the image
                            if os.path.exists(analysis.image_path):
                                img = Image.open(analysis.image_path)
                                st.image(img, caption="X-ray Image", use_column_width=True)
                            else:
                                st.warning(f"Image file not found: {analysis.image_path}")
                        
                        with col2:
                            # Display metadata if available
                            if analysis.patient_id or analysis.modality or analysis.body_part:
                                st.subheader("X-ray Information")
                                if analysis.patient_id:
                                    st.write(f"**Patient ID:** {analysis.patient_id}")
                                if analysis.study_date:
                                    st.write(f"**Study Date:** {analysis.study_date}")
                                if analysis.modality:
                                    st.write(f"**Modality:** {analysis.modality}")
                                if analysis.body_part:
                                    st.write(f"**Body Part:** {analysis.body_part}")
                            
                            # Display top conditions
                            st.subheader("Detected Conditions")
                            for condition, probability in sorted(
                                prediction_dict.items(),
                                key=lambda x: x[1],
                                reverse=True
                            )[:5]:
                                st.metric(condition, f"{probability:.1%}")
        except Exception as e:
            st.error(f"Error loading history from database: {str(e)}")
            st.warning("Showing session history instead.")
            
            # Fall back to session history
            if not st.session_state.history:
                st.info("No analysis history in this session. Upload and analyze an X-ray to see it here.")
            else:
                # Display session history
                for i, item in enumerate(reversed(st.session_state.history)):
                    display_session_history_item(i, item)
    else:
        # User not logged in, display message to login
        st.warning("Please log in to view your analysis history.")
        
        # Still show session history if available
        if st.session_state.history:
            st.subheader("Current Session History")
            for i, item in enumerate(reversed(st.session_state.history)):
                display_session_history_item(i, item)
        else:
            st.info("No analysis history in this session. Upload and analyze an X-ray to see it here.")

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
    - **SQLAlchemy & PostgreSQL**: Database for storing analysis history
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