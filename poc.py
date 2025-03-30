import streamlit as st
import requests
import json
from PIL import Image
import io
import base64
from urllib.parse import quote

# Set page configuration
st.set_page_config(
    page_title="ShrushrutAI - Skin Disease Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #0f4c81, #1e88e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 10px;
    }
    .sub-header {
        font-size: 1.7rem;
        color: #0f4c81;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #0f4c81;
        padding-bottom: 5px;
    }
    .result-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 24px;
        border-left: 5px solid #0f4c81;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 4px solid #0f4c81;
    }
    .diagnosis-positive {
        color: #d32f2f;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .diagnosis-neutral {
        color: #ed6c02;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .diagnosis-negative {
        color: #2e7d32;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .confidence {
        font-size: 1rem;
        color: #555;
        margin-top: 5px;
    }
    .report-container {
        max-height: 450px;
        overflow-y: auto;
        padding: 20px;
        background-color: #000;
        color: #fff;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .report-container h1, .report-container h2, .report-container h3 {
        color: #fff;
    }
    .report-container a {
        color: #4fc3f7;
    }
    .assistant-container {
        background: linear-gradient(145deg, #1a2035, #192841);
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        color: #fff;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .assistant-header {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
        background: rgba(255, 255, 255, 0.1);
        padding: 10px 15px;
        border-radius: 8px;
    }
    .assistant-icon {
        font-size: 2rem;
        margin-right: 15px;
    }
    .assistant-title {
        font-size: 1.3rem;
        font-weight: 600;
    }
    .stButton > button {
        background-color: #0f4c81;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 25px;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1976d2;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #bdbdbd;
        padding: 12px 15px;
    }
    .stCheckbox > div {
        padding: 10px;
    }
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 15px;
        margin-top: 5px;
        margin-right: 10px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .status-healthy {
        background-color: rgba(46, 125, 50, 0.15);
        color: #2e7d32;
        border: 1px solid rgba(46, 125, 50, 0.3);
    }
    .status-unhealthy {
        background-color: rgba(211, 47, 47, 0.15);
        color: #d32f2f;
        border: 1px solid rgba(211, 47, 47, 0.3);
    }
    .result-title {
        font-size: 1.3rem;
        color: #0f4c81;
        margin-bottom: 15px;
        font-weight: 600;
    }
    .divider {
        height: 1px;
        background-color: #e0e0e0;
        margin: 15px 0;
    }
    .skin-info {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 10px;
    }
    .skin-info-item {
        background-color: #f1f8e9;
        padding: 8px 12px;
        border-radius: 8px;
        color: #33691e;
    }
</style>
""", unsafe_allow_html=True)

# API URL
API_BASE_URL = "http://127.0.0.1:6780"

def get_prediction(image_url):
    response = requests.get(f"{API_BASE_URL}/predict", params={"image_url": image_url, "ObjId": "default"})
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

def get_answer(question, deep_search=False):
    response = requests.post(f"{API_BASE_URL}/ans", params={"question": question, "deep_search": deep_search})
    if response.status_code == 200:
        return response.json()["response"]
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return "Sorry, I couldn't process your question at this time."

def parse_verification(verification_text):
    parts = verification_text.split(',', 3)
    if len(parts) >= 3:
        status = parts[0].strip()
        confidence = parts[1].strip()
        skin_type = parts[2].strip()
        remarks = parts[3].strip() if len(parts) > 3 else ""
        return status, confidence, skin_type, remarks
    return "Unknown", "Unknown", "Unknown", "Unknown"

def parse_prediction(prediction_text):
    parts = prediction_text.split(',', 2)
    if len(parts) >= 2:
        disease = parts[0].strip()
        confidence = parts[1].strip()
        remarks = parts[2].strip() if len(parts) > 2 else ""
        return disease, confidence, remarks
    return "Unknown", "Unknown", "Unknown"

# App header with logo
st.markdown("<h1 class='main-header'>ShrushrutAI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.3rem; margin-top: -10px; margin-bottom: 20px;'>Advanced Skin Disease Diagnosis Platform</p>", unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
    <h4 style='margin-top: 0;'>How It Works</h4>
    <p>Upload a skin image URL below to get a comprehensive AI-powered analysis, including disease detection, 
    confidence scores, and physician guidance. Our platform combines deep learning with dermatological expertise.</p>
</div>
""", unsafe_allow_html=True)

# Main layout with two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h2 class='sub-header'>Image Analysis</h2>", unsafe_allow_html=True)
    
    # Image URL input with better styling
    image_url = st.text_input(
        "Enter Image URL", 
        "https://res.cloudinary.com/dkuitm79x/image/upload/v1743246584/ISIC_0027184_tgjtwl.jpg",
        placeholder="Paste an image URL here..."
    )
    
    # Analyze button with better styling
    analyze_button = st.button("Analyze Image", use_container_width=True)
    
    if image_url:
        try:
            st.image(image_url, caption="Skin Image for Analysis", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")

if analyze_button and image_url:
    with st.spinner("Analyzing image... This may take a moment."):
        prediction_result = get_prediction(image_url)
        
        if prediction_result:
            verify_status, verify_confidence, skin_type, verify_remarks = parse_verification(prediction_result["verify"])
            
            disease, confidence, remarks = parse_prediction(prediction_result["prediction"])
            
            report = prediction_result["report"]
            
            jarvis_instructions = prediction_result["jarvis"]
            
            st.session_state.prediction_result = prediction_result
            
            with col2:
                st.markdown("<h2 class='sub-header'>Diagnosis Results</h2>", unsafe_allow_html=True)
                
                # Initial verification
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='result-title'>Initial Assessment</h3>", unsafe_allow_html=True)
                
                status_badge_class = "status-healthy" if verify_status.lower() == "healthy" else "status-unhealthy"
                st.markdown(f"<span class='status-badge {status_badge_class}'>{verify_status}</span>", unsafe_allow_html=True)
                st.markdown(f"<p class='confidence'>Confidence: {verify_confidence}</p>", unsafe_allow_html=True)
                
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown("<div class='skin-info'>", unsafe_allow_html=True)
                st.markdown(f"<div class='skin-info-item'>Skin Type: {skin_type}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown(f"<p style='margin-top: 15px;'><strong>Remarks:</strong> {verify_remarks}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Disease prediction
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='result-title'>Detailed Diagnosis</h3>", unsafe_allow_html=True)
                
                disease_color = "diagnosis-negative" if disease.lower() == "healthy" else "diagnosis-positive"
                st.markdown(f"<p>Condition: <span class='{disease_color}'>{disease}</span></p>", unsafe_allow_html=True)
                st.markdown(f"<p class='confidence'>Confidence: {confidence}</p>", unsafe_allow_html=True)
                
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Clinical Notes:</strong> {remarks}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Physician instructions
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='result-title'>Physician Instructions</h3>", unsafe_allow_html=True)
                st.markdown(jarvis_instructions, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Full report section (below the two columns)
            st.markdown("<h2 class='sub-header'>Detailed Medical Report</h2>", unsafe_allow_html=True)
            st.markdown("<div class='report-container'>", unsafe_allow_html=True)
            st.markdown(report, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # AI Assistant section
            st.markdown("<h2 class='sub-header'>AI Dermatology Assistant</h2>", unsafe_allow_html=True)
            st.markdown("""
            <div class='info-box'>
                Ask questions about the diagnosis, treatment options, or general skin health information. 
                For research-based answers, enable 'Deep Research' mode.
            </div>
            """, unsafe_allow_html=True)
            
            col_q1, col_q2 = st.columns([3, 1])
            
            with col_q1:
                user_question = st.text_input(
                    "Ask a question about the diagnosis or skin health:",
                    placeholder="E.g., What are the treatment options for this condition?"
                )
            
            with col_q2:
                deep_search = st.checkbox("Deep Research", help="Uses research papers for in-depth answers")
            
            if st.button("Get Answer", use_container_width=True):
                if user_question:
                    with st.spinner("Searching for information..."):
                        answer = get_answer(user_question, deep_search)
                        
                        st.markdown("<div class='assistant-container'>", unsafe_allow_html=True)
                        st.markdown("""
                        <div class='assistant-header'>
                            <span class='assistant-icon'>🩺</span>
                            <span class='assistant-title'>AI Dermatology Assistant</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(answer, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("Please enter a question.")

# Sidebar with additional information
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/caduceus.png", width=80)
    st.title("About ShrushrutAI")
    st.markdown("""
    ShrushrutAI combines cutting-edge deep learning with medical expertise to provide accurate skin disease diagnosis.
    
    ### Key Features
    - **AI-Powered Analysis**: Utilizes advanced machine learning to identify skin conditions
    - **Comprehensive Reports**: Detailed medical reports with evidence-based insights
    - **Clinical Guidance**: Physician instructions for treatment planning
    - **Research Integration**: Access to the latest dermatological research
    
    ### How To Use
    1. Enter the URL of a skin image
    2. Click "Analyze Image"
    3. Review the detailed diagnosis
    4. Ask questions to the AI assistant
    
    ### Disclaimer
    This tool is designed to assist medical professionals and should not replace proper medical diagnosis. Always consult with a qualified dermatologist for definitive medical advice.
    """)
    
    st.markdown("---")
    st.markdown("© 2025 ShrushrutAI - Advanced Dermatological Diagnostics")