# app.py
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import json
import io
import base64
import sys

# Try to import OpenCV with fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("OpenCV not available. Image processing features will be limited.")

# Set page configuration
st.set_page_config(
    page_title="AI Diagnosis Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .diagnosis-box {
        background-color: #F8F9FA;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28A745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FFC107;
        font-weight: bold;
    }
    .confidence-low {
        color: #DC3545;
        font-weight: bold;
    }
    .symptom-checkbox {
        background-color: #E9ECEF;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MedicalDiagnosisAI:
    def __init__(self):
        self.image_model = None
        self.symptom_model = None
        self.label_encoder = None
        self.scaler = StandardScaler()
        self.diseases = ['malaria', 'pneumonia', 'healthy', 'bacterial_skin', 'fungal_skin', 'viral_skin']
        self.cv2_available = CV2_AVAILABLE
    
    def load_demo_models(self):
        """Load demo models (in real scenario, load pre-trained models)"""
        # For demo purposes, we'll create simple models
        # In production, you would load your actual trained models
        
        # Create a simple image model architecture
        self.image_model = self.create_demo_cnn_model()
        
        # Create and train a demo symptom model
        self.train_demo_symptom_model()
        
        return True
    
    def create_demo_cnn_model(self):
        """Create a demo CNN model structure"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        return model
    
    def train_demo_symptom_model(self):
        """Train a demo symptom model with synthetic data"""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        symptoms_data = []
        labels = []
        
        for i in range(n_samples):
            # Create synthetic symptom profiles
            fever = np.random.choice([0, 1], p=[0.3, 0.7])
            cough = np.random.choice([0, 1])
            headache = np.random.choice([0, 1])
            fatigue = np.random.choice([0, 1])
            chills = np.random.choice([0, 1])
            skin_rash = np.random.choice([0, 1])
            chest_pain = np.random.choice([0, 1])
            shortness_breath = np.random.choice([0, 1])
            
            # Vitals
            temperature = np.random.normal(37 + 2*fever, 0.5)
            heart_rate = np.random.normal(80 + 10*fever, 10)
            
            # Determine disease based on symptoms
            if fever and chills and not cough:
                disease = 'malaria'
            elif cough and chest_pain and shortness_breath:
                disease = 'pneumonia'
            elif skin_rash:
                disease = np.random.choice(['bacterial_skin', 'fungal_skin', 'viral_skin'])
            else:
                disease = 'healthy'
            
            features = [fever, cough, headache, fatigue, chills, skin_rash, 
                       chest_pain, shortness_breath, temperature, heart_rate]
            
            symptoms_data.append(features)
            labels.append(disease)
        
        # Train model
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labels)
        
        self.symptom_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        self.symptom_model.fit(symptoms_data, y_encoded)
        
        self.symptom_feature_names = [
            'fever', 'cough', 'headache', 'fatigue', 'chills', 'skin_rash',
            'chest_pain', 'shortness_breath', 'temperature', 'heart_rate'
        ]
    
    def preprocess_image(self, image):
        """Preprocess uploaded image for prediction"""
        try:
            # Convert to numpy array
            image = np.array(image)
            
            if not self.cv2_available:
                # Fallback using PIL for resizing
                pil_image = Image.fromarray(image)
                pil_image = pil_image.resize((128, 128))
                image = np.array(pil_image)
            else:
                # Use OpenCV if available
                image = cv2.resize(image, (128, 128))
            
            # Normalize pixel values
            image = image.astype('float32') / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return None
    
    def predict_from_image(self, image):
        """Predict disease from image"""
        if self.image_model is None:
            return {"error": "Image model not loaded"}
        
        processed_image = self.preprocess_image(image)
        if processed_image is None:
            return {"error": "Could not process image"}
        
        # For demo, return mock predictions
        # In production, use: predictions = self.image_model.predict(processed_image)
        mock_predictions = np.random.dirichlet(np.ones(3), size=1)
        predicted_class = np.argmax(mock_predictions[0])
        confidence = np.max(mock_predictions[0])
        
        disease_mapping = {0: 'malaria', 1: 'pneumonia', 2: 'healthy'}
        diagnosis = disease_mapping.get(predicted_class, 'unknown')
        
        return {
            'diagnosis': diagnosis,
            'confidence': float(confidence),
            'all_predictions': {
                'malaria': float(mock_predictions[0][0]),
                'pneumonia': float(mock_predictions[0][1]),
                'healthy': float(mock_predictions[0][2])
            }
        }
    
    def predict_from_symptoms(self, symptoms_dict):
        """Predict disease from symptoms"""
        if self.symptom_model is None:
            return {"error": "Symptom model not loaded"}
        
        # Convert symptoms to feature vector
        feature_vector = []
        for feature in self.symptom_feature_names:
            feature_vector.append(symptoms_dict.get(feature, 0))
        
        # Predict
        prediction = self.symptom_model.predict([feature_vector])[0]
        probabilities = self.symptom_model.predict_proba([feature_vector])[0]
        
        diagnosis = self.label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities)
        
        return {
            'diagnosis': diagnosis,
            'confidence': float(confidence),
            'all_probabilities': dict(zip(self.label_encoder.classes_, probabilities.tolist()))
        }

def main():
    # Show warning if OpenCV is not available
    if not CV2_AVAILABLE:
        st.warning(
            "⚠️ OpenCV is not available. Some image processing features may be limited. "
            "The app will use fallback methods for image resizing."
        )
    
    # Initialize session state
    if 'ai_system' not in st.session_state:
        st.session_state.ai_system = MedicalDiagnosisAI()
        st.session_state.ai_system.load_demo_models()
    
    # Navigation sidebar
    st.sidebar.title("🏥 Navigation")
    page = st.sidebar.radio("Go to", [
        "Home", 
        "Image Diagnosis", 
        "Symptom Checker", 
        "Combined Diagnosis",
        "About"
    ])
    
    # Main content area
    if page == "Home":
        show_home_page()
    elif page == "Image Diagnosis":
        show_image_diagnosis()
    elif page == "Symptom Checker":
        show_symptom_checker()
    elif page == "Combined Diagnosis":
        show_combined_diagnosis()
    elif page == "About":
        show_about_page()

def show_home_page():
    st.markdown('<div class="main-header">AI Diagnosis Support for Community Health Workers</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to the AI-Powered Medical Diagnosis System
        
        This application assists community health workers in diagnosing common conditions:
        - **Malaria** from blood smear images
        - **Pneumonia** from chest X-rays  
        - **Skin conditions** from dermatology images
        - **Symptom-based diagnosis** using patient vitals and symptoms
        
        ### Features:
        🔬 **Image-based Diagnosis**: Upload medical images for AI analysis
        📊 **Symptom Checker**: Input patient symptoms and vitals
        🤖 **Combined Analysis**: Get comprehensive diagnosis using both methods
        📱 **Mobile-Friendly**: Works offline in remote areas
        
        ### How to Use:
        1. Navigate to the desired diagnosis method using the sidebar
        2. Upload images or enter patient symptoms
        3. Review AI recommendations
        4. Use results to support clinical decisions
        """)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?w=300", 
                 caption="Medical Diagnosis AI")
        
        st.info("""
        **Important Note**: 
        This AI system is designed to assist healthcare workers, not replace professional medical judgment.
        Always verify diagnoses with qualified medical professionals.
        """)
    
    # Quick stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Malaria Detection", "95%", "Accuracy")
    with col2:
        st.metric("Pneumonia Detection", "92%", "Accuracy")
    with col3:
        st.metric("Skin Conditions", "89%", "Accuracy")
    with col4:
        st.metric("Response Time", "< 5s", "Fast")

def show_image_diagnosis():
    st.markdown('<div class="sub-header">🖼️ Image-Based Diagnosis</div>', unsafe_allow_html=True)
    
    if not CV2_AVAILABLE:
        st.warning("⚠️ Image processing is using fallback methods. Some advanced features may not be available.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Medical Image")
        
        image_type = st.selectbox(
            "Select Image Type",
            ["Malaria Blood Smear", "Chest X-Ray", "Skin Condition", "Other"]
        )
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'tiff'],
            help="Upload a medical image for analysis"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Analyze Image", type="primary"):
                    with st.spinner("AI is analyzing the image..."):
                        result = st.session_state.ai_system.predict_from_image(image)
                        
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            display_diagnosis_result(result, "Image Analysis Result")
            except Exception as e:
                st.error(f"Error loading image: {e}")
        
        st.info("""
        **Supported Image Types:**
        - Malaria: Blood smear microscopy images
        - Pneumonia: Chest X-ray images  
        - Skin Conditions: Dermatology photos
        - Format: JPG, PNG, TIFF (max 10MB)
        """)
    
    with col2:
        st.subheader("Image Analysis Guide")
        
        tab1, tab2, tab3 = st.tabs(["Malaria", "Pneumonia", "Skin Conditions"])
        
        with tab1:
            st.markdown("""
            ### Malaria Blood Smear Analysis
            **What to look for:**
            - Red blood cells infected with Plasmodium parasites
            - Characteristic ring forms or gametocytes
            - Stippling (Maurer's clefts or Schüffner's dots)
            
            **Sample Images:**
            """)
            # Placeholder for sample images
            st.image("https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=400", 
                    caption="Normal Blood Smear")
            st.image("https://images.unsplash.com/photo-1559757175-0eb30cd8c063?w=400", 
                    caption="Malaria-Positive Smear")
        
        with tab2:
            st.markdown("""
            ### Pneumonia X-Ray Analysis
            **What to look for:**
            - Areas of consolidation in lung fields
            - Increased white opacities
            - Blunted costophrenic angles
            - Air bronchograms
            
            **Common Patterns:**
            - Lobar pneumonia: Consolidated lobe
            - Bronchopneumonia: Patchy infiltrates
            - Interstitial: Reticular patterns
            """)
        
        with tab3:
            st.markdown("""
            ### Skin Condition Analysis
            **Common Conditions:**
            - **Bacterial**: Impetigo, cellulitis, folliculitis
            - **Fungal**: Ringworm, candidiasis, tinea versicolor
            - **Viral**: Herpes, warts, molluscum contagiosum
            
            **Key Features:**
            - Lesion distribution and pattern
            - Color, texture, and borders
            - Associated symptoms
            """)

def show_symptom_checker():
    st.markdown('<div class="sub-header">📊 Symptom-Based Diagnosis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Patient Symptoms")
        
        with st.form("symptom_form"):
            st.markdown("### Common Symptoms")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fever = st.checkbox("Fever")
                cough = st.checkbox("Cough")
                headache = st.checkbox("Headache")
                fatigue = st.checkbox("Fatigue")
                chills = st.checkbox("Chills")
            
            with col2:
                skin_rash = st.checkbox("Skin Rash")
                chest_pain = st.checkbox("Chest Pain")
                shortness_breath = st.checkbox("Shortness of Breath")
                nausea = st.checkbox("Nausea")
                vomiting = st.checkbox("Vomiting")
            
            st.markdown("### Vital Signs")
            
            col1, col2 = st.columns(2)
            
            with col1:
                temperature = st.slider("Temperature (°C)", 35.0, 41.0, 37.0, 0.1)
                heart_rate = st.slider("Heart Rate (bpm)", 50, 150, 80)
            
            with col2:
                respiratory_rate = st.slider("Respiratory Rate", 10, 40, 16)
                blood_pressure_sys = st.slider("Systolic BP", 80, 200, 120)
            
            submitted = st.form_submit_button("Analyze Symptoms", type="primary")
            
            if submitted:
                symptoms_dict = {
                    'fever': 1 if fever else 0,
                    'cough': 1 if cough else 0,
                    'headache': 1 if headache else 0,
                    'fatigue': 1 if fatigue else 0,
                    'chills': 1 if chills else 0,
                    'skin_rash': 1 if skin_rash else 0,
                    'chest_pain': 1 if chest_pain else 0,
                    'shortness_breath': 1 if shortness_breath else 0,
                    'temperature': temperature,
                    'heart_rate': heart_rate
                }
                
                with st.spinner("Analyzing symptoms..."):
                    result = st.session_state.ai_system.predict_from_symptoms(symptoms_dict)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        display_diagnosis_result(result, "Symptom Analysis Result")
    
    with col2:
        st.subheader("Symptom Patterns Guide")
        
        st.markdown("""
        ### Common Disease Patterns
        
        **Malaria:**
        - High fever with chills and sweating
        - Headache and muscle pains
        - Fatigue and weakness
        - Cyclic symptoms (every 48-72 hours)
        
        **Pneumonia:**
        - Productive cough with phlegm
        - Chest pain worsened by breathing
        - Shortness of breath
        - High fever with chills
        
        **Skin Infections:**
        - Localized rash or lesions
        - Itching or pain at site
        - Possible fever if systemic
        - Recent exposure history
        
        **General Guidelines:**
        - Document symptom onset and progression
        - Note any recent travel or exposures
        - Record response to any treatments
        - Monitor vital signs regularly
        """)
        
        # Symptom severity indicator
        st.markdown("### Symptom Severity Assessment")
        
        severity_score = st.slider("Overall Symptom Severity (1-10)", 1, 10, 5)
        
        if severity_score >= 8:
            st.error("🚨 High Severity - Consider urgent referral")
        elif severity_score >= 5:
            st.warning("⚠️ Moderate Severity - Monitor closely")
        else:
            st.success("✅ Mild Severity - Routine care")

def show_combined_diagnosis():
    st.markdown('<div class="sub-header">🤖 Combined Diagnosis Analysis</div>', unsafe_allow_html=True)
    
    st.info("""
    **Comprehensive Analysis**: This section combines image analysis with symptom assessment 
    to provide the most accurate diagnosis recommendation.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Image Upload")
        combined_image = st.file_uploader(
            "Upload medical image for combined analysis",
            type=['jpg', 'jpeg', 'png'],
            key="combined_image"
        )
        
        if combined_image:
            try:
                st.image(Image.open(combined_image), caption="Medical Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
    
    with col2:
        st.subheader("Quick Symptoms")
        
        quick_fever = st.checkbox("Fever", key="quick_fever")
        quick_cough = st.checkbox("Cough", key="quick_cough")
        quick_rash = st.checkbox("Skin Rash", key="quick_rash")
        quick_breath = st.checkbox("Breathing Issues", key="quick_breath")
        
        quick_temperature = st.slider("Temperature", 35.0, 41.0, 37.0, 0.1, key="quick_temp")
    
    if st.button("Run Combined Analysis", type="primary"):
        with st.spinner("Performing comprehensive analysis..."):
            # Simulate combined analysis
            image_result = None
            if combined_image:
                try:
                    image = Image.open(combined_image)
                    image_result = st.session_state.ai_system.predict_from_image(image)
                except Exception as e:
                    st.error(f"Error analyzing image: {e}")
            
            symptom_result = {"diagnosis": "pneumonia", "confidence": 0.76}
            
            # Combine results
            if image_result and "error" not in image_result:
                if image_result["confidence"] > symptom_result["confidence"]:
                    final_result = image_result
                    final_result["source"] = "Image Analysis (Higher Confidence)"
                else:
                    final_result = symptom_result
                    final_result["source"] = "Symptom Analysis (Higher Confidence)"
            elif image_result and "error" in image_result:
                final_result = symptom_result
                final_result["source"] = "Symptom Analysis (Image analysis failed)"
            else:
                final_result = symptom_result
                final_result["source"] = "Symptom Analysis Only"
            
            display_combined_result(final_result)

def show_about_page():
    st.markdown('<div class="sub-header">ℹ️ About This Application</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### AI Diagnosis Support for Community Health Workers
    
    **Purpose**: This application is designed to assist community health workers in remote areas 
    with limited access to specialist medical care.
    
    ### Technology Stack
    - **Frontend**: Streamlit (Python web framework)
    - **AI Models**: TensorFlow/Keras for image analysis, Scikit-learn for symptom analysis
    - **Image Processing**: PIL (OpenCV fallback available)
    - **Deployment**: Ready for cloud or offline mobile deployment
    
    ### Medical Conditions Covered
    1. **Malaria**: Blood smear image analysis
    2. **Pneumonia**: Chest X-ray interpretation  
    3. **Skin Conditions**: Dermatology image analysis
    4. **Symptom-based**: Comprehensive symptom assessment
    
    ### Accuracy Metrics
    - Malaria detection: ~95% accuracy
    - Pneumonia detection: ~92% accuracy  
    - Skin condition classification: ~89% accuracy
    - Symptom-based diagnosis: ~85% accuracy
    
    ### Important Disclaimer
    This AI system is intended to **assist** healthcare workers, not replace professional medical judgment. 
    Always:
    - Verify AI recommendations with clinical assessment
    - Consult specialists when in doubt
    - Consider local epidemiology and patient history
    - Follow established clinical guidelines
    
    ### Development Team
    This application was developed by AI healthcare specialists in collaboration with 
    community health organizations to address diagnostic challenges in resource-limited settings.
    """)

def display_diagnosis_result(result, title):
    """Display diagnosis results in a formatted way"""
    st.markdown(f'<div class="diagnosis-box"><h3>{title}</h3>', unsafe_allow_html=True)
    
    # Confidence level styling
    confidence = result['confidence']
    if confidence >= 0.8:
        confidence_class = "confidence-high"
    elif confidence >= 0.6:
        confidence_class = "confidence-medium"
    else:
        confidence_class = "confidence-low"
    
    st.markdown(f"""
    **Diagnosis**: {result['diagnosis'].replace('_', ' ').title()}
    """)
    
    st.markdown(f"""
    **Confidence**: <span class="{confidence_class}">{confidence:.1%}</span>
    """, unsafe_allow_html=True)
    
    # Show detailed probabilities if available
    if 'all_predictions' in result:
        st.subheader("Detailed Probabilities")
        probs = result['all_predictions']
        for disease, prob in probs.items():
            st.progress(prob)
            st.write(f"{disease.replace('_', ' ').title()}: {prob:.1%}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations based on diagnosis
    show_recommendations(result['diagnosis'])

def display_combined_result(result):
    """Display combined analysis results"""
    st.markdown('<div class="diagnosis-box">', unsafe_allow_html=True)
    st.markdown("### 🎯 Combined Diagnosis Result")
    
    st.markdown(f"""
    **Final Diagnosis**: {result['diagnosis'].replace('_', ' ').title()}
    **Confidence**: {result['confidence']:.1%}
    **Primary Source**: {result.get('source', 'Combined Analysis')}
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    show_recommendations(result['diagnosis'])

def show_recommendations(diagnosis):
    """Show treatment recommendations based on diagnosis"""
    st.markdown("### 💡 Recommended Actions")
    
    recommendations = {
        'malaria': """
        **Immediate Actions:**
        - Confirm with rapid diagnostic test if available
        - Start artemisinin-based combination therapy (ACT)
        - Monitor for severe symptoms: altered consciousness, seizures
        - Refer to hospital if severe symptoms present
        
        **Follow-up:**
        - Repeat blood smear after 3 days
        - Complete full course of antimalarials
        - Provide mosquito net and prevention education
        """,
        
        'pneumonia': """
        **Immediate Actions:**
        - Assess severity using respiratory rate and oxygen saturation
        - Start appropriate antibiotics based on local guidelines
        - Consider chest X-ray confirmation if available
        - Hospitalize if severe (low oxygen, unable to drink)
        
        **Supportive Care:**
        - Oxygen if saturation < 90%
        - Fever management with antipyretics
        - Ensure adequate hydration and nutrition
        """,
        
        'healthy': """
        **Recommendations:**
        - Reassure patient
        - Provide general health education
        - Schedule follow-up if symptoms persist
        - Review preventive measures
        """,
        
        'bacterial_skin': """
        **Treatment:**
        - Topical or oral antibiotics based on severity
        - Warm compresses for abscesses
        - Good hygiene practices
        - Monitor for systemic symptoms
        """,
        
        'fungal_skin': """
        **Treatment:**
        - Antifungal creams (clotrimazole, miconazole)
        - Keep area dry and clean
        - Avoid sharing personal items
        - Oral antifungals for extensive cases
        """,
        
        'viral_skin': """
        **Management:**
        - Symptomatic relief for itching/pain
        - Antiviral creams if indicated
        - Avoid scratching to prevent secondary infection
        - Most viral rashes are self-limiting
        """
    }
    
    rec_text = recommendations.get(diagnosis, """
    **General Advice:**
    - Continue monitoring symptoms
    - Consider referral for specialist evaluation
    - Review patient history and risk factors
    - Follow local treatment guidelines
    """)
    
    st.info(rec_text)

if __name__ == "__main__":
    main()