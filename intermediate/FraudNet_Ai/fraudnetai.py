import streamlit as st
import pickle
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import os
import sklearn  # Include sklearn for model loading support
import re

# Page configuration
st.set_page_config(
    page_title="FraudNet - Fraud Detection Suite",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-bottom: 1rem;
    }
    .description {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
        text-align: center;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #F3F4F6;
        margin-bottom: 2rem;
        border-left: 5px solid #3B82F6;
    }
    .success-message {
        padding: 1rem;
        border-radius: 5px;
        background-color: #DCFCE7;
        border-left: 5px solid #22C55E;
        color: #166534;
    }
    .warning-message {
        padding: 1rem;
        border-radius: 5px;
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        color: #991B1B;
    }
    </style>
""", unsafe_allow_html=True)

# Custom class to handle NumPy array models
class ArrayPredictor:
    def __init__(self, array):
        self.array = array
    
    def predict(self, X):
        # For binary classification, use 0.5 as threshold
        # This assumes the saved array contains probabilities or scores
        if len(self.array.shape) == 1:  # 1D array
            return (self.array > 0.5).astype(int)
        elif len(self.array.shape) == 2:  # 2D array (each row is a sample)
            return (self.array[:, 1] > 0.5).astype(int)
        else:
            raise ValueError("Unsupported array shape for prediction")
    
    def predict_proba(self, X):
        # If the array is already probabilities
        if len(self.array.shape) == 1:  # 1D array
            # Convert to 2D array with probabilities for both classes
            probs = np.zeros((len(self.array), 2))
            probs[:, 0] = 1 - self.array
            probs[:, 1] = self.array
            return probs
        elif len(self.array.shape) == 2:  # Already 2D
            return self.array
        else:
            raise ValueError("Unsupported array shape for prediction probabilities")

# Try different approaches to load the models
def load_models():
    phishing_model = None
    cc_fraud_model = None
    
    # Debug information
    st.sidebar.markdown("### Model Loading Debug")
    
    # List files in current directory
    files = os.listdir(".")
    st.sidebar.write("Files in directory:", files)
    
    # Try to load the phishing model
    phishing_path = "phishing_model.pkl"
    if os.path.exists(phishing_path):
        try:
            # Try regular pickle
            with open(phishing_path, 'rb') as file:
                phishing_model = pickle.load(file)
            st.sidebar.success(f"Loaded phishing model: {type(phishing_model)}")
        except Exception as e1:
            st.sidebar.error(f"Error loading with pickle: {e1}")
            
    else:
        st.sidebar.warning(f"Phishing model file not found: {phishing_path}")
    
    # Try to load the credit card fraud model
    cc_path = "creditcrad_model.pkl"
    if os.path.exists(cc_path):
        try:
            # Try regular pickle
            with open(cc_path, 'rb') as file:
                cc_fraud_model = pickle.load(file)
            st.sidebar.success(f"Loaded CC model: {type(cc_fraud_model)}")
        except Exception as e1:
            st.sidebar.error(f"Error loading with pickle: {e1}")
    else:
        st.sidebar.warning(f"Credit card model file not found: {cc_path}")
    
    # Check if models are NumPy arrays instead of sklearn estimators
    # If so, wrap them in a simple predictor class
    if phishing_model is not None and isinstance(phishing_model, np.ndarray):
        st.sidebar.info("Converting phishing model from array to predictor")
        phishing_model = ArrayPredictor(phishing_model)
    
    if cc_fraud_model is not None and isinstance(cc_fraud_model, np.ndarray):
        st.sidebar.info("Converting CC model from array to predictor")
        cc_fraud_model = ArrayPredictor(cc_fraud_model)
    
    return phishing_model, cc_fraud_model

# Function to extract features from URL for phishing detection
def extract_url_features(url):
    # Implement the same feature extraction as used during training
    # Based on the provided reference code
    return simple_url_features(url)

def simple_url_features(url):
    """Extract features from a URL"""
    try:
        parsed_url = urlparse(url)
        
        features = {
            'url_length': len(url),
            'count_digits': sum(c.isdigit() for c in url),
            'count_letters': sum(c.isalpha() for c in url),
            'count_special_chars': len(re.findall(r'[^\w]', url)),
            'count_dots': url.count('.'),
            'has_https': int('https' in url),
            'has_http': int('http' in url),
            'has_at': int('@' in url),
            'has_hyphen': int('-' in url),
            'has_double_slash': int('//' in url),
            'has_suspicious_words': int(any(word in url.lower() for word in [
                'login', 'secure', 'bank', 'account', 'verify', 'update'
            ]))
        }

        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None
    
def predict_phishing(url, model):
    """Predict if a URL is phishing using the loaded model"""
    try:
        # Extract features from URL
        features = simple_url_features(url)
        if features is None:
            return None, None
            
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        result = "Phishing" if prediction == 1 else "Legitimate"
        
        # Try to get probability if available
        try:
            proba = model.predict_proba(features_df)[0]
            confidence = proba[1] if len(proba) > 1 else proba
        except (AttributeError, IndexError, TypeError):
            confidence = None
            
        return result, confidence
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

def predict_cc_fraud(transaction_data, model):
    """Predict if a credit card transaction is fraudulent using the loaded model"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        result = "Fraudulent" if prediction == 1 else "Legitimate"
        
        # Try to get probability if available
        try:
            proba = model.predict_proba(df)[0]
            confidence = proba[1] if len(proba) > 1 else proba
        except (AttributeError, IndexError, TypeError):
            confidence = None
            
        return result, confidence
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

# Main app function
def main():
    # Load the models
    phishing_model, cc_fraud_model = load_models()
    
    # Header
    st.markdown('<div class="main-header">FraudNet</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Advanced Fraud Detection Suite</div>', unsafe_allow_html=True)
    
    # Navigation tabs
    tab1, tab2 = st.tabs(["🔍 Phishing Detection", "💳 Credit Card Fraud Detection"])
    
    # Phishing Detection tab
    with tab1:
        st.markdown('<div class="sub-header">Phishing URL Detection</div>', unsafe_allow_html=True)
        st.markdown('Enter a website URL to check if it might be a phishing attempt.')
        
        with st.form("phishing_form"):
            url = st.text_input("Enter Website URL:", placeholder="https://example.com")
            submit_phishing = st.form_submit_button("Check URL")
        
        if submit_phishing and url:
            if phishing_model is not None:
                with st.spinner('Analyzing URL...'):
                    try:
                        # Extract features from URL
                        features = extract_url_features(url)
                        features_df = pd.DataFrame([features])
                        
                        # Make prediction
                        prediction = phishing_model.predict(features_df)[0]
                        
                        # Try to get probabilities if the method exists
                        try:
                            probability = phishing_model.predict_proba(features_df)[0]
                            prob_value = probability[1] if len(probability) > 1 else probability
                        except (AttributeError, IndexError):
                            # If predict_proba doesn't exist or returns unexpected format
                            prob_value = 0.5  # Default value
                        
                        if prediction == 1:
                            st.markdown(f'''
                            <div class="warning-message">
                                <b>⚠️ Potential Phishing Detected!</b><br>
                                This URL shows characteristics commonly associated with phishing attempts.
                                Confidence: {prob_value:.2%}
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'''
                            <div class="success-message">
                                <b>✅ URL Appears Safe</b><br>
                                No phishing indicators detected.
                                Confidence: {1-prob_value:.2%}
                            </div>
                            ''', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error processing URL: {str(e)}")
                        st.info("Please check that the URL format is correct and try again.")
            else:
                st.error("Phishing detection model is not available. Please check if the model file exists.")
    
    # Credit Card Fraud Detection tab
    with tab2:
        st.markdown('<div class="sub-header">Credit Card Fraud Detection</div>', unsafe_allow_html=True)
        st.markdown('Enter transaction details to check for potential fraud.')
        
        with st.form("cc_fraud_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                time = st.number_input("Time (seconds from reference)", min_value=0.0)
                v1 = st.number_input("V1", format="%.6f")
                v2 = st.number_input("V2", format="%.6f")
                v3 = st.number_input("V3", format="%.6f")
            
            with col2:
                v4 = st.number_input("V4", format="%.6f")
                v5 = st.number_input("V5", format="%.6f")
                amount = st.number_input("Amount ($)", min_value=0.0, format="%.2f")
            
            submit_cc = st.form_submit_button("Check Transaction")
        
        if submit_cc:
            if cc_fraud_model is not None:
                with st.spinner('Analyzing transaction...'):
                    # Prepare data for model
                    transaction_data = {
                        'Time': time,
                        'V1': v1, 'V2': v2, 'V3': v3, 'V4': v4, 'V5': v5,
                        'Amount': amount
                    }
                    
                    # Convert to DataFrame
                    df = pd.DataFrame([transaction_data])
                    
                    try:
                        # Make prediction
                        prediction = cc_fraud_model.predict(df)[0]
                        
                        # Try to get probabilities if the method exists
                        try:
                            probability = cc_fraud_model.predict_proba(df)[0]
                            prob_value = probability[1] if len(probability) > 1 else probability
                        except (AttributeError, IndexError):
                            # If predict_proba doesn't exist or returns unexpected format
                            prob_value = 0.5  # Default value
                        
                        if prediction == 1:
                            st.markdown(f'''
                            <div class="warning-message">
                                <b>🚨 Potential Fraud Detected!</b><br>
                                This transaction shows patterns consistent with fraudulent activity.
                                Fraud probability: {prob_value:.2%}
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'''
                            <div class="success-message">
                                <b>✅ Transaction Appears Legitimate</b><br>
                                No fraud indicators detected.
                                Legitimate probability: {1-prob_value:.2%}
                            </div>
                            ''', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error processing transaction: {str(e)}")
                        st.info("Please check your input values and try again.")
            else:
                st.error("Credit card fraud detection model is not available. Please check if the model file exists.")
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 FraudNet - Advanced Fraud Detection")

if __name__ == "__main__":
    main()