import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('saved_models/optimized_xgboost.pkl', 'rb'))
        scaler = joblib.load('saved_models/scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, scaler = load_model()

def main():
    st.title("Credit Card Fraud Detection System")
    st.write("This application uses machine learning to detect fraudulent credit card transactions.")

    # Create two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Transaction Details")
        
        # Create form for input
        with st.form("transaction_form"):
            # Time and Amount in the same row
            col_time, col_amount = st.columns(2)
            with col_time:
                time = st.number_input("Time (in seconds)", min_value=0)
            with col_amount:
                amount = st.number_input("Transaction Amount", min_value=0.0)

            # Create 4 rows with 7 columns each for V1-V28
            features = {}
            for i in range(0, 28, 7):
                cols = st.columns(7)
                for j in range(7):
                    if i + j < 28:  # To avoid going beyond V28
                        features[f'V{i+j+1}'] = cols[j].number_input(
                            f'V{i+j+1}',
                            format="%.6f"
                        )

            submitted = st.form_submit_button("Predict")

    # If form is submitted
    if submitted:
        try:
            # Create feature vector
            feature_vector = []
            # Add V1-V28
            for i in range(1, 29):
                feature_vector.append(features[f'V{i}'])
            # Add Amount and Time
            feature_vector.extend([amount, time])
            
            # Convert to DataFrame and scale
            features_df = pd.DataFrame([feature_vector], 
                                    columns=[f'V{i+1}' for i in range(28)] + ['Amount', 'Time'])
            scaled_features = scaler.transform(features_df)
            
            # Make prediction
            prediction = int(model.predict(scaled_features)[0])
            probability = float(model.predict_proba(scaled_features)[0][1])
            
            # Display results in the second column
            with col2:
                st.subheader("Prediction Results")
                
                # Create a card-like container for the results
                with st.container():
                    st.markdown("""
                    <style>
                    .prediction-card {
                        padding: 20px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    if prediction == 1:
                        st.error("⚠️ Fraudulent Transaction Detected!")
                    else:
                        st.success("✅ Legitimate Transaction")
                    
                    st.metric("Fraud Probability", f"{probability:.2%}")
                    st.write(f"Prediction made at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

if __name__ == '__main__':
    main()
