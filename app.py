
import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
from features import EmailFeatureTransformer 
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_pipeline.joblib")

st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model artifacts not found. Please run `python train.py` first.")
        return None
    return joblib.load(MODEL_PATH)

def explain_prediction(model, text, top_n=4):
    features_step = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    
    features_transformed = features_step.transform([text])
    feature_names = features_step.get_feature_names_out()
    
    coeffs = classifier.coef_[0]
    contributions = features_transformed.toarray()[0] * coeffs
    
    sorted_idx = np.argsort(contributions)
    
    negative = [{"feature": feature_names[i], "score": float(contributions[i])} for i in sorted_idx[:top_n]]
    positive = [{"feature": feature_names[i], "score": float(contributions[i])} for i in sorted_idx[-top_n:][::-1]]
    
    return positive, negative

st.title("🛡️ Phishing Email Detector")
st.markdown("Enter the content of an email below to analyze it for phishing risks.")

model = load_model()

email_text = st.text_area("Email Content:", placeholder="Paste email text here...", height=200)

if st.button("Analyze Email"):
    if not email_text.strip():
        st.warning("Please enter some text to analyze.")
    elif model:
        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]
        
        result = "Phishing" if prediction == 1 else "Legitimate"
        confidence = max(probabilities) * 100
        
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Prediction", result)
        with col2:
            st.metric("Confidence", f"{confidence:.2f}%")
            
        if prediction == 1:
            st.error("⚠️ This email looks like a Phishing attempt.")
        else:
            st.success("✅ This email appears to be Legitimate.")

        st.subheader("Key Indicators")
        pos_features, neg_features = explain_prediction(model, email_text)
        
        expl_col1, expl_col2 = st.columns(2)
        with expl_col1:
            st.write("**Top Phishing Signals:**")
            for item in pos_features:
                if item['score'] > 0:
                    st.write(f"- `{item['feature']}`")
        
        with expl_col2:
            st.write("**Top Legitimate Signals:**")
            for item in neg_features:
                if item['score'] < 0:
                    st.write(f"- `{item['feature']}`")