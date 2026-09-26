import os
import sys
import joblib
import streamlit as st

# Add src folder to module import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocessing import clean_text

st.set_page_config(page_title="Spam Message Detection", page_icon="✉️")

st.title("✉️ Spam Message Detection App")
st.write("Enter a SMS or email message below to classify it as Spam or Legitimate.")

# Load resources
@st.cache_resource
def load_assets():
    vec_path = os.path.join("models", "tfidf_vectorizer.pkl")
    model_path = os.path.join("models", "multinomial_nb.pkl")
    
    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)
    return vectorizer, model

try:
    vectorizer, model = load_assets()

    message = st.text_area("Enter a message:", height=120)

    if st.button("Predict"):
        if message.strip() == "":
            st.warning("Please enter a valid text message.")
        else:
            # Preprocess
            cleaned = clean_text(message)
            
            # Vectorize
            transformed = vectorizer.transform([cleaned])
            
            # Predict
            prediction = model.predict(transformed)[0]

            if prediction == 1:
                st.error("🚨 **Spam Message Detected!**")
            else:
                st.success("✅ **Legitimate (Ham) Message**")

except FileNotFoundError:
    st.error("Model files not found. Please train models using `python src/train.py` first.")
