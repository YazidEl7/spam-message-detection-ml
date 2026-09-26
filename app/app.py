import os
import pickle
import streamlit as st

# Define relative paths to model files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

# Load model and vectorizer
model = pickle.load(open(MODEL_PATH, "rb"))
vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

# Streamlit App Interface
st.title("Spam Email Detection System")

message = st.text_area("Enter Email Content")

if st.button("Predict"):
  if message.strip():
    vector = vectorizer.transform([message])
    prediction = model.predict(vector)

    if prediction[0] == 1:
      st.error("Spam Email")
    else:
      st.success("Legitimate Email")
  else:
    st.warning("Please enter some text before predicting.")
