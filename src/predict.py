import os
import joblib
from preprocessing import clean_text


def predict_message(message: str, model_name: str = "multinomial_nb") -> str:
  """Preprocesses input text, transforms using trained TF-IDF vectorizer,

  and returns classification ('Spam' or 'Legitimate').
  """
  vec_path = os.path.join("models", "tfidf_vectorizer.pkl")
  model_path = os.path.join("models", f"{model_name}.pkl")

  if not os.path.exists(vec_path) or not os.path.exists(model_path):
    raise FileNotFoundError("Model or vectorizer pkl files missing in models/")

  vectorizer = joblib.load(vec_path)
  model = joblib.load(model_path)

  # Preprocess
  cleaned = clean_text(message)

  # Transform
  vectorized = vectorizer.transform([cleaned])

  # Predict
  prediction = model.predict(vectorized)[0]

  return "Spam" if prediction == 1 else "Legitimate"


if __name__ == "__main__":
  sample = "Free entry in 2 a wkly comp to win FA Cup final tkts!"
  print(f"Sample Message: {sample}")
  print(f"Prediction: {predict_message(sample)}")
