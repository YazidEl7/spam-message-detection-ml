import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from preprocessing import clean_text


def train_models():
  # 1. Load dataset
  data_path = os.path.join("data", "spam.csv")
  df = pd.read_csv(data_path, encoding="latin-1")
  df = df[["label", "message"]]

  # Map labels: ham -> 0, spam -> 1
  df["label"] = df["label"].map({"ham": 0, "spam": 1})

  # Drop missing values if any
  df.dropna(inplace=True)

  # 2. Preprocess text
  df["clean_message"] = df["message"].apply(clean_text)

  # 3. Features and Targets
  X = df["clean_message"]
  y = df["label"]

  # 4. Train/Test Split
  X_train_text, X_test_text, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  # 5. TF-IDF Vectorization
  vectorizer = TfidfVectorizer(max_features=3000)
  X_train = vectorizer.fit_transform(X_train_text)
  X_test = vectorizer.transform(X_test_text)

  # 6. Initialize & Train Models
  models = {
      "multinomial_nb": MultinomialNB(),
      "logistic_regression": LogisticRegression(random_state=42),
      "linear_svc": LinearSVC(C=1.0, random_state=42),
  }

  os.makedirs("models", exist_ok=True)

  # Save vectorizer
  joblib.dump(vectorizer, os.path.join("models", "tfidf_vectorizer.pkl"))

  for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join("models", f"{name}.pkl"))
    print(f"Model saved: models/{name}.pkl")

  # Save test sets for evaluation
  joblib.dump((X_test, y_test), os.path.join("models", "test_data.pkl"))
  print("Training complete.")


if __name__ == "__main__":
  train_models()
