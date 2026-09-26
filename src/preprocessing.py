import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure stopwords are available
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def clean_text(text: str) -> str:
  """Exact text cleaning and stemming logic extracted from notebook:

  - Lowercasing
  - URL removal (http/www)
  - Number removal
  - Punctuation removal
  - Stopword filtering
  - Porter Stemming
  """
  if not isinstance(text, str):
    return ""

  # Lowercase
  text = text.lower()

  # Remove URLs
  text = re.sub(r"http\S+|www\S+", "", text)

  # Remove numbers
  text = re.sub(r"\d+", "", text)

  # Remove punctuation
  text = re.sub(r"[^\w\s]", "", text)

  # Tokenize and remove stopwords
  words = text.split()
  words = [word for word in words if word not in stop_words]

  # Apply Stemming
  words = [stemmer.stem(word) for word in words]

  return " ".join(words)
