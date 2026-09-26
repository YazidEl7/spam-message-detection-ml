# Spam Message Detection Using Machine Learning

A machine learning project for classifying text messages as **spam** or **legitimate** using Natural Language Processing and supervised learning.

The project was developed as a Master's Final Year Project and focuses on the complete workflow from text preprocessing and feature extraction to model evaluation and deployment.

> **Project scope:** The experimental evaluation uses the SMS Spam Collection Dataset. The system is therefore a text-based spam message classification prototype. The same pipeline could be adapted to basic email-body classification, but it is not a complete email security solution.

---

## Overview

Spam messages are a common problem in digital communication and can contain unwanted advertisements, misleading content, phishing attempts, or other potentially harmful information.

This project investigates whether traditional machine learning algorithms can effectively classify messages using their textual content.

The implemented pipeline includes:

- Text cleaning and preprocessing
- Stopword removal
- Porter stemming
- TF-IDF feature extraction
- Training of multiple classification models
- Model evaluation using several metrics
- Comparison of model performance
- Deployment through a Streamlit web application

---

## Project Workflow

```text
SMS Dataset
     │
     ▼
Data Exploration
     │
     ▼
Text Cleaning
     │
     ▼
Stopword Removal + Stemming
     │
     ▼
TF-IDF Feature Extraction
     │
     ▼
Train / Test Split
     │
     ├──────────────┬──────────────┐
     ▼              ▼              ▼
Naïve Bayes   Logistic Regression   Linear SVM
     │              │              │
     └──────────────┴──────────────┘
                    │
                    ▼
              Model Evaluation
                    │
                    ▼
              Best Model Selected
                    │
                    ▼
             Streamlit Application
```
## Dataset

The project uses the SMS Spam Collection Dataset.

The dataset contains two classes:

- ham — legitimate messages
- spam — unwanted messages

The dataset is not included in this repository. See data/README.md for information about obtaining and preparing the dataset.

## Text Preprocessing

The text preprocessing pipeline includes:

- Converting text to lowercase
- Removing URLs
- Removing numbers and punctuation
- Removing stopwords
- Applying stemming

The goal is to reduce unnecessary variation in the text before feature extraction.

## Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text messages into numerical features.

TF-IDF gives higher importance to terms that are useful for distinguishing messages while reducing the importance of terms that occur frequently throughout the dataset.

The vocabulary is limited to 3,000 features in the experimental configuration.

## Machine Learning Models:

Three machine learning models were evaluated:

- Multinomial Naïve Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

The models were trained using the same TF-IDF representation and evaluated using the same test set.

## Evaluation:

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The final model was selected based on the experimental results.

See results/results.md for the detailed results.

Streamlit Application

The final model is integrated into a Streamlit interface where a user can enter a message and receive a prediction.

Possible output classes are:

- Spam
- Legitimate

Interface

Prediction Example

## Project Structure:
```
spam-message-detection-ml/
│
├── data/              # Dataset instructions
├── src/               # Preprocessing, training and evaluation
├── app/               # Streamlit application
├── models/            # Trained model information
├── results/           # Evaluation results and figures
├── screenshots/       # Application screenshots
└── docs/              # Project report and presentation
```
## Installation:
Clone the repository:
```
git clone https://github.com/YAZIDEL7/spam-message-detection-ml.git
cd spam-message-detection-ml
```
Create a virtual environment:
```
python -m venv venv
```
Activate it on Windows:
```
venv\Scripts\activate
```
Install the required packages:
```
pip install -r requirements.txt
```
## Training:

After preparing the dataset, run:
```
python src/train.py
```
## Evaluation:

To evaluate the trained models:
```
python src/evaluate.py
```
## Running the Application:

Start the Streamlit application:
```
streamlit run app/app.py
```
## Limitations:

This project focuses on text-based message classification.

It does not currently analyze:

- Email headers
- Sender reputation
- Attachments
- URL reputation
- SPF/DKIM information
- Domain reputation
- Other email metadata

The experimental results are based on SMS messages and should therefore not be interpreted as a complete evaluation of an email filtering system.

## Future Work:

Possible improvements include:

- Testing the system on a real email spam dataset
- Adding URL analysis
- Using email header information
- Adding sender and domain reputation features
- Detecting image-based spam
- Testing word embeddings or transformer-based models
- Integrating the classifier with an email gateway

## Technologies:
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Streamlit

Author:
YazidEL7

Master's Project — Cybersecurity
