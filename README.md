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
