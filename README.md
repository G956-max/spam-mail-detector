# 📧 Spam Mail Detector

A Machine Learning based Spam Mail Detector that classifies text messages as **Spam** or **Ham (Non-Spam)** using Natural Language Processing (NLP) techniques.

## 🎯 Objective

The objective of this project is to build a classifier that can distinguish between spam and non-spam messages using textual data.

## 📊 Dataset

This project uses the **SMS Spam Collection Dataset**.

The dataset contains SMS messages labelled as:

- `spam` – Unwanted or promotional messages
- `ham` – Normal messages

Dataset format:

| Label | Message |
|-------|---------|
| ham | Hey, are you coming to college today? |
| spam | Congratulations! You won a free lottery ticket. |

## 🔄 Project Workflow

```text
Dataset
   ↓
Text Preprocessing
   ↓
Lowercasing
   ↓
Stopword Removal
   ↓
TF-IDF Feature Extraction
   ↓
Train/Test Split
   ↓
Naive Bayes Classifier
   ↓
Prediction
   ↓
Performance Evaluation
