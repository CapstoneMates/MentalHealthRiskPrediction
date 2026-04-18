# 🧠 Hybrid Mental Health Risk Prediction Using Unpaired Heterogeneous Data Sources with Anomaly-Aware Fusion and Explainable AI

## 📌 Overview

This project presents a hybrid mental health risk prediction system
integrating textual and behavioral data from independent sources. The
system performs decision-level fusion and provides interpretable
outputs.

## 🎯 Motivation

Most systems rely on single modality or aligned data. This project
handles unpaired heterogeneous data, reflecting real-world scenarios.

## 📊 Datasets

### Reddit Depression Dataset

-   Text data
-   Target: is_depression

### Student Mental Health Dataset

-   Behavioral/tabular data
-   Target: Depression indicator

## ⚙️ Methodology

-   Text Model: Logistic Regression + TF-IDF
-   Tabular Model: Logistic Regression
-   Anomaly Detection: Autoencoder
-   Fusion: Weighted decision-level fusion
-   Explainability: Feature contribution analysis

## 🧪 Evaluation

-   Text Accuracy: \~95%
-   Tabular Accuracy: \~80%
-   Metrics: Accuracy, Precision, Recall, F1, ROC-AUC

## 🖥️ System Implementation

-   Streamlit-based application
-   Inputs: Text + behavioral data
-   Outputs: Risk score, level, anomaly flag, explanations

## ⚠️ Limitations

-   Small tabular dataset
-   Unpaired data sources
-   Decision-level fusion

## 🔬 Future Work

-   Transformer models (BERT)
-   Larger datasets
-   Learned fusion weights

