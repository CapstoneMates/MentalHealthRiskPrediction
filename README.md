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

### Feature Selection
The system processes two distinct sets of inputs to produce specific classification and detection outputs.

### **Case 1: Text Data (Social Media Posts)**

* **Input Features:** The primary feature is the `clean_text` column from the Reddit dataset. This raw text is transformed into a numerical matrix using **TF-IDF Vectorization**
    * **Vocabulary:** The top 10,000 most frequent words.
    * **Context:** It captures both single words (unigrams) and two-word combinations (bigrams).
* **Output:** A binary classification label (`is_depression`).
    * **0:** Not Depressed.
    * **1:** Depressed.
    * **Probability:** During inference, it outputs a decimal (0.0 to 1.0) representing the risk level.

---

### **Case 2: Tabular Data (Student Surveys)**

* **Input Features:** The model uses several lifestyle and demographic features extracted from the survey:
    1.  **Gender**
    2.  **Age**
    3.  **Course of Study**
    4.  **Current Year of Study**
    5.  **CGPA**
    6.  **Marital Status**
    7.  **Anxiety Status**
    8.  **Panic Attack History**
    9.  **Specialist Treatment History**
* **Outputs:** This case generates two different types of outputs:
    1.  **Classification (`Do you have Depression?`):** A binary prediction (Yes/No) of depression risk based on the features above.
    2.  **Anomaly Detection:** A "Reconstruction Error" score. If the error is higher than the calculated **threshold of ~1.20**, the output flags the data as an "Anomaly," meaning the user's profile is highly unusual compared to the average student.

### **Summary Table**

| Data Type | Features Used | Final Output |
| :--- | :--- | :--- |
| **Text** | 10,000 TF-IDF Word Patterns | Depression Probability (0-1) |
| **Tabular** | Demographics + Lifestyle + Health History | Depression Probability (0-1) |
| **Anomaly** | Scaled Survey Features | Anomaly Flag (Normal vs. Unusual) |

## ⚙️ Methodology

-   Text Model: Logistic Regression + TF-IDF
-   Tabular Model: Logistic Regression
-   Anomaly Detection: Autoencoder
-   Fusion: Weighted decision-level fusion
-   Explainability: Feature contribution analysis

## 🧪 Evaluation

-   Text Accuracy: \~95%
-   Tabular Accuracy: \~85%
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

