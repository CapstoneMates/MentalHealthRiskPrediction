### **Artifact Definitions**

* **`text_model.pkl`**: This is the trained XGBoost classifier that has learned to identify depression patterns in text. During inference, it takes the processed text and outputs the probability of depression risk.

* **`tabular_model.pkl`**: This is the trained XGBoost classifier for survey data. It analyzes lifestyle factors (like CGPA and Marital Status) to predict risk.

* **`tfidf_vectorizer.pkl`**: This contains the vocabulary (top 10,000 words) and the mathematical weights learned during training. We cannot feed raw text into the model; we must use this specific vectorizer to transform new text into the exact numerical format the model expects.

* **`scaler.pkl`**: This stores the mean and standard deviation of our training features. It ensures that when a user enters their age or CGPA, those numbers are scaled exactly like the data the model was trained on.

* **`label_encoders.pkl`**: This dictionary stores the mappings for categorical text (e.g., "Yes" = 1, "No" = 0). It is used to convert a user's dashboard "Yes/No" selections into the integers the model understands.

* **`feature_columns.pkl`**: This is a list of the exact column names and their order. It ensures the dashboard sends the data to the model in the correct sequence (e.g., Age first, then Gender)

* **`threshold.pkl`**: This is the specific numerical value (the 95th percentile of error) calculated by our Autoencoder. It is the "cutoff" used to decide if a user's input is an anomaly.
* **`autoencoder_model.h5`**: This is the deep learning neural network (saved in HDF5 format). It is used to reconstruct the tabular data to check for unusual patterns.

---

### **Which Files are Needed for Inference?**

To run the dashboard correctly, we need **all of them**. Here is how they are used in the inference workflow:

| Step | Task | Artifacts Needed |
| :--- | :--- | :--- |
| **1. Categorical Prep** | Convert "Male/Female" or "Yes/No" to numbers. | `label_encoders.pkl` |
| **2. Text Prep** | Convert the user's post into a TF-IDF matrix. | `tfidf_vectorizer.pkl` |
| **3. Numerical Prep** | Scale age and CGPA so they aren't "too large." | `scaler.pkl` |
| **4. Feature Alignment** | Ensure data is in the right order. | `feature_columns.pkl` |
| **5. Text Scoring** | Get the 75% weighted risk score. | `text_model.pkl` |
| **6. Tabular Scoring** | Get the 25% weighted risk score. | `tabular_model.pkl` |
| **7. Anomaly Check** | Reconstruct data and compare against the limit. | `autoencoder_model.h5` and `threshold.pkl` |