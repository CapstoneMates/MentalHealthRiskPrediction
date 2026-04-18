The project leverages two distinct datasets—**unstructured social media text** and **structured survey data**—by employing a **Decision-Level Fusion** strategy. Instead of mixing the data into one massive table (which often causes "noise" due to the different nature of words vs. numbers), the system processes them through independent pipelines and then combines their outputs.

Here is exactly how that fusion works:

### 1. Independent Feature Extraction (The "Separation" Phase)
Before any fusion happens, each dataset is treated according to its unique structure:
* **The Text Dataset (Reddit Posts):** This data is high-dimensional. We use **TF-IDF Vectorization** (with unigrams and bigrams) to turn sentences into a mathematical matrix of word importance.
* **The Tabular Dataset (Student Surveys):** This data is low-dimensional but dense. We use **Label Encoding** for categories (like "Yes/No" to 1/0) and **Standard Scaling** for numbers (like Age or CGPA) so the model doesn't get biased by larger scales.

### 2. Specialized Modeling (The "Expert" Phase)
We train two separate **XGBoost Classifiers**. This is critical because:
* The **Text Model** becomes an "expert" at recognizing linguistic patterns of depression (certain keywords, tone, or sentiment).
* The **Tabular Model** becomes an "expert" at recognizing lifestyle risk factors (academic pressure, financial status, or sleep patterns).

### 3. Fusing for Decision Making (The "Integration" Phase)
The fusion occurs at the **Prediction Level** rather than the data level. When a new person is being evaluated, the system follows this logic:

1.  **Parallel Scoring:** The text input is fed to the Text XGBoost, and the survey data is fed to the Tabular XGBoost.
2.  **Probability Generation:** Instead of just saying "Depressed" or "Not Depressed," each model outputs a **Probability Score** (e.g., Text Model says 85% risk, Tabular Model says 60% risk).
3.  **Weighted Voting / Consensus:** The system weighs these probabilities. If *either* model detects a high risk, or if their *combined average* passes a certain threshold, the final decision is triggered. 
4.  **Anomaly Layer (The Safety Net):** Simultaneously, the **Autoencoder** (Neural Network) looks at the tabular data. If the user's data is so "unusual" that the Autoencoder can't reconstruct it (high MSE), it flags an **Anomaly**.

### Why do it this way?
* **Higher Accuracy:** The Text model handles the complexity of language (95.5% accuracy), while the Tabular model provides context that text might miss.
* **Handling Imbalance:** By using `scale_pos_weight` in both XGBoost models, we ensure that even if the datasets are small or imbalanced, both "experts" are trained to prioritize finding the mental health risks accurately.
* **Robustness:** If a user provides a survey but no social media text (or vice versa), the system can still make a partial decision based on the available "expert" model.