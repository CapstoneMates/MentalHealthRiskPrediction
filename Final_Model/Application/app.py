import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import time

st.set_page_config(
    page_title="Mental Health Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f6f9fc 0%, #eef4ff 100%);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.hero-card {
    background: linear-gradient(135deg, #f59e0b 0%, #8b5cf6 45%, #22d3ee 100%);
    padding: 2rem;
    border-radius: 24px;
    color: white;
    box-shadow: 0 10px 30px rgba(37, 99, 235, 0.22);
    margin-bottom: 1.5rem;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}
.hero-subtitle {
    font-size: 1rem;
    opacity: 0.95;
    line-height: 1.6;
}
.section-card {
    background: white;
    padding: 1.25rem;
    border-radius: 20px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    margin-bottom: 1rem;
    border: 1px solid rgba(148, 163, 184, 0.18);
}
.metric-card {
    background: white;
    padding: 1rem 1.2rem;
    border-radius: 18px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.16);
    text-align: center;
}
.metric-title {
    font-size: 0.95rem;
    color: #475569;
    margin-bottom: 0.4rem;
    font-weight: 600;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #0f172a;
}
.result-card-low {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    padding: 1.4rem;
    border-radius: 22px;
    border-left: 8px solid #16a34a;
    box-shadow: 0 8px 24px rgba(22, 163, 74, 0.12);
}
.result-card-medium {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    padding: 1.4rem;
    border-radius: 22px;
    border-left: 8px solid #d97706;
    box-shadow: 0 8px 24px rgba(217, 119, 6, 0.12);
}
.result-card-high {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    padding: 1.4rem;
    border-radius: 22px;
    border-left: 8px solid #dc2626;
    box-shadow: 0 8px 24px rgba(220, 38, 38, 0.12);
}
.result-title {
    font-size: 1.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    color: #111827;
}
.result-text {
    font-size: 1rem;
    color: #1f2937;
    line-height: 1.6;
}
.small-note {
    color: #475569;
    font-size: 0.92rem;
}
.stButton > button {
    width: 100%;
    border-radius: 14px;
    padding: 0.8rem 1rem;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 45%, #60a5fa 100%);
    color: white;
    box-shadow: 0 10px 20px rgba(37, 99, 235, 0.22);
}
.stButton > button:hover {
    filter: brightness(1.05);
}

</style>
""", unsafe_allow_html=True) 






st.markdown("""
<div class="hero-card">
    <div class="hero-title">🧠 Mental Health Risk Prediction System</div>
    <div class="hero-subtitle">
        A hybrid screening prototype using textual and behavioral inputs.
        Enter user information below to estimate mental health risk, generate a fused score,
        and detect unusual behavioral patterns.
    </div>
</div>
""", unsafe_allow_html=True)



def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def encode_user_input(user_df, label_encoders):
    user_df = user_df.copy()
    for col, encoder in label_encoders.items():
        if col in user_df.columns:
            user_df[col] = encoder.transform(user_df[col].astype(str))
    return user_df

import tensorflow as tf
from tensorflow.keras.models import load_model

@st.cache_resource
def load_artifacts():
    with open("text_model.pkl", "rb") as f:
        text_model = pickle.load(f)

    with open("tabular_model.pkl", "rb") as f:
        tabular_model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    with open("threshold.pkl", "rb") as f:
        threshold = pickle.load(f)
    
    with open("tabular_decision_threshold.pkl", "rb") as f:
        tabular_decision_threshold = pickle.load(f)

    autoencoder = load_model("autoencoder_model.h5", compile=False)

    return text_model, tabular_model, tfidf, scaler, label_encoders, feature_columns, threshold, tabular_decision_threshold, autoencoder

text_model, tabular_model, tfidf, scaler, label_encoders, feature_columns, threshold, tabular_decision_threshold, autoencoder = load_artifacts()


def build_display_mapping(classes):
    mapping = {}
    for c in classes:
        cleaned = str(c).strip().lower()
        display = cleaned.title()
        if display not in mapping:
            mapping[display] = c
    return dict(sorted(mapping.items()))

def normalize_course_label(text):
    t = str(text).strip().lower()

    if t in ["engin", "engine", "engineering", "enginering"]:
        return "Engineering"
    if t in ["law", "laws"]:
        return "Law"
    if t in ["bcs", "computer science", "cse", "cs"]:
        return "BCS/CSE"

    return t.title()




form_container = st.container()

with form_container:
    st.subheader("User Input Form")
    st.caption("Please select values carefully for accurate prediction.")

    col1, col2 = st.columns([1.15, 1])

    with col1:
        user_text = st.text_area(
            "How are you feeling today?",
            height=220,
            placeholder="Example: I feel tired, stressed, lonely, and unable to focus on my studies."
        )

    with col2:
        age = st.number_input("Age", min_value=15, max_value=60, value=21)

    c1, c2 = st.columns(2)

    with c1:
        gender_map = build_display_mapping(label_encoders['Choose your gender'].classes_)
        gender_display = st.selectbox("Gender", list(gender_map.keys()))
        gender = gender_map[gender_display]

        year_map = build_display_mapping(label_encoders['Your current year of Study'].classes_)
        year_display_options = list(year_map.keys())
        year_display = st.selectbox("Year of Study", year_display_options)
        year_of_study = year_map[year_display]

        marital_map = build_display_mapping(label_encoders['Marital status'].classes_)
        marital_display = st.selectbox("Marital Status", list(marital_map.keys()))
        marital_status = marital_map[marital_display]

        panic_map = build_display_mapping(label_encoders['Do you have Panic attack?'].classes_)
        panic_display = st.selectbox("Panic Attack", list(panic_map.keys()))
        panic_attack = panic_map[panic_display]

    with c2:
        raw_course_classes = list(label_encoders['What is your course?'].classes_)
        course_display_to_raw = {}
        for raw in raw_course_classes:
            display = normalize_course_label(raw)
            if display not in course_display_to_raw:
                course_display_to_raw[display] = raw

        course_display = st.selectbox("Course", sorted(course_display_to_raw.keys()))
        course = course_display_to_raw[course_display]

        cgpa_map = build_display_mapping(label_encoders['What is your CGPA?'].classes_)
        cgpa_display = st.selectbox("CGPA Range", list(cgpa_map.keys()))
        cgpa = cgpa_map[cgpa_display]

        anxiety_map = build_display_mapping(label_encoders['Do you have Anxiety?'].classes_)
        anxiety_display = st.selectbox("Anxiety", list(anxiety_map.keys()))
        anxiety = anxiety_map[anxiety_display]

        treatment_map = build_display_mapping(label_encoders['Did you seek any specialist for a treatment?'].classes_)
        treatment_display = st.selectbox("Specialist Treatment", list(treatment_map.keys()))
        treatment = treatment_map[treatment_display]


def explain_text_prediction(user_text, tfidf, text_model, top_n=8):
    cleaned = clean_text(user_text)
    text_vector = tfidf.transform([cleaned])

    feature_names = np.array(tfidf.get_feature_names_out())
    vector_dense = text_vector.toarray()[0]
    coefs = text_model.coef_[0]

    present_idx = np.where(vector_dense > 0)[0]

    if len(present_idx) == 0:
        return []

    contributions = []
    for idx in present_idx:
        contribution = vector_dense[idx] * coefs[idx]
        contributions.append((feature_names[idx], contribution))

    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_n]

def explain_tabular_prediction(user_tab_df, scaler, tabular_model, feature_columns, top_n=6):
    user_tab_scaled = scaler.transform(user_tab_df[feature_columns])[0]
    coefs = tabular_model.coef_[0]

    contributions = []
    for feature, value, coef in zip(feature_columns, user_tab_scaled, coefs):
        contribution = value * coef
        contributions.append((feature, contribution))

    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_n]

def predict_user_mental_health_app(
    user_text,
    age,
    gender,
    course,
    year_of_study,
    cgpa,
    marital_status,
    anxiety,
    panic_attack,
    treatment
):
    cleaned = clean_text(user_text)
    text_vector = tfidf.transform([cleaned])
    text_prob = text_model.predict_proba(text_vector)[0][1]

    user_tab_df = pd.DataFrame([{
        'Choose your gender': gender,
        'Age': age,
        'What is your course?': course,
        'Your current year of Study': year_of_study,
        'What is your CGPA?': cgpa,
        'Marital status': marital_status,
        'Do you have Anxiety?': anxiety,
        'Do you have Panic attack?': panic_attack,
        'Did you seek any specialist for a treatment?': treatment
    }])

    user_tab_df = encode_user_input(user_tab_df, label_encoders)
    user_tab_df = user_tab_df[feature_columns]

    user_tab_scaled = scaler.transform(user_tab_df)

    # probability from logistic regression
    tab_prob = tabular_model.predict_proba(user_tab_scaled)[0][1]

    # optional tuned threshold-based class for tabular branch
    tab_pred = int(tab_prob >= tabular_decision_threshold)

    # weighted fusion
    TEXT_WEIGHT = 0.70
    TAB_WEIGHT = 0.30
    base_prob = (TEXT_WEIGHT * text_prob) + (TAB_WEIGHT * tab_prob)

    # anomaly detection
    reconstructed = autoencoder.predict(user_tab_scaled, verbose=0)
    user_mse = np.mean((user_tab_scaled - reconstructed) ** 2, axis=1)[0]
    anomaly = user_mse > threshold

    # anomaly-aware adjustment
    anomaly_boost = 0.05 if anomaly else 0.0
    final_prob = min(base_prob + anomaly_boost, 1.0)

    risk_score = final_prob * 100

    if risk_score < 30:
        risk_level = "Low"
        message = "Low mental health risk."
    elif risk_score < 70:
        risk_level = "Medium"
        message = "Moderate mental health risk."
    else:
        risk_level = "High"
        message = "High mental health risk. Consider professional support."


    if anomaly:
        message += " Unusual behavior pattern detected."

    return {
    "text_probability": round(float(text_prob), 4),
    "tabular_probability": round(float(tab_prob), 4),
    "risk_score": round(float(risk_score), 2),
    "risk_level": risk_level,
    "anomaly": bool(anomaly),
    "message": message,
    "user_tab_df_encoded": user_tab_df
}



if st.button("Predict Mental Health Risk"):
    if not user_text.strip():
        st.warning("Please enter a short text description before prediction.")
    else:
        try:
            with st.spinner("Analyzing text and behavioral signals..."):
                time.sleep(1.2)
                result = predict_user_mental_health_app(
                    user_text=user_text,
                    age=age,
                    gender=gender,
                    course=course,
                    year_of_study=year_of_study,
                    cgpa=cgpa,
                    marital_status=marital_status,
                    anxiety=anxiety,
                    panic_attack=panic_attack,
                    treatment=treatment
                )
            
            text_explanations = explain_text_prediction(
                user_text=user_text,
                tfidf=tfidf,
                text_model=text_model,
                top_n=8
            )

            tabular_explanations = explain_tabular_prediction(
                user_tab_df=result["user_tab_df_encoded"],
                scaler=scaler,
                tabular_model=tabular_model,
                feature_columns=feature_columns,
                top_n=6
            )

            st.success("Prediction completed successfully.")

            risk_level = result["risk_level"]

            if risk_level == "Low":
                result_class = "result-card-low"
                emoji = "🟢"
            elif risk_level == "Medium":
                result_class = "result-card-medium"
                emoji = "🟠"
            else:
                result_class = "result-card-high"
                emoji = "🔴"

            st.markdown(
                f"""
                <div class="{result_class}">
                    <div class="result-title">{emoji} Risk Level: {risk_level}</div>
                    <div class="result-text">
                        <b>Risk Score:</b> {result['risk_score']} / 100<br>
                        <b>Anomaly Detected:</b> {result['anomaly']}<br><br>
                        {result['message']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### Risk Breakdown")

            m1, m2, m3 = st.columns(3)

            with m1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Text Probability</div>
                        <div class="metric-value">{result['text_probability']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with m2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Tabular Probability</div>
                        <div class="metric-value">{result['tabular_probability']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with m3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Final Risk Score</div>
                        <div class="metric-value">{result['risk_score']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("### Risk Meter")
            st.progress(min(int(result["risk_score"]), 100))


            st.markdown("### Explainable AI Insights")

            x1, x2 = st.columns(2)

            with x1:
                st.markdown("#### Text Signals")
                if text_explanations:
                    for word, score in text_explanations:
                        direction = "⬆ increases risk" if score > 0 else "⬇ lowers risk"
                        st.write(f"**{word}** — {direction} ({score:.4f})")
                else:
                    st.write("No strong text indicators found.")

            with x2:
                st.markdown("#### Behavioral Signals")
                for feature, score in tabular_explanations:
                    direction = "⬆ increases risk" if score > 0 else "⬇ lowers risk"
                    pretty_feature = feature.replace("Do you have ", "").replace("?", "").replace("Did you seek any specialist for a treatment", "Specialist treatment")
                    st.write(f"**{pretty_feature}** — {direction} ({score:.4f})")

        except Exception as e:
            st.error(f"Error: {e}")

            st.markdown("---")
st.caption("Built as a hybrid mental health screening prototype using text classification, tabular prediction, and anomaly detection.")

with st.sidebar:
    st.markdown("## Project Summary")
    st.write("This prototype combines:")
    st.write("- Text-based mental health signal analysis")
    st.write("- Behavioral/tabular risk estimation")
    st.write("- Autoencoder-based anomaly detection")