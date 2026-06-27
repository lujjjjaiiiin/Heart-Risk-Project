# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Heart Risk AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease_risk_dataset_earlymed.csv")
    df = df.drop_duplicates()
    return df

df = load_data()

# =========================
# TRAIN MODELS (FIXED)
# =========================
@st.cache_resource
def train_models(df):

    X = df.drop("Heart_Risk", axis=1)
    y = df["Heart_Risk"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.93)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    return scaler, pca, rf, lr, X, X_test, y_test, y_pred_rf, y_pred_lr


scaler, pca, rf, lr, X, X_test, y_test, y_pred_rf, y_pred_lr = train_models(df)

# =========================
# SIDEBAR
# =========================
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 EDA", "🤖 Models", "🧠 Predict"]
)

# =========================
# OVERVIEW
# =========================
if menu == "🏠 Overview":

    st.title("🫀 Heart Disease Risk AI Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Features", df.shape[1]-1)
    col3.metric("High Risk %", f"{df['Heart_Risk'].mean():.2%}")

    st.dataframe(df.head())

# =========================
# EDA
# =========================
elif menu == "📊 EDA":

    st.title("Exploratory Data Analysis")

    fig, ax = plt.subplots()
    sns.countplot(x=df["Heart_Risk"], ax=ax)
    st.pyplot(fig)

    st.write("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), ax=ax, cmap="Reds")
    st.pyplot(fig)

# =========================
# MODELS
# =========================
elif menu == "🤖 Models":

    st.title("Model Evaluation")

    acc_rf = accuracy_score(y_test, y_pred_rf)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    col1, col2 = st.columns(2)
    col1.metric("Random Forest", f"{acc_rf:.2%}")
    col2.metric("Logistic Regression", f"{acc_lr:.2%}")

    st.subheader("Confusion Matrix - RF")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

    st.subheader("Confusion Matrix - LR")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

    # FIXED PCA curve (no refit!)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots()
    ax.plot(cumvar)
    ax.axhline(0.93, linestyle="--", color="red")
    ax.set_title("PCA Explained Variance")
    st.pyplot(fig)

# =========================
# PREDICT
# =========================
elif menu == "🧠 Predict":

    st.title("🫀 AI Heart Risk Assessment System")

    st.markdown("""
    <div style='color:#aaa; font-size:15px; margin-bottom:10px;'>
    Advanced clinical decision support system powered by Machine Learning.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # STEP PROGRESS STYLE
    # =========================
    step = st.radio(
        "Assessment Stage",
        ["🧬 Symptoms Check", "🏥 Medical Profile", "🚶 Lifestyle Analysis"],
        horizontal=True
    )

    inputs = {}

    # =========================
    # STAGE 1
    # =========================
    if step == "🧬 Symptoms Check":

        st.subheader("Symptom Evaluation")

        cols = st.columns(2)

        symptoms = [
            "Chest_Pain", "Shortness_of_Breath", "Fatigue",
            "Palpitations", "Dizziness", "Swelling",
            "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea"
        ]

        for i, c in enumerate(symptoms):
            with cols[i % 2]:
                inputs[c] = st.slider(
                    c,
                    0.0, 1.0,
                    0.0,
                    help="0 = No symptom, 1 = Severe symptom"
                )

        st.info("Move to next stage to continue evaluation.")

    # =========================
    # STAGE 2
    # =========================
    elif step == "🏥 Medical Profile":

        st.subheader("Medical History")

        cols = st.columns(2)

        medical = [
            "High_BP", "High_Cholesterol", "Diabetes",
            "Smoking", "Obesity", "Family_History"
        ]

        for i, c in enumerate(medical):
            with cols[i % 2]:
                inputs[c] = st.slider(
                    c,
                    0.0, 1.0, 0.0,
                    help="Medical condition presence"
                )

        st.warning("Ensure all medical conditions are correctly filled.")

    # =========================
    # STAGE 3
    # =========================
    elif step == "🚶 Lifestyle Analysis":

        st.subheader("Lifestyle & Demographics")

        col1, col2 = st.columns(2)

        inputs["Sedentary_Lifestyle"] = col1.slider("Sedentary Lifestyle", 0.0, 1.0, 0.0)
        inputs["Chronic_Stress"] = col2.slider("Chronic Stress", 0.0, 1.0, 0.0)

        inputs["Gender"] = col1.selectbox("Gender", [0, 1])
        inputs["Age"] = col2.number_input("Age", 1, 100, 30)

    # =========================
    # PREDICT BUTTON (GLOBAL)
    # =========================

    st.markdown("---")

    col_btn, col_note = st.columns([1, 2])

    with col_btn:
        run = st.button("🔍 Run AI Analysis")

    with col_note:
        st.caption("AI will analyze all clinical + lifestyle + symptom data")

    # =========================
    # RESULT
    # =========================
    if run:

        input_df = pd.DataFrame([inputs])[X.columns]

        scaled = scaler.transform(input_df)
        pca_input = pca.transform(scaled)

        pred = rf.predict(pca_input)[0]
        prob = rf.predict_proba(pca_input)[0][1]

        st.markdown("---")
        st.subheader("AI Diagnosis Result")

        if pred == 1:

            st.error("⚠ HIGH RISK DETECTED")

            st.markdown(f"""
            **Risk Probability:** `{prob:.2%}`  
            **Recommendation:** Immediate cardiology consultation is advised.
            """)

            st.markdown("🟥 System Confidence Indicator")
            st.progress(float(prob))

        else:

            st.success("✅ LOW RISK DETECTED")

            st.markdown(f"""
            **Risk Probability:** `{prob:.2%}`  
            **Recommendation:** Maintain healthy lifestyle and regular checkups.
            """)

            st.markdown("🟩 System Confidence Indicator")
            st.progress(float(prob))
