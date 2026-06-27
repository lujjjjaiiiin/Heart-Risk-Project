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
# RED NAVIGATION THEME
# =========================
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0000, #0f0f0f);
    border-right: 1px solid #3a0a0a;
}

/* radio container */
.stRadio > div {
    background: rgba(255, 0, 0, 0.05);
    padding: 10px;
    border-radius: 10px;
}

/* general UI */
html, body {
    background-color: #0f0f0f;
}
</style>
""", unsafe_allow_html=True)

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
# TRAIN MODELS
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

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    return scaler, pca, rf, lr, X, X_test, y_test, y_pred_rf, y_pred_lr

scaler, pca, rf, lr, X, X_test, y_test, y_pred_rf, y_pred_lr = train_models(df)

# =========================
# SIDEBAR NAVIGATION
# =========================
with st.sidebar:

    st.markdown("## 🫀 Heart AI")

    menu = st.radio(
        "Navigation",
        ["🏠 Overview", "📊 EDA", "🤖 Models", "🧠 Predict"],
        label_visibility="collapsed"
    )

# =========================
# OVERVIEW
# =========================
if menu == "🏠 Overview":

    st.title("Heart Disease AI Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Records", len(df))
    col2.metric("Features", df.shape[1]-1)
    col3.metric("High Risk", int(df["Heart_Risk"].sum()))
    col4.metric("Risk Rate", f"{df['Heart_Risk'].mean():.1%}")

    st.markdown("---")

    st.dataframe(df.head())

# =========================
# EDA
# =========================
elif menu == "📊 EDA":

    st.title("Exploratory Data Analysis")

    feature = st.selectbox("Select Feature", df.columns[:-1])

    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    st.pyplot(fig)

# =========================
# MODELS
# =========================
elif menu == "🤖 Models":

    st.title("Model Evaluation")

    acc_rf = accuracy_score(y_test, y_pred_rf)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    st.write("Random Forest:", acc_rf)
    st.write("Logistic Regression:", acc_lr)

    st.subheader("Confusion Matrix RF")
    st.pyplot(sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d").figure)

    st.subheader("Confusion Matrix LR")
    st.pyplot(sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d").figure)

# =========================
# PREDICT
# =========================
elif menu == "🧠 Predict":

    st.title("Risk Prediction")

    tab1, tab2, tab3 = st.tabs(["Symptoms", "Medical", "Lifestyle"])

    inputs = {}

    # ================= TAB 1 =================
    with tab1:
        symptoms = [
            "Chest_Pain","Shortness_of_Breath","Fatigue",
            "Palpitations","Dizziness","Swelling",
            "Pain_Arms_Jaw_Back","Cold_Sweats_Nausea"
        ]

        for col in symptoms:
            inputs[col] = st.slider(col, 0.0, 1.0, 0.0)

    # ================= TAB 2 =================
    with tab2:
        medical = [
            "High_BP","High_Cholesterol","Diabetes",
            "Smoking","Obesity","Family_History"
        ]

        for col in medical:
            inputs[col] = st.slider(col, 0.0, 1.0, 0.0)

    # ================= TAB 3 =================
    with tab3:

        lifestyle = ["Sedentary_Lifestyle","Chronic_Stress"]

        for col in lifestyle:
            inputs[col] = st.slider(col, 0.0, 1.0, 0.0)

        inputs["Age"] = st.number_input("Age", 0, 100, 30)

        gender = st.radio("Gender", ["Male", "Female"])
        inputs["Gender"] = 0 if gender == "Male" else 1

    st.markdown("---")

    if st.button("Predict Risk"):

        input_df = pd.DataFrame([inputs])[X.columns]

        input_scaled = scaler.transform(input_df)
        input_pca = pca.transform(input_scaled)

        model = rf
        pred = model.predict(input_pca)[0]
        prob = model.predict_proba(input_pca)[0][1]

        if pred == 1:
            st.error(f"High Risk ({prob:.2%})")
        else:
            st.success(f"Low Risk ({prob:.2%})")
