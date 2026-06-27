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
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:15px 0;'>
        <div style='font-size:3rem;'>🫀</div>
        <div style='font-size:1.2rem; font-weight:700; color:#fff;'>
            Heart AI System
        </div>
        <div style='font-size:0.75rem; color:#888; margin-top:5px;'>
            Medical ML Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    menu = st.radio(
        "📍 Navigation Menu",
        ["🏠 Overview", "📊 EDA", "🤖 Models", "🧠 Predict"],
        label_visibility="collapsed"
    )

    st.markdown("---")

# =========================
# OVERVIEW
# =========================
if menu == "🏠 Overview":

    st.title("🫀 Heart Disease AI Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Records", len(df))
    col2.metric("Features", df.shape[1]-1)
    col3.metric("High Risk", int(df["Heart_Risk"].sum()))
    col4.metric("Risk Rate", f"{df['Heart_Risk'].mean():.1%}")

    st.dataframe(df.head())

# =========================
# EDA
# =========================
elif menu == "📊 EDA":

    st.title("EDA")

    feature = st.selectbox("Feature", df.columns[:-1])

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

    st.title("Models")

    acc_rf = accuracy_score(y_test, y_pred_rf)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    st.write(acc_rf, acc_lr)

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

# =========================
# PREDICT
# =========================
elif menu == "🧠 Predict":

    st.title("Predict")

    tab1, tab2, tab3 = st.tabs(["Symptoms", "Medical", "Lifestyle"])

    inputs = {}

    # TAB 1
    with tab1:
        cols = st.columns(2)
        symptom_cols = [
            "Chest_Pain","Shortness_of_Breath","Fatigue",
            "Palpitations","Dizziness","Swelling",
            "Pain_Arms_Jaw_Back","Cold_Sweats_Nausea"
        ]

        for i, col in enumerate(symptom_cols):
            with cols[i % 2]:
                inputs[col] = st.slider(col, 0.0, 1.0, 0.0)

    # TAB 2
    with tab2:
        cols = st.columns(2)
        medical_cols = [
            "High_BP","High_Cholesterol","Diabetes",
            "Smoking","Obesity","Family_History"
        ]

        for i, col in enumerate(medical_cols):
            with cols[i % 2]:
                inputs[col] = st.slider(col, 0.0, 1.0, 0.0)

    # TAB 3 (FIXED ONLY)
    with tab3:

        col1, col2 = st.columns(2)

        lifestyle_cols = ["Sedentary_Lifestyle", "Chronic_Stress"]

        for i, col in enumerate(lifestyle_cols):
            with col1 if i % 2 == 0 else col2:
                inputs[col] = st.slider(col, 0.0, 1.0, 0.0)

        inputs["Age"] = st.number_input("Age", 0, 100, 30)

        gender = st.radio("Gender", ["Male", "Female"])
        inputs["Gender"] = 0 if gender == "Male" else 1

    st.markdown("---")

    if st.button("Predict Risk"):

        input_df = pd.DataFrame([inputs])[X.columns]

        input_scaled = scaler.transform(input_df)
        input_pca = pca.transform(input_scaled)

        prediction = rf.predict(input_pca)[0]
        prob = rf.predict_proba(input_pca)[0][1]

        if prediction == 1:
            st.error(f"High Risk ({prob:.2%})")
        else:
            st.success(f"Low Risk ({prob:.2%})")
