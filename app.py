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

    st.title("Heart Risk Prediction")

    model_choice = st.radio(
        "Choose Model",
        ["Random Forest", "Logistic Regression"]
    )

    model = rf if model_choice == "Random Forest" else lr

    inputs = []

    st.write("Enter values:")

    for col in X.columns:
        val = st.number_input(col, value=float(X[col].mean()))
        inputs.append(val)

    if st.button("Predict ❤️"):

        # FIXED: keep column structure
        input_df = pd.DataFrame([inputs], columns=X.columns)

        input_scaled = scaler.transform(input_df)
        input_pca = pca.transform(input_scaled)

        prediction = model.predict(input_pca)[0]
        prob = model.predict_proba(input_pca)[0][1]

        st.subheader("Result")

        if prediction == 1:
            st.error(f"⚠ High Risk ({prob:.2%})")
        else:
            st.success(f"✅ Low Risk ({prob:.2%})")
