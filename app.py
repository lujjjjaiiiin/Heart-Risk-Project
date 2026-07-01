import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

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
    df = df.dropna()
    return df

df = load_data()

st.write("APP IS RUNNING")  # debug بسيط

# =========================
# TRAIN MODELS
# =========================
@st.cache_resource
def train_models(df):
    X = df.drop("Heart_Risk", axis=1)
    y = df["Heart_Risk"]

    X = X.astype(float)

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

    xgb = XGBClassifier(random_state=42, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    return scaler, pca, rf, lr, xgb, X, X_test, y_test, y_pred_rf, y_pred_lr, y_pred_xgb


st.write("Before training")

scaler, pca, rf, lr, xgb, X, X_test, y_test, y_pred_rf, y_pred_lr, y_pred_xgb = train_models(df)

st.write("After training")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    menu = st.radio(
        "nav",
        ["Overview", "EDA", "Models", "Predict"],
        label_visibility="collapsed"
    )

# =========================
# OVERVIEW
# =========================
if menu == "Overview":

    st.title("🫀 Heart Disease AI Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Features", df.shape[1]-1)
    col3.metric("High Risk", int(df["Heart_Risk"].sum()))
    col4.metric("Risk Rate", f"{df['Heart_Risk'].mean():.1%}")

# =========================
# EDA
# =========================
elif menu == "EDA":

    st.title("📊 EDA")

    feature = st.selectbox("Feature", [c for c in df.columns if c != "Heart_Risk"])

    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    st.pyplot(fig)

# =========================
# MODELS
# =========================
elif menu == "Models":

    st.title("🤖 Models")

    acc_rf = accuracy_score(y_test, y_pred_rf)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)

    st.write("RF:", acc_rf)
    st.write("LR:", acc_lr)
    st.write("XGB:", acc_xgb)

# =========================
# PREDICT
# =========================
elif menu == "Predict":

    st.title("🧠 Predict")

    inputs = {}

    for col in X.columns:
        inputs[col] = st.number_input(col, value=0.0)

    model = st.selectbox("Model", ["Random Forest", "Logistic Regression", "XGBoost"])
    model_choice = rf if model == "Random Forest" else lr if model == "Logistic Regression" else xgb

    if st.button("Predict"):

        input_df = pd.DataFrame([inputs]).reindex(columns=X.columns, fill_value=0)

        input_scaled = scaler.transform(input_df)
        input_pca = pca.transform(input_scaled)

        pred = model_choice.predict(input_pca)[0]
        prob = model_choice.predict_proba(input_pca)[0][1]

        st.subheader("Result")
        st.write("Risk:", "High" if pred == 1 else "Low")
        st.write("Probability:", prob)
