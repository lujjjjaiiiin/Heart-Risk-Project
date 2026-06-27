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

    # =========================
    # NAVIGATION MENU (PREMIUM STYLE)
    # =========================
    menu = st.radio(
        "📍 Navigation Menu",
        ["🏠 Overview", "📊 EDA", "🤖 Models", "🧠 Predict"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # =========================
    # DATA INFO CARD
    # =========================
    st.markdown(f"""
    <div style='background:#1a1a1a; padding:12px; border-radius:10px; border:1px solid #333;'>
        <div style='color:#ff4d4d; font-weight:600;'>📊 Dataset Info</div>
        <div style='color:#aaa; font-size:12px; margin-top:5px;'>
            Records: {len(df):,}<br>
            Features: {df.shape[1]-1}<br>
            Target: Heart Risk
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # MODEL INFO CARD
    # =========================
    st.markdown("""
    <div style='background:#1a1a1a; padding:12px; border-radius:10px; border:1px solid #333;'>
        <div style='color:#4d88ff; font-weight:600;'>🧠 AI Models</div>
        <div style='color:#aaa; font-size:12px; margin-top:5px;'>
            • Random Forest<br>
            • Logistic Regression<br>
            • PCA + Scaling
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # FOOTER
    # =========================
    st.markdown("""
    <div style='text-align:center; margin-top:20px; color:#555; font-size:11px;'>
        Built for Tuwaiq Academy<br>
        ML Project © 2026
    </div>
    """, unsafe_allow_html=True)

# =========================
# OVERVIEW
# =========================
if menu == "🏠 Overview":

    st.title("🫀 Heart Disease AI Dashboard")

    st.markdown("""
    <div style='color:#aaa; font-size:15px; margin-bottom:20px;'>
    A machine learning system that analyzes patient health data and predicts heart disease risk with high accuracy.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # HERO CARDS (PRO KPI DESIGN)
    # =========================
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"""
    <div style='background:#1a1a1a; padding:15px; border-radius:12px; border:1px solid #333;'>
    <h3 style='color:#ff4d4d;'>📊 Records</h3>
    <h2>{len(df):,}</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div style='background:#1a1a1a; padding:15px; border-radius:12px; border:1px solid #333;'>
    <h3 style='color:#ff4d4d;'>🧬 Features</h3>
    <h2>{df.shape[1]-1}</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div style='background:#1a1a1a; padding:15px; border-radius:12px; border:1px solid #333;'>
    <h3 style='color:#ff4d4d;'>⚠ High Risk</h3>
    <h2>{int(df['Heart_Risk'].sum()):,}</h2>
    </div>
    """, unsafe_allow_html=True)

    col4.markdown(f"""
    <div style='background:#1a1a1a; padding:15px; border-radius:12px; border:1px solid #333;'>
    <h3 style='color:#ff4d4d;'>📈 Risk Rate</h3>
    <h2>{df['Heart_Risk'].mean():.1%}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # VISUAL SECTION (PRO LAYOUT)
    # =========================
    colA, colB = st.columns([1.3, 1])

    # -------------------------
    # LEFT: PIE CHART
    # -------------------------
    with colA:
        st.subheader("Risk Distribution Overview")

        fig, ax = plt.subplots(figsize=(5,4))

        counts = df["Heart_Risk"].value_counts()

        ax.pie(
            counts,
            labels=["Low Risk", "High Risk"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#4d88ff", "#ff4d4d"],
            wedgeprops={"edgecolor":"#111"}
        )

        fig.patch.set_facecolor("#0f0f0f")
        ax.set_facecolor("#0f0f0f")

        st.pyplot(fig)
        plt.close()

    # -------------------------
    # RIGHT: QUICK INSIGHT BOX
    # -------------------------
    with colB:
        st.subheader("Dataset Insights")

        st.markdown("""
        <div style='background:#1a1a1a; padding:20px; border-radius:12px; border:1px solid #333;'>

        <h4 style='color:#ff4d4d;'>📌 Key Findings</h4>

        <ul style='color:#ccc; line-height:1.8;'>
        <li>Dataset contains balanced classification target</li>
        <li>Binary medical indicators dominate features</li>
        <li>Age & lifestyle are strong risk factors</li>
        <li>Suitable for classification models (RF, LR)</li>
        </ul>

        <h4 style='color:#ff4d4d;'>🧠 ML Approach</h4>

        <ul style='color:#ccc; line-height:1.8;'>
        <li>StandardScaler for normalization</li>
        <li>PCA for dimensionality reduction</li>
        <li>Random Forest + Logistic Regression comparison</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # MINI HEATMAP (PRO TOUCH)
    # =========================
    st.subheader("Feature Correlation Snapshot")

    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax, linewidths=0.3)
    fig.patch.set_facecolor("#0f0f0f")

    st.pyplot(fig)
    plt.close()
# =========================
# EDA
# =========================
elif menu == "📊 EDA":

    st.title("📊 Exploratory Data Analysis")

    st.markdown("""
    <div style='color:#aaa; font-size:15px; margin-bottom:10px;'>
    Interactive analysis of patient health patterns and risk factors.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # SIDEBAR FILTER STYLE INSIDE PAGE
    # =========================
    col1, col2 = st.columns([1, 2])

    # =========================
    # LEFT CONTROL PANEL
    # =========================
    with col1:
        st.subheader("🎛 Controls")

        feature = st.selectbox(
            "Select Feature",
            [c for c in df.columns if c != "Heart_Risk"]
        )

        chart_type = st.radio(
            "Chart Type",
            ["Distribution", "Boxplot", "Risk Comparison"]
        )

        st.markdown("---")

        st.info("Use controls to explore patterns interactively")

    # =========================
    # RIGHT VISUAL AREA
    # =========================
    with col2:

        # -------------------------
        # 1. Distribution Plot
        # -------------------------
        if chart_type == "Distribution":

            st.subheader(f"📈 Distribution of {feature}")

            fig, ax = plt.subplots(figsize=(7,4))

            sns.histplot(
                df[feature],
                bins=30,
                kde=True,
                color="#ff4d4d"
            )

            ax.set_facecolor("#0f0f0f")
            fig.patch.set_facecolor("#0f0f0f")

            st.pyplot(fig)
            plt.close()

        # -------------------------
        # 2. Boxplot
        # -------------------------
        elif chart_type == "Boxplot":

            st.subheader(f"📦 Boxplot of {feature} by Risk")

            fig, ax = plt.subplots(figsize=(7,4))

            sns.boxplot(
                x="Heart_Risk",
                y=feature,
                data=df,
                palette=["#4d88ff", "#ff4d4d"],
                ax=ax
            )

            ax.set_xticklabels(["Low Risk", "High Risk"])

            ax.set_facecolor("#0f0f0f")
            fig.patch.set_facecolor("#0f0f0f")

            st.pyplot(fig)
            plt.close()

        # -------------------------
        # 3. Risk Comparison
        # -------------------------
        else:

            st.subheader(f"⚠ Risk Comparison: {feature}")

            fig, ax = plt.subplots(figsize=(7,4))

            sns.barplot(
                x="Heart_Risk",
                y=feature,
                data=df,
                palette=["#4d88ff", "#ff4d4d"],
                ax=ax
            )

            ax.set_xticklabels(["Low Risk", "High Risk"])

            ax.set_facecolor("#0f0f0f")
            fig.patch.set_facecolor("#0f0f0f")

            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    # =========================
    # CORRELATION (PREMIUM STYLE)
    # =========================
    st.subheader("🔥 Feature Correlation Matrix")

    fig, ax = plt.subplots(figsize=(10,6))

    sns.heatmap(
        df.corr(),
        cmap="coolwarm",
        linewidths=0.3,
        ax=ax
    )

    ax.set_title("Correlation Between Features", color="white")

    fig.patch.set_facecolor("#0f0f0f")

    st.pyplot(fig)
    plt.close()
# =========================
# MODELS
# =========================
elif menu == "🤖 Models":

    st.title("🤖 Model Performance Dashboard")

    st.markdown("""
    <div style='color:#aaa; font-size:15px; margin-bottom:10px;'>
    Comparison between machine learning models trained on heart disease dataset.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # ACCURACY CARDS
    # =========================
    acc_rf = accuracy_score(y_test, y_pred_rf)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    col1, col2 = st.columns(2)

    col1.markdown(f"""
    <div style='background:#1a1a1a; padding:20px; border-radius:12px; border:1px solid #333; text-align:center;'>
        <h3 style='color:#ff4d4d;'>🌲 Random Forest</h3>
        <h2>{acc_rf:.2%}</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div style='background:#1a1a1a; padding:20px; border-radius:12px; border:1px solid #333; text-align:center;'>
        <h3 style='color:#4d88ff;'>📈 Logistic Regression</h3>
        <h2>{acc_lr:.2%}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # MODEL COMPARISON BAR CHART
    # =========================
    st.subheader("📊 Model Comparison")

    fig, ax = plt.subplots(figsize=(6,4))

    models = ["Random Forest", "Logistic Regression"]
    scores = [acc_rf, acc_lr]
    colors = ["#ff4d4d", "#4d88ff"]

    ax.bar(models, scores, color=colors)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")

    for i, v in enumerate(scores):
        ax.text(i, v + 0.01, f"{v:.2%}", ha="center", color="white")

    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # =========================
    # CONFUSION MATRIX SIDE BY SIDE
    # =========================
    st.subheader("📉 Confusion Matrices")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Random Forest**")

        fig, ax = plt.subplots(figsize=(4,4))

        sns.heatmap(
            confusion_matrix(y_test, y_pred_rf),
            annot=True,
            fmt="d",
            cmap="Reds",
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        fig.patch.set_facecolor("#0f0f0f")

        st.pyplot(fig)
        plt.close()

    with colB:
        st.markdown("**Logistic Regression**")

        fig, ax = plt.subplots(figsize=(4,4))

        sns.heatmap(
            confusion_matrix(y_test, y_pred_lr),
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        fig.patch.set_facecolor("#0f0f0f")

        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # =========================
    # CLASSIFICATION REPORT SUMMARY
    # =========================
    st.subheader("🧠 Model Insights")

    better_model = "Random Forest" if acc_rf > acc_lr else "Logistic Regression"

    st.success(f"🏆 Best Performing Model: {better_model}")

    st.markdown("""
    ### 📌 Key Observations:
    - Random Forest handles non-linear relationships better
    - Logistic Regression is more interpretable
    - Both models perform strongly due to clean dataset structure
    """)

    st.info("""
    💡 In medical datasets, recall is often more important than accuracy
    because missing a high-risk patient is critical.
    """)

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

    st.title("🫀 Heart Disease Risk Assessment")

    st.markdown("""
    <p style='color:#aaa;'>
    Please fill in the patient information below. The system will analyze the risk using AI.
    </p>
    """, unsafe_allow_html=True)

    # =========================
    # TABS (PRO STRUCTURE)
    # =========================
    tab1, tab2, tab3 = st.tabs(["🧬 Symptoms", "🏥 Medical History", "🧑 Lifestyle"])

    inputs = {}

    # =========================
    # TAB 1: SYMPTOMS
    # =========================
    with tab1:
        col1, col2 = st.columns(2)

        symptom_cols = [
            "Chest_Pain", "Shortness_of_Breath", "Fatigue",
            "Palpitations", "Dizziness", "Swelling",
            "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea"
        ]

        for i, col in enumerate(symptom_cols):
            with col1 if i % 2 == 0 else col2:
                inputs[col] = st.slider(col, 0.0, 1.0, 0.0)

    # =========================
    # TAB 2: MEDICAL HISTORY
    # =========================
    with tab2:
        col1, col2 = st.columns(2)

        medical_cols = [
            "High_BP", "High_Cholesterol", "Diabetes",
            "Smoking", "Obesity", "Family_History"
        ]

        for i, col in enumerate(medical_cols):
            with col1 if i % 2 == 0 else col2:
                inputs[col] = st.slider(col, 0.0, 1.0, 0.0)

    # =========================
    # TAB 3: LIFESTYLE
    # =========================
   # =========================
# TAB 3: LIFESTYLE
# =========================
with tab3:

    st.subheader("🚶 Lifestyle & Demographics")

    col1, col2 = st.columns(2)

    # =====================
    # Lifestyle features
    # =====================
    lifestyle_cols = [
        "Sedentary_Lifestyle",
        "Chronic_Stress"
    ]

    for i, col in enumerate(lifestyle_cols):
        with col1 if i % 2 == 0 else col2:
            inputs[col] = st.slider(col, 0.0, 1.0, 0.0)

    st.markdown("---")

    # =====================
    # AGE (separate UI)
    # =====================
    inputs["Age"] = st.number_input("Age", 0, 100, 30)

    # =====================
    # GENDER (proper UI)
    # =====================
    gender = st.radio("Gender", ["Male", "Female"])
    inputs["Gender"] = 0 if gender == "Male" else 1

    # =========================
    # PREDICT BUTTON
    # =========================
    st.markdown("---")

    if st.button("🔍 Predict Risk"):

        # convert to correct order
        input_df = pd.DataFrame([inputs])[X.columns]

        input_scaled = scaler.transform(input_df)
        input_pca = pca.transform(input_scaled)

        model_choice = rf  # default
        prediction = model_choice.predict(input_pca)[0]
        prob = model_choice.predict_proba(input_pca)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"⚠ High Risk of Heart Disease ({prob:.2%})")
            st.markdown("👉 Recommendation: Immediate medical consultation.")
        else:
            st.success(f"✅ Low Risk ({prob:.2%})")
            st.markdown("👉 Recommendation: Maintain healthy lifestyle.")
