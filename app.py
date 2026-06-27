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
# CUSTOM STYLE
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background-color: #0f0f0f;
    color: #f0f0f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1a1a1a;
    border-right: 1px solid #2a2a2a;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1c1c1c, #2a1a1a);
    border: 1px solid #3d1f1f;
    border-radius: 12px;
    padding: 16px 20px;
}

[data-testid="stMetricLabel"] {
    color: #999 !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

[data-testid="stMetricValue"] {
    color: #ff4d4d !important;
    font-size: 2rem;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #c1121f, #8b0000);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6em 2em;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #e01b2a, #a01010);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(193, 18, 31, 0.4);
}

/* Headers */
h1 { color: #ffffff !important; font-weight: 700; letter-spacing: -0.02em; }
h2 { color: #f0f0f0 !important; font-weight: 600; }
h3 { color: #e0e0e0 !important; font-weight: 600; }

/* Divider */
hr { border-color: #2a2a2a; margin: 1.5rem 0; }

/* DataFrames */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Number inputs */
.stNumberInput input {
    background-color: #1c1c1c;
    border: 1px solid #333;
    color: #f0f0f0;
    border-radius: 6px;
}

/* Radio buttons */
.stRadio label { color: #ccc !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #999;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: #ff4d4d !important;
    border-bottom-color: #ff4d4d !important;
}

/* Accent pulse for hero */
.hero-badge {
    display: inline-block;
    background: rgba(193, 18, 31, 0.15);
    border: 1px solid rgba(193, 18, 31, 0.4);
    color: #ff6b6b;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.risk-box-high {
    background: linear-gradient(135deg, #2a0a0a, #1a0505);
    border: 1px solid #8b0000;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}

.risk-box-low {
    background: linear-gradient(135deg, #0a1a0a, #051205);
    border: 1px solid #1a6b1a;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}

.risk-label-high { font-size: 2rem; font-weight: 700; color: #ff4d4d; }
.risk-label-low  { font-size: 2rem; font-weight: 700; color: #4dff88; }
.risk-prob { font-size: 0.9rem; color: #aaa; margin-top: 4px; }
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

df = load_data()
scaler, pca, rf, lr, X, X_test, y_test, y_pred_rf, y_pred_lr = train_models(df)

# =========================
# MATPLOTLIB DARK THEME
# =========================
plt.rcParams.update({
    "figure.facecolor": "#141414",
    "axes.facecolor": "#141414",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#ccc",
    "xtick.color": "#999",
    "ytick.color": "#999",
    "text.color": "#eee",
    "grid.color": "#2a2a2a",
    "grid.linestyle": "--",
})

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
        <div style='font-size:2.5rem'>🫀</div>
        <div style='font-size:1.1rem; font-weight:700; color:#fff;'>Heart Risk AI</div>
        <div style='font-size:0.75rem; color:#777; margin-top:4px;'>ML Prediction Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    menu = st.radio(
        "Navigate",
        ["🏠  Overview", "📊  EDA", "🤖  Models", "🧠  Predict"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#555; padding: 8px 0;'>
        Dataset: Heart Disease Risk<br>
        Records: {:,} &nbsp;|&nbsp; Features: {}<br>
        Models: Random Forest · Logistic Reg
    </div>
    """.format(len(df), df.shape[1] - 1), unsafe_allow_html=True)

# =========================
# PAGE: OVERVIEW
# =========================
if menu == "🏠  Overview":

    st.markdown('<div class="hero-badge">🫀 Medical AI · Machine Learning</div>', unsafe_allow_html=True)
    st.title("Heart Disease Risk Prediction")
    st.markdown("<p style='color:#888; font-size:1rem; margin-bottom:2rem;'>An AI-powered dashboard to analyze and predict heart disease risk using clinical features.</p>", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{df.shape[0]:,}")
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("High Risk Cases", f"{int(df['Heart_Risk'].sum()):,}")
    col4.metric("Risk Rate", f"{df['Heart_Risk'].mean():.1%}")

    st.markdown("---")

    col_a, col_b = st.columns([1.4, 1])

    with col_a:
        st.subheader("Sample Data")
        st.dataframe(
            df.head(8).style.background_gradient(cmap="Reds", subset=["Heart_Risk"]),
            use_container_width=True,
            height=280
        )

    with col_b:
        st.subheader("Risk Distribution")
        fig, ax = plt.subplots(figsize=(4, 3))
        counts = df["Heart_Risk"].value_counts()
        colors = ["#ff4d4d", "#555"]
        ax.pie(counts, labels=["High Risk", "Low Risk"], colors=colors,
               autopct="%1.1f%%", startangle=90,
               wedgeprops={"edgecolor": "#1a1a1a", "linewidth": 2},
               textprops={"color": "#ccc", "fontsize": 9})
        ax.set_facecolor("#141414")
        fig.patch.set_facecolor("#141414")
        st.pyplot(fig)
        plt.close()

# =========================
# PAGE: EDA
# =========================
elif menu == "📊  EDA":

    st.title("Exploratory Data Analysis")
    st.markdown("<p style='color:#888;'>Visualizing key relationships in the dataset.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution by Risk")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for risk, color in zip([0, 1], ["#4d88ff", "#ff4d4d"]):
            ax.hist(df[df["Heart_Risk"] == risk]["Age"], bins=25,
                    alpha=0.7, color=color, label=f"Risk={risk}", edgecolor="none")
        ax.legend(fontsize=8)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        fig.patch.set_facecolor("#141414")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Chest Pain vs Heart Risk")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(x="Heart_Risk", y="Chest_Pain", data=df,
                    palette=["#4d88ff", "#ff4d4d"], ax=ax)
        ax.set_xlabel("Heart Risk")
        ax.set_ylabel("Chest Pain (avg)")
        ax.set_xticklabels(["Low Risk", "High Risk"])
        ax.grid(True, alpha=0.3)
        fig.patch.set_facecolor("#141414")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    mask = np.zeros_like(df.corr())
    np.fill_diagonal(mask, True)
    sns.heatmap(df.corr(), cmap="RdYlBu_r", ax=ax, linewidths=0.4,
                linecolor="#1a1a1a", annot=True, fmt=".1f",
                annot_kws={"size": 7}, mask=mask)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)
    fig.patch.set_facecolor("#141414")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    st.subheader("Feature Boxplots by Risk")
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "Heart_Risk"]
    selected_feat = st.selectbox("Select Feature", num_cols)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.boxplot(x="Heart_Risk", y=selected_feat, data=df,
                palette=["#4d88ff", "#ff4d4d"], ax=ax,
                flierprops={"markerfacecolor": "#ff4d4d", "markersize": 3})
    ax.set_xticklabels(["Low Risk", "High Risk"])
    ax.grid(True, alpha=0.3)
    fig.patch.set_facecolor("#141414")
    st.pyplot(fig)
    plt.close()

# =========================
# PAGE: MODELS
# =========================
elif menu == "🤖  Models":

    st.title("Model Evaluation")
    st.markdown("<p style='color:#888;'>Comparing Random Forest vs Logistic Regression.</p>", unsafe_allow_html=True)
    st.markdown("---")

    acc_rf = accuracy_score(y_test, y_pred_rf)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    col1, col2 = st.columns(2)
    col1.metric("🌲 Random Forest Accuracy", f"{acc_rf:.2%}")
    col2.metric("📈 Logistic Regression Accuracy", f"{acc_lr:.2%}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    for col, model_name, y_pred in [(col_a, "Random Forest", y_pred_rf), (col_b, "Logistic Regression", y_pred_lr)]:
        with col:
            st.subheader(f"{model_name}")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                        linewidths=1, linecolor="#1a1a1a",
                        xticklabels=["Low", "High"],
                        yticklabels=["Low", "High"], ax=ax,
                        annot_kws={"size": 13, "weight": "bold"})
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("Actual", fontsize=9)
            ax.set_title("Confusion Matrix", fontsize=10, color="#ccc")
            fig.patch.set_facecolor("#141414")
            st.pyplot(fig)
            plt.close()

            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(2)
            st.dataframe(report_df.style.background_gradient(cmap="Reds", subset=["f1-score"]),
                         use_container_width=True)

    st.markdown("---")

    st.subheader("PCA Explained Variance")
    pca_full = PCA()
    X_scaled_full = scaler.transform(X)
    pca_full.fit(X_scaled_full)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, color="#ff4d4d", linewidth=2.5, marker="o", markersize=5)
    ax.axhline(0.93, color="#ffaa00", linestyle="--", linewidth=1.2, label="93% threshold")
    ax.fill_between(range(1, len(cumvar) + 1), cumvar, alpha=0.15, color="#ff4d4d")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Explained Variance Curve", color="#eee")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.patch.set_facecolor("#141414")
    st.pyplot(fig)
    plt.close()

# =========================
# PAGE: PREDICT
# =========================
elif menu == "🧠  Predict":

    st.title("Risk Prediction")
    st.markdown("<p style='color:#888;'>Enter patient clinical values to predict heart disease risk.</p>", unsafe_allow_html=True)
    st.markdown("---")

    model_choice = st.radio(
        "Choose Model",
        ["🌲 Random Forest", "📈 Logistic Regression"],
        horizontal=True
    )
    active_model = rf if "Random" in model_choice else lr

    st.markdown("---")

    inputs = []
    cols_per_row = 3
    feature_cols = list(X.columns)
    rows = [feature_cols[i:i+cols_per_row] for i in range(0, len(feature_cols), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for i, feat in enumerate(row):
            with cols[i]:
                val = st.number_input(
                    feat,
                    value=float(round(X[feat].mean(), 2)),
                    step=0.1,
                    format="%.2f"
                )
                inputs.append(val)

    st.markdown("---")

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        predict_clicked = st.button("Predict Risk 🫀")

    if predict_clicked:
        input_array = np.array(inputs).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        input_pca = pca.transform(input_scaled)

        prediction = active_model.predict(input_pca)[0]
        prob = active_model.predict_proba(input_pca)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")

        col_res, col_gauge = st.columns([1, 1.5])

        with col_res:
            if prediction == 1:
                st.markdown(f"""
                <div class="risk-box-high">
                    <div class="risk-label-high">⚠️ High Risk</div>
                    <div class="risk-prob">Probability: <b>{prob:.1%}</b></div>
                    <div style='margin-top:12px; font-size:0.83rem; color:#bbb;'>
                        Recommend immediate consultation with a cardiologist.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-box-low">
                    <div class="risk-label-low">✅ Low Risk</div>
                    <div class="risk-prob">Probability: <b>{prob:.1%}</b></div>
                    <div style='margin-top:12px; font-size:0.83rem; color:#bbb;'>
                        Maintain a healthy lifestyle and routine checkups.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_gauge:
            fig, ax = plt.subplots(figsize=(5, 3))
            bar_color = "#ff4d4d" if prediction == 1 else "#4dff88"
            ax.barh(["Risk Score"], [prob], color=bar_color, height=0.4)
            ax.barh(["Risk Score"], [1 - prob], left=[prob], color="#2a2a2a", height=0.4)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.axvline(0.5, color="#777", linewidth=1, linestyle="--")
            ax.text(0.5, -0.45, "Threshold 50%", ha="center", fontsize=8, color="#777")
            ax.set_title(f"Risk Probability: {prob:.1%}", color="#eee", fontsize=11)
            fig.patch.set_facecolor("#141414")
            ax.set_facecolor("#141414")
            st.pyplot(fig)
            plt.close()
