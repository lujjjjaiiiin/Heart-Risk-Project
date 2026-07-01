import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# GLOBAL STYLE — LIGHT BLUE / ADVANCED THEME
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background: light, layered blue ── */
.stApp {
    background: radial-gradient(circle at 15% 0%, #f0f7ff 0%, transparent 45%),
                radial-gradient(circle at 100% 20%, #e3edff 0%, transparent 50%),
                linear-gradient(160deg, #eef6ff 0%, #e2edfd 45%, #dce9fb 100%);
}

/* ── Sidebar: deep navy for contrast against light body ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1024 0%, #0f172a 100%);
    border-right: 1px solid rgba(96,165,250,0.15);
    width: 285px !important;
}
section[data-testid="stSidebar"] * {
    color: #e7edf7 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    padding: 6px 0 !important;
    letter-spacing: 0.01em;
}
section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 1.05rem !important;
}

/* ── Metric cards: glassmorphism ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(37,99,235,0.18);
    border-radius: 16px;
    padding: 18px 22px;
    box-shadow: 0 8px 28px rgba(37,99,235,0.08);
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stMetricValue"] {
    color: #e11d48 !important;
    font-size: 2.1rem;
    font-weight: 800;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #e11d48, #9f1239);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65em 2.2em;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
    width: 100%;
    box-shadow: 0 6px 20px rgba(225,29,72,0.25);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #f43f5e, #be123c);
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(225,29,72,0.35);
}

/* ── Typography ── */
h1 {
    background: linear-gradient(90deg, #e11d48, #2563eb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
}
h2 { color: #1e293b !important; font-weight: 700 !important; }
h3 { color: #1e293b !important; font-weight: 600 !important; }
p, span, label, .stMarkdown { color: #334155; }
hr { border-color: rgba(37,99,235,0.18); margin: 1.8rem 0; }

.stDataFrame {
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 6px 20px rgba(37,99,235,0.08);
}

.stNumberInput input, .stSelectbox select, .stSelectbox div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid rgba(37,99,235,0.25) !important;
    color: #1e293b !important;
    border-radius: 8px !important;
}

.stTabs [data-baseweb="tab"] {
    color: #64748b;
    font-weight: 500;
    font-size: 0.95rem;
}
.stTabs [aria-selected="true"] {
    color: #e11d48 !important;
    border-bottom-color: #e11d48 !important;
}

/* ── Info / success boxes ── */
div[data-testid="stAlert"] {
    background: rgba(255,255,255,0.75) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(37,99,235,0.2) !important;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# MATPLOTLIB THEME — LIGHT / RED-BLUE PALETTE
# =========================
DARK_BG = "none"
CARD_BG = "none"
GRID_CLR = "#cbd5e1"
TEXT_CLR = "#334155"
RED = "#e11d48"
RED_LIGHT = "#fb7185"
BLUE = "#2563eb"
BLUE_LIGHT = "#60a5fa"
VIOLET = "#7c3aed"   # blend of red + blue, used only as a 3rd tone when needed

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#94a3b8",
    "axes.labelcolor": "#1e293b",
    "axes.titlecolor": "#1e293b",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.color": TEXT_CLR,
    "grid.color": GRID_CLR,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "legend.facecolor": "none",
    "legend.edgecolor": "#94a3b8",
    "legend.fontsize": 9,
    "lines.linewidth": 2.5,
})

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

    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    xgb = XGBClassifier(random_state=42, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    return scaler, pca, rf, lr, xgb, X, X_test, y_test, y_pred_rf, y_pred_lr, y_pred_xgb

scaler, pca, rf, lr, xgb, X, X_test, y_test, y_pred_rf, y_pred_lr, y_pred_xgb = train_models(df)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:28px 0 20px;'>
        <div style='font-size:3.2rem; line-height:1;'>🫀</div>
        <div style='font-size:1.25rem; font-weight:800; color:#fff; margin-top:10px; letter-spacing:-0.02em;'>Heart AI System</div>
        <div style='font-size:0.75rem; color:#8ea0c2; margin-top:5px; letter-spacing:0.08em; text-transform:uppercase;'>Medical ML Dashboard</div>
    </div>
    <hr style='border-color:rgba(96,165,250,0.15); margin:0 0 20px;'>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "nav",
        ["🏠 Overview", "📊 EDA", "🤖 Models", "🧠 Predict"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:rgba(96,165,250,0.15); margin:20px 0;'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.04); padding:14px 16px; border-radius:12px; border:1px solid rgba(225,29,72,0.25); margin-bottom:12px;'>
        <div style='color:#fb7185; font-weight:700; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;'>📊 Dataset</div>
        <div style='color:#c3cee3; font-size:0.82rem; line-height:1.9;'>
            Records &nbsp;<span style='color:#fff; font-weight:600; float:right;'>{len(df):,}</span><br>
            Features &nbsp;<span style='color:#fff; font-weight:600; float:right;'>{df.shape[1]-1}</span><br>
            Target &nbsp;<span style='color:#fff; font-weight:600; float:right;'>Heart Risk</span>
        </div>
    </div>
    <div style='background:rgba(255,255,255,0.04); padding:14px 16px; border-radius:12px; border:1px solid rgba(37,99,235,0.3);'>
        <div style='color:#60a5fa; font-weight:700; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;'>🧠 Models</div>
        <div style='color:#c3cee3; font-size:0.82rem; line-height:2;'>
            🌲 Random Forest<br>📈 Logistic Regression<br>⚡ XGBoost<br>🔬 PCA + Scaling
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; margin-top:24px; color:#3f4c6b; font-size:0.7rem; letter-spacing:0.05em;'>
        TUWAIQ ACADEMY · ML PROJECT © 2026
    </div>
    """, unsafe_allow_html=True)

# =========================
# OVERVIEW
# =========================
if menu == "🏠 Overview":
    st.title("🫀 Heart Disease AI Dashboard")
    st.markdown("<p style='color:#64748b; font-size:1rem; margin-bottom:2rem;'>A machine learning system that analyzes patient health data and predicts heart disease risk.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("High Risk Cases", f"{int(df['Heart_Risk'].sum()):,}")
    col4.metric("Risk Rate", f"{df['Heart_Risk'].mean():.1%}")

    st.markdown("---")
    colA, colB = st.columns([1.3, 1])

    with colA:
        st.subheader("Risk Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df["Heart_Risk"].value_counts()
        wedges, texts, autotexts = ax.pie(
            counts, labels=["Low Risk", "High Risk"], autopct="%1.1f%%",
            startangle=90, colors=[BLUE, RED],
            wedgeprops={"edgecolor": "#eef6ff", "linewidth": 3},
            textprops={"color": TEXT_CLR, "fontsize": 10}
        )
        for at in autotexts:
            at.set_color("#fff")
            at.set_fontweight("bold")
        fig.patch.set_facecolor(DARK_BG)
        st.pyplot(fig)
        plt.close()

    with colB:
        st.subheader("Key Insights")
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.75); backdrop-filter:blur(12px); padding:22px; border-radius:16px; border:1px solid rgba(37,99,235,0.18); box-shadow:0 8px 24px rgba(37,99,235,0.08);'>
            <div style='color:#e11d48; font-weight:700; margin-bottom:10px;'>📌 Findings</div>
            <ul style='color:#475569; line-height:2; margin:0; padding-left:18px;'>
                <li>Balanced classification target</li>
                <li>Binary medical indicators dominate</li>
                <li>Age & lifestyle are key risk factors</li>
                <li>Clean data — no missing values</li>
            </ul>
            <div style='color:#2563eb; font-weight:700; margin:14px 0 10px;'>🧠 ML Pipeline</div>
            <ul style='color:#475569; line-height:2; margin:0; padding-left:18px;'>
                <li>StandardScaler normalization</li>
                <li>PCA dimensionality reduction (93%)</li>
                <li>3 models trained & compared</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Feature Correlation Heatmap")

    corr = df.corr().round(2)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu_r",
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        textfont={"size": 9, "color": "white"},
        hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>Correlation: %{z}<extra></extra>",
        colorbar=dict(
            title=dict(text="Correlation", font=dict(color="#334155")),
            tickfont=dict(color="#334155"),
            thickness=15,
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155", family="Inter"),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10), showgrid=False),
        yaxis=dict(tickfont=dict(size=10), showgrid=False, autorange="reversed"),
        height=550,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# EDA
# =========================
elif menu == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("<p style='color:#64748b; font-size:1rem;'>Interactive analysis of patient health patterns.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1, 2.2])

    with col1:
        st.subheader("🎛 Controls")
        feature = st.selectbox("Select Feature", [c for c in df.columns if c != "Heart_Risk"])
        chart_type = st.radio("Chart Type", ["Distribution", "Boxplot", "Risk Comparison"])
        st.markdown("---")
        st.info("Explore patterns interactively using the controls above.")

    with col2:
        if chart_type == "Distribution":
            st.subheader(f"Distribution — {feature}")
            fig, ax = plt.subplots(figsize=(7, 4))
            for risk, color, label in zip([0, 1], [BLUE, RED], ["Low Risk", "High Risk"]):
                sns.histplot(df[df["Heart_Risk"] == risk][feature], bins=25, kde=True,
                             color=color, alpha=0.6, label=label, ax=ax)
            ax.legend()
            ax.grid(True)
            ax.set_xlabel(feature)
            fig.patch.set_facecolor(DARK_BG)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        elif chart_type == "Boxplot":
            st.subheader(f"Boxplot — {feature} by Risk")
            fig, ax = plt.subplots(figsize=(7, 4))
            group0 = df[df["Heart_Risk"] == 0][feature].dropna().astype(float).values
            group1 = df[df["Heart_Risk"] == 1][feature].dropna().astype(float).values
            bp = ax.boxplot(
                [group0, group1],
                tick_labels=["Low Risk", "High Risk"],
                patch_artist=True,
                medianprops={"color": "#1e293b", "linewidth": 2},
                flierprops={"marker": "o", "markerfacecolor": RED, "markersize": 3, "alpha": 0.5},
                whiskerprops={"color": "#94a3b8"},
                capprops={"color": "#94a3b8"},
            )
            bp["boxes"][0].set_facecolor(BLUE + "55")
            bp["boxes"][0].set_edgecolor(BLUE)
            bp["boxes"][1].set_facecolor(RED + "55")
            bp["boxes"][1].set_edgecolor(RED)
            ax.grid(True)
            fig.patch.set_facecolor(DARK_BG)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        else:
            st.subheader(f"Risk Comparison — {feature}")
            fig, ax = plt.subplots(figsize=(7, 4))
            means = df.groupby("Heart_Risk")[feature].mean()
            bars = ax.bar(["Low Risk", "High Risk"], means.values, color=[BLUE, RED],
                           width=0.45, edgecolor="#eef6ff", linewidth=1.5)
            for bar, val in zip(bars, means.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * means.max(),
                        f"{val:.2f}", ha="center", color="#1e293b", fontsize=10, fontweight="bold")
            ax.set_ylabel(f"Mean {feature}")
            ax.grid(True, axis="y")
            fig.patch.set_facecolor(DARK_BG)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")
    st.subheader("Full Correlation Matrix")

    corr_eda = df.corr().round(2)
    fig = go.Figure(data=go.Heatmap(
        z=corr_eda.values,
        x=corr_eda.columns,
        y=corr_eda.columns,
        colorscale="RdBu_r",
        zmid=0,
        text=corr_eda.values,
        texttemplate="%{text}",
        textfont={"size": 9, "color": "white"},
        hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>Correlation: %{z}<extra></extra>",
        colorbar=dict(
            title=dict(text="Correlation", font=dict(color="#334155")),
            tickfont=dict(color="#334155"),
            thickness=15,
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155", family="Inter"),
        title=dict(text="Feature Correlation Matrix", font=dict(color="#1e293b", size=14)),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10), showgrid=False),
        yaxis=dict(tickfont=dict(size=10), showgrid=False, autorange="reversed"),
        height=600,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# MODELS
# =========================
elif menu == "🤖 Models":
    st.title("🤖 Model Performance")
    st.markdown("<p style='color:#64748b; font-size:1rem;'>Comparing all three trained models.</p>", unsafe_allow_html=True)
    st.markdown("---")

    acc_rf = accuracy_score(y_test, y_pred_rf)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)

    col1, col2, col3 = st.columns(3)
    col1.metric("🌲 Random Forest", f"{acc_rf:.2%}")
    col2.metric("📈 Logistic Regression", f"{acc_lr:.2%}")
    col3.metric("⚡ XGBoost", f"{acc_xgb:.2%}")

    st.markdown("---")
    st.subheader("Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(7, 4))
    model_names = ["Random Forest", "Logistic\nRegression", "XGBoost"]
    scores = [acc_rf, acc_lr, acc_xgb]
    colors = [RED, BLUE, VIOLET]
    bars = ax.bar(model_names, scores, color=colors, width=0.45, edgecolor="#eef6ff", linewidth=1.5)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{val:.2%}", ha="center", color="#1e293b", fontsize=11, fontweight="bold")
    ax.set_ylim(min(scores) - 0.05, 1.0)
    ax.set_ylabel("Accuracy")
    ax.grid(True, axis="y")
    fig.patch.set_facecolor(DARK_BG)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Confusion Matrices")
    colA, colB, colC = st.columns(3)
    for col, title, y_pred, cmap in [
        (colA, "Random Forest", y_pred_rf, "Reds"),
        (colB, "Logistic Regression", y_pred_lr, "Blues"),
        (colC, "XGBoost", y_pred_xgb, "Purples"),
    ]:
        with col:
            st.markdown(f"**{title}**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(3.8, 3.4))
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                        linewidths=1.5, linecolor="#eef6ff",
                        xticklabels=["Low", "High"], yticklabels=["Low", "High"],
                        annot_kws={"size": 14, "weight": "bold"})
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("Actual", fontsize=9)
            fig.patch.set_facecolor(DARK_BG)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")
    best_acc = max(acc_rf, acc_lr, acc_xgb)
    best_name = ["Random Forest", "XGBoost", "Logistic Regression"][
        [acc_rf, acc_xgb, acc_lr].index(best_acc)
    ]
    st.success(f"🏆 Best Model: **{best_name}** — {best_acc:.2%} accuracy")

    st.markdown("---")
    st.subheader("PCA Explained Variance")
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, len(cumvar)+1), cumvar, color=RED, linewidth=2.5,
            marker="o", markersize=6, markerfacecolor="#eef6ff", markeredgecolor=RED)
    ax.fill_between(range(1, len(cumvar)+1), cumvar, alpha=0.12, color=RED)
    ax.axhline(0.93, linestyle="--", color=BLUE, linewidth=1.5, label="93% threshold")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Explained Variance Curve")
    ax.legend()
    ax.grid(True)
    fig.patch.set_facecolor(DARK_BG)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# =========================
# PREDICT
# =========================
elif menu == "🧠 Predict":
    st.title("🧠 Heart Disease Risk Assessment")
    st.markdown("<p style='color:#64748b; font-size:1rem;'>Fill in patient details to get an AI-powered risk prediction.</p>", unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🧬 Symptoms", "🏥 Medical History", "🧑 Lifestyle"])
    inputs = {}

    with tab1:
        col1, col2 = st.columns(2)
        symptom_cols = [
            "Chest_Pain", "Shortness_of_Breath", "Fatigue", "Palpitations",
            "Dizziness", "Swelling", "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea"
        ]
        for i, col in enumerate(symptom_cols):
            if col in X.columns:
                with col1 if i % 2 == 0 else col2:
                    inputs[col] = st.slider(col.replace("_", " "), 0.0, 1.0, 0.0)

    with tab2:
        col1, col2 = st.columns(2)
        medical_cols = [
            "High_BP", "High_Cholesterol", "Diabetes", "Smoking", "Obesity", "Family_History"
        ]
        for i, col in enumerate(medical_cols):
            if col in X.columns:
                with col1 if i % 2 == 0 else col2:
                    inputs[col] = st.slider(col.replace("_", " "), 0.0, 1.0, 0.0)

    with tab3:
        st.subheader("🚶 Lifestyle & Demographics")
        col1, col2 = st.columns(2)
        lifestyle_cols = ["Sedentary_Lifestyle", "Chronic_Stress"]
        for i, col in enumerate(lifestyle_cols):
            if col in X.columns:
                with col1 if i % 2 == 0 else col2:
                    inputs[col] = st.slider(col.replace("_", " "), 0.0, 1.0, 0.0)

    st.markdown("---")
    inputs["Age"] = st.number_input("Age", 0, 100, 30)
    gender = st.radio("Gender", ["Male", "Female"])
    inputs["Gender"] = 0 if gender == "Male" else 1

    st.markdown("---")
    model_name = st.selectbox(
        "Choose Prediction Model",
        ["Random Forest", "Logistic Regression", "XGBoost"]
    )
    model_choice = rf if model_name == "Random Forest" else (lr if model_name == "Logistic Regression" else xgb)

    if st.button("🔍 Predict Risk"):
        input_df = pd.DataFrame([inputs])[X.columns]
        input_scaled = scaler.transform(input_df)
        input_pca = pca.transform(input_scaled)

        prediction = model_choice.predict(input_pca)[0]
        prob = model_choice.predict_proba(input_pca)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")

        col_res, col_gauge = st.columns([1, 1.6])

        with col_res:
            if prediction == 1:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#fff1f2,#ffe4e6); border:1px solid #e11d48; border-radius:16px; padding:28px 20px; text-align:center; box-shadow:0 8px 24px rgba(225,29,72,0.12);'>
                    <div style='font-size:2.4rem; font-weight:800; color:#e11d48;'>⚠️ High Risk</div>
                    <div style='color:#7f1d3a; margin-top:8px; font-size:0.9rem;'>
                        Probability: <span style='color:#e11d48; font-weight:700; font-size:1.1rem;'>{prob:.1%}</span>
                    </div>
                    <div style='margin-top:16px; font-size:0.82rem; color:#9f4a5f; line-height:1.6;'>
                        Immediate consultation with a cardiologist is strongly recommended.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#eff6ff,#dbeafe); border:1px solid #2563eb; border-radius:16px; padding:28px 20px; text-align:center; box-shadow:0 8px 24px rgba(37,99,235,0.12);'>
                    <div style='font-size:2.4rem; font-weight:800; color:#2563eb;'>✅ Low Risk</div>
                    <div style='color:#1e3a72; margin-top:8px; font-size:0.9rem;'>
                        Probability: <span style='color:#2563eb; font-weight:700; font-size:1.1rem;'>{prob:.1%}</span>
                    </div>
                    <div style='margin-top:16px; font-size:0.82rem; color:#4a6fa8; line-height:1.6;'>
                        Maintain a healthy lifestyle and schedule routine checkups.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_gauge:
            bar_color = RED if prediction == 1 else BLUE
            fig, ax = plt.subplots(figsize=(5.5, 3))
            ax.barh(["Risk Score"], [prob], color=bar_color, height=0.35, zorder=3)
            ax.barh(["Risk Score"], [1 - prob], left=[prob], color="#dbe9fb", height=0.35, zorder=3)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability", fontsize=9)
            ax.axvline(0.5, color="#94a3b8", linewidth=1.2, linestyle="--", zorder=4)
            ax.text(0.5, -0.62, "Threshold", ha="center", fontsize=8, color="#64748b")
            ax.text(prob / 2, 0, f"{prob:.0%}", ha="center", va="center",
                    color="#fff", fontsize=12, fontweight="bold", zorder=5)
            ax.set_title(f"Model: {model_name} · Risk: {prob:.1%}", fontsize=10)
            ax.grid(False)
            fig.patch.set_facecolor(DARK_BG)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
