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
                linear-gradient(160deg, #ffffff 0%, #e2edfd 45%, #dce9fb 100%);
}

/* ── Sidebar: soft glowing pink gradient ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #2a0a2e 0%, #831843 28%, #ec4899 60%, #db2777 85%, #2a0a2e 100%);
    border-right: 1px solid rgba(255,255,255,0.12);
    box-shadow: 6px 0 40px rgba(236,72,153,0.35);
    width: 285px !important;
    position: relative;
}
section[data-testid="stSidebar"]::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 30% 15%, rgba(255,255,255,0.14) 0%, transparent 40%);
    pointer-events: none;
}
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
    text-shadow: 0 1px 4px rgba(0,0,0,0.35);
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 1.08rem !important;
    font-weight: 600 !important;
    padding: 9px 12px !important;
    margin-bottom: 4px;
    letter-spacing: 0.01em;
    border-radius: 10px;
    transition: all 0.2s ease;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.14);
}
section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 1.08rem !important;
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
    color: #db2777 !important;
    font-size: 2.1rem;
    font-weight: 800;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #ec4899, #be185d);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65em 2.2em;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
    width: 100%;
    box-shadow: 0 6px 20px rgba(236,72,153,0.25);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #f472b6, #9d174d);
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(236,72,153,0.35);
}

/* ── Typography ── */
h1 {
    background: linear-gradient(90deg, #ec4899, #2563eb);
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
    color: #db2777 !important;
    border-bottom-color: #db2777 !important;
}

/* ── Info / success boxes ── */
div[data-testid="stAlert"] {
    background: rgba(255,255,255,0.75) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(37,99,235,0.2) !important;
    border-radius: 12px;
}

/* ── Chart cards: crisp white frame behind every figure for max clarity ── */
[data-testid="stImage"] {
    background: #ffffff;
    border-radius: 16px;
    padding: 14px;
    border: 1px solid rgba(37,99,235,0.15);
    box-shadow: 0 10px 30px rgba(37,99,235,0.10);
}
[data-testid="stImage"] img {
    border-radius: 8px;
}
div[data-testid="stPlotlyChart"] {
    background: #ffffff;
    border-radius: 16px;
    padding: 10px;
    border: 1px solid rgba(37,99,235,0.15);
    box-shadow: 0 10px 30px rgba(37,99,235,0.10);
}
</style>
""", unsafe_allow_html=True)

# =========================
# MATPLOTLIB THEME — LIGHT / RED-BLUE PALETTE / HIGH CLARITY
# =========================
DARK_BG = "#ffffff"
CARD_BG = "#ffffff"
GRID_CLR = "#94a3b8"
TEXT_CLR = "#1e293b"
RED = "#e11d48"
RED_LIGHT = "#fb7185"
BLUE = "#2563eb"
BLUE_LIGHT = "#60a5fa"
VIOLET = "#7c3aed"   # blend of red + blue, used only as a 3rd tone when needed

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#334155",
    "axes.linewidth": 1.4,
    "axes.labelcolor": "#0f172a",
    "axes.titlecolor": "#0f172a",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.color": "#1e293b",
    "ytick.color": "#1e293b",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "text.color": TEXT_CLR,
    "grid.color": GRID_CLR,
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "grid.linewidth": 0.9,
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#94a3b8",
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "lines.linewidth": 3.4,
    "figure.dpi": 170,
    "savefig.dpi": 170,
})

# =========================
# SHARED UI HELPERS — keeps every chart the same "family" & crystal clear
# =========================
def section_header(text):
    """Unified section title used above every single chart in the app."""
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:12px; margin:6px 0 16px;'>
        <div style='width:6px; height:26px; border-radius:4px;
                    background:linear-gradient(180deg,#ec4899,#2563eb);
                    box-shadow:0 0 12px rgba(236,72,153,0.35);'></div>
        <div style='font-size:1.22rem; font-weight:800; color:#0f172a; letter-spacing:-0.01em;'>{text}</div>
    </div>
    """, unsafe_allow_html=True)

# Consistent figure sizes so every chart feels like the same family
FIG_WIDE   = (8.6, 4.4)   # full-width bar / line charts
FIG_SIDE   = (5.4, 4.4)   # narrower column charts (paired with a card)
FIG_SQUARE = (4.3, 3.9)   # small multiples (confusion matrices)

def render_corr_heatmap(corr_df, height=620):
    """Polished, always-legible correlation heatmap: bold per-cell text
    that auto-switches white/navy for contrast, clean gaps between cells,
    and a soft red-blue diverging palette matched to the rest of the app."""
    z = corr_df.values
    labels = corr_df.columns.tolist()

    custom_scale = [
        [0.00, "#1e3a8a"],
        [0.25, "#3b82f6"],
        [0.50, "#f8fafc"],
        [0.75, "#f87171"],
        [1.00, "#991b1b"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale=custom_scale,
        zmid=0, zmin=-1, zmax=1,
        xgap=3, ygap=3,
        hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Correlation", font=dict(color="#0f172a", size=12, family="Inter")),
            tickfont=dict(color="#0f172a", size=11, family="Inter"),
            thickness=16,
            outlinewidth=0,
        ),
    ))

    annotations = []
    for i, ylab in enumerate(labels):
        for j, xlab in enumerate(labels):
            val = z[i][j]
            color = "#ffffff" if abs(val) > 0.55 else "#0f172a"
            annotations.append(dict(
                x=xlab, y=ylab, text=f"<b>{val:.2f}</b>",
                showarrow=False,
                font=dict(color=color, size=11, family="Inter"),
            ))

    fig.update_layout(
        annotations=annotations,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Inter"),
        xaxis=dict(tickangle=-45, tickfont=dict(size=11, color="#1e293b"), showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(size=11, color="#1e293b"), showgrid=False, zeroline=False, autorange="reversed"),
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig

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
        <div style='font-size:3.2rem; line-height:1; filter:drop-shadow(0 0 14px rgba(255,255,255,0.5));'>🫀</div>
        <div style='font-size:1.25rem; font-weight:800; color:#fff; margin-top:10px; letter-spacing:-0.02em;'>Heart AI System</div>
        <div style='font-size:0.75rem; color:#ffd7dd; margin-top:5px; letter-spacing:0.08em; text-transform:uppercase; opacity:0.9;'>Medical ML Dashboard</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.25); margin:0 0 20px;'>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "nav",
        ["🏠 Overview", "📊 EDA", "🤖 Models", "🧠 Predict"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:rgba(96,165,250,0.15); margin:20px 0;'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:rgba(0,0,0,0.28); padding:14px 16px; border-radius:12px; border:1px solid rgba(255,255,255,0.3); margin-bottom:12px; backdrop-filter:blur(6px);'>
        <div style='color:#ffe4e9; font-weight:800; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;'>📊 Dataset</div>
        <div style='color:#ffffff; font-size:0.82rem; line-height:1.9;'>
            Records &nbsp;<span style='color:#fff; font-weight:700; float:right;'>{len(df):,}</span><br>
            Features &nbsp;<span style='color:#fff; font-weight:700; float:right;'>{df.shape[1]-1}</span><br>
            Target &nbsp;<span style='color:#fff; font-weight:700; float:right;'>Heart Risk</span>
        </div>
    </div>
    <div style='background:rgba(37,99,235,0.35); padding:14px 16px; border-radius:12px; border:1px solid rgba(147,197,253,0.5); backdrop-filter:blur(6px);'>
        <div style='color:#dbeafe; font-weight:800; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;'>🧠 Models</div>
        <div style='color:#ffffff; font-size:0.82rem; line-height:2;'>
            🌲 Random Forest<br>📈 Logistic Regression<br>⚡ XGBoost<br>🔬 PCA + Scaling
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; margin-top:24px; color:#ffd7dd; font-size:0.7rem; letter-spacing:0.05em; opacity:0.85;'>
        TUWAIQ ACADEMY · ML PROJECT © 2026
    </div>
    """, unsafe_allow_html=True)

# =========================
# OVERVIEW
# =========================
if menu == "🏠 Overview":
    st.title("Heart Disease AI Dashboard")
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
        section_header("Risk Distribution")
        fig, ax = plt.subplots(figsize=FIG_SIDE)
        counts = df["Heart_Risk"].value_counts()
        wedges, texts, autotexts = ax.pie(
            counts, labels=["Low Risk", "High Risk"], autopct="%1.1f%%",
            startangle=90, colors=[BLUE, RED],
            wedgeprops={"edgecolor": "#ffffff", "linewidth": 3},
            textprops={"color": TEXT_CLR, "fontsize": 11, "fontweight": "bold"}
        )
        for at in autotexts:
            at.set_color("#fff")
            at.set_fontweight("bold")
            at.set_fontsize(12)
        fig.patch.set_facecolor(DARK_BG)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with colB:
        section_header("Key Insights")
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.75); backdrop-filter:blur(12px); padding:22px; border-radius:16px; border:1px solid rgba(37,99,235,0.18); box-shadow:0 8px 24px rgba(37,99,235,0.08);'>
            <div style='color:#db2777; font-weight:700; margin-bottom:10px;'>📌 Findings</div>
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
    section_header("Feature Correlation Heatmap")
    st.plotly_chart(render_corr_heatmap(df.corr().round(2)), use_container_width=True)

# =========================
# EDA
# =========================
elif menu == "📊 EDA":
    st.title("Exploratory Data Analysis")
    st.markdown("<p style='color:#64748b; font-size:1rem;'>Interactive analysis of patient health patterns.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1, 2.2])

    with col1:
        section_header("Controls")
        feature = st.selectbox("Select Feature", [c for c in df.columns if c != "Heart_Risk"])
        chart_type = st.radio("Chart Type", ["Distribution", "Boxplot", "Risk Comparison"])
        st.markdown("---")
        st.info("Explore patterns interactively using the controls above.")

    with col2:
        if chart_type == "Distribution":
            section_header(f"Distribution — {feature}")
            fig, ax = plt.subplots(figsize=FIG_WIDE)
            for risk, color, label in zip([0, 1], [BLUE, RED], ["Low Risk", "High Risk"]):
                sns.histplot(df[df["Heart_Risk"] == risk][feature], bins=25, kde=True,
                             color=color, alpha=0.55, label=label, ax=ax,
                             edgecolor="#ffffff", linewidth=0.6,
                             line_kws={"linewidth": 3.4})
            ax.legend()
            ax.grid(True)
            ax.set_xlabel(feature)
            fig.patch.set_facecolor(DARK_BG)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        elif chart_type == "Boxplot":
            section_header(f"Boxplot — {feature} by Risk")
            fig, ax = plt.subplots(figsize=FIG_WIDE)
            group0 = df[df["Heart_Risk"] == 0][feature].dropna().astype(float).values
            group1 = df[df["Heart_Risk"] == 1][feature].dropna().astype(float).values
            bp = ax.boxplot(
                [group0, group1],
                tick_labels=["Low Risk", "High Risk"],
                patch_artist=True,
                medianprops={"color": "#1e293b", "linewidth": 2.4},
                flierprops={"marker": "o", "markerfacecolor": RED, "markersize": 4, "alpha": 0.6},
                whiskerprops={"color": "#334155", "linewidth": 1.6},
                capprops={"color": "#334155", "linewidth": 1.6},
                boxprops={"linewidth": 2},
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
            section_header(f"Risk Comparison — {feature}")
            fig, ax = plt.subplots(figsize=FIG_WIDE)
            means = df.groupby("Heart_Risk")[feature].mean()
            bars = ax.bar(["Low Risk", "High Risk"], means.values, color=[BLUE, RED],
                           width=0.45, edgecolor="#ffffff", linewidth=1.8)
            for bar, val in zip(bars, means.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * means.max(),
                        f"{val:.2f}", ha="center", color="#1e293b", fontsize=12, fontweight="bold")
            ax.set_ylabel(f"Mean {feature}")
            ax.grid(True, axis="y")
            fig.patch.set_facecolor(DARK_BG)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")
    section_header("Full Correlation Matrix")
    st.plotly_chart(render_corr_heatmap(df.corr().round(2), height=650), use_container_width=True)

# =========================
# MODELS
# =========================
elif menu == "🤖 Models":
    st.title("Model Performance")
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
    section_header("Accuracy Comparison")
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    model_names = ["Random Forest", "Logistic\nRegression", "XGBoost"]
    scores = [acc_rf, acc_lr, acc_xgb]
    colors = [RED, BLUE, VIOLET]
    bars = ax.bar(model_names, scores, color=colors, width=0.45, edgecolor="#ffffff", linewidth=1.8)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{val:.2%}", ha="center", color="#0f172a", fontsize=13, fontweight="bold")
    ax.set_ylim(min(scores) - 0.05, 1.0)
    ax.set_ylabel("Accuracy")
    ax.grid(True, axis="y")
    fig.patch.set_facecolor(DARK_BG)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    section_header("Confusion Matrices")
    colA, colB, colC = st.columns(3)
    for col, title, y_pred, cmap in [
        (colA, "Random Forest", y_pred_rf, "Reds"),
        (colB, "Logistic Regression", y_pred_lr, "Blues"),
        (colC, "XGBoost", y_pred_xgb, "Purples"),
    ]:
        with col:
            st.markdown(f"<div style='text-align:center; font-weight:800; color:#0f172a; margin-bottom:6px;'>{title}</div>", unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=FIG_SQUARE)
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                        linewidths=2, linecolor="#ffffff",
                        xticklabels=["Low", "High"], yticklabels=["Low", "High"],
                        annot_kws={"size": 16, "weight": "bold"}, cbar=False)
            ax.set_xlabel("Predicted", fontsize=11, fontweight="bold")
            ax.set_ylabel("Actual", fontsize=11, fontweight="bold")
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
    section_header("PCA Explained Variance")
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.plot(range(1, len(cumvar)+1), cumvar, color=RED, linewidth=3.4,
            marker="o", markersize=7, markerfacecolor="#ffffff", markeredgecolor=RED, markeredgewidth=2)
    ax.fill_between(range(1, len(cumvar)+1), cumvar, alpha=0.12, color=RED)
    ax.axhline(0.93, linestyle="--", color=BLUE, linewidth=2, label="93% threshold")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
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
    st.title("Heart Disease Risk Assessment")
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
        section_header("🚶 Lifestyle & Demographics")
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
        section_header("Prediction Result")

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
            fig, ax = plt.subplots(figsize=(6.2, 2.9))
            ax.barh(["Risk Score"], [prob], color=bar_color, height=0.42, zorder=3)
            ax.barh(["Risk Score"], [1 - prob], left=[prob], color="#e2e8f0", height=0.42, zorder=3)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.65, 0.65)
            ax.set_yticks([])
            ax.axvline(0.5, color="#334155", linewidth=1.8, linestyle="--", zorder=4)
            ax.text(0.5, 0.48, "50% Threshold", ha="center", va="bottom",
                    fontsize=9.5, color="#334155", fontweight="bold")
            label_x = prob / 2 if prob > 0.12 else prob + 0.08
            label_color = "#fff" if prob > 0.12 else "#0f172a"
            ax.text(label_x, 0, f"{prob:.0%}", ha="center", va="center",
                    color=label_color, fontsize=15, fontweight="bold", zorder=5)
            ax.set_xlabel("Probability", fontsize=11, fontweight="bold", labelpad=10)
            ax.set_title(f"Model: {model_name}", fontsize=13, fontweight="bold", pad=14)
            ax.grid(False)
            fig.patch.set_facecolor(DARK_BG)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
