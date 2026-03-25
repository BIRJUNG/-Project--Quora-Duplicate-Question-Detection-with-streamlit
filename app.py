import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Quora Duplicate Question Detector",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        color: white;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f766e 100%);
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #dbeafe;
        line-height: 1.6;
    }

    .metric-card {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1rem 1.1rem;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
        text-align: center;
    }

    .metric-title {
        color: #93c5fd;
        font-size: 0.9rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: white;
        font-size: 1.5rem;
        font-weight: 800;
    }

    .section-card {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1.2rem;
        border-radius: 16px;
        margin-top: 0.8rem;
        margin-bottom: 1rem;
    }

    .small-note {
        color: #cbd5e1;
        font-size: 0.92rem;
    }

    .success-box {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 14px;
        padding: 1rem;
    }

    .warning-box {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 14px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# HELPER FUNCTIONS
# =========================
def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def token_set(text):
    return set(normalize_text(text).split())

def safe_word_count(text):
    return len(str(text).split())

def safe_char_count(text):
    return len(str(text))

def jaccard_similarity(q1, q2):
    s1 = token_set(q1)
    s2 = token_set(q2)
    if len(s1.union(s2)) == 0:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def token_overlap_ratio(q1, q2):
    s1 = token_set(q1)
    s2 = token_set(q2)
    denom = min(len(s1), len(s2))
    if denom == 0:
        return 0.0
    return len(s1.intersection(s2)) / denom

def cosine_similarity_matrix(a, b):
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return np.sum(a_norm * b_norm, axis=1)

def build_single_pair_features(q1, q2, embedder):
    q1_emb = embedder.encode([q1], convert_to_numpy=True)
    q2_emb = embedder.encode([q2], convert_to_numpy=True)

    cosine_sim = cosine_similarity_matrix(q1_emb, q2_emb).reshape(-1, 1)
    abs_diff = np.abs(q1_emb - q2_emb)
    prod = q1_emb * q2_emb

    lexical = np.array([[
        safe_word_count(q1),
        safe_word_count(q2),
        safe_char_count(q1),
        safe_char_count(q2),
        float(safe_word_count(q1) - safe_word_count(q2)),
        float(safe_char_count(q1) - safe_char_count(q2)),
        jaccard_similarity(q1, q2),
        token_overlap_ratio(q1, q2)
    ]], dtype=float)

    X = np.hstack([cosine_sim, lexical, abs_diff, prod])
    return X

def predict_duplicate(q1, q2, embedder, trained_model, threshold):
    X = build_single_pair_features(q1, q2, embedder)
    prob = trained_model.predict_proba(X)[:, 1][0]
    pred = int(prob >= threshold)
    label = "Duplicate" if pred == 1 else "Not Duplicate"

    return {
        "question1": q1,
        "question2": q2,
        "duplicate_probability": float(prob),
        "prediction": pred,
        "label": label,
        "cosine_similarity": float(cosine_similarity_matrix(
            embedder.encode([q1], convert_to_numpy=True),
            embedder.encode([q2], convert_to_numpy=True)
        )[0]),
        "jaccard_similarity": float(jaccard_similarity(q1, q2)),
        "token_overlap_ratio": float(token_overlap_ratio(q1, q2)),
        "q1_word_count": int(safe_word_count(q1)),
        "q2_word_count": int(safe_word_count(q2)),
    }

def build_batch_features(df, embedder):
    q1_list = df["question1"].astype(str).tolist()
    q2_list = df["question2"].astype(str).tolist()

    q1_emb = embedder.encode(q1_list, convert_to_numpy=True, batch_size=128, show_progress_bar=False)
    q2_emb = embedder.encode(q2_list, convert_to_numpy=True, batch_size=128, show_progress_bar=False)

    cosine_sim = cosine_similarity_matrix(q1_emb, q2_emb).reshape(-1, 1)
    abs_diff = np.abs(q1_emb - q2_emb)
    prod = q1_emb * q2_emb

    lexical_features = pd.DataFrame({
        "q1_word_count": df["question1"].apply(safe_word_count).values,
        "q2_word_count": df["question2"].apply(safe_word_count).values,
        "q1_char_count": df["question1"].apply(safe_char_count).values,
        "q2_char_count": df["question2"].apply(safe_char_count).values,
        "word_count_diff": (
            df["question1"].apply(safe_word_count).values -
            df["question2"].apply(safe_word_count).values
        ).astype(float),
        "char_count_diff": (
            df["question1"].apply(safe_char_count).values -
            df["question2"].apply(safe_char_count).values
        ).astype(float),
        "jaccard_similarity": [
            jaccard_similarity(a, b) for a, b in zip(df["question1"], df["question2"])
        ],
        "token_overlap_ratio": [
            token_overlap_ratio(a, b) for a, b in zip(df["question1"], df["question2"])
        ]
    })

    X = np.hstack([
        cosine_sim,
        lexical_features.values.astype(float),
        abs_diff,
        prod
    ])
    return X, lexical_features, cosine_sim.flatten()

def probability_band(prob):
    if prob >= 0.85:
        return "Very High"
    elif prob >= 0.70:
        return "High"
    elif prob >= 0.50:
        return "Moderate"
    elif prob >= 0.30:
        return "Low"
    return "Very Low"


# =========================
# CACHED LOADERS
# =========================
@st.cache_resource
def load_embedder(model_name):
    return SentenceTransformer(model_name)

@st.cache_resource
def load_classifier(model_path):
    return joblib.load(model_path)

@st.cache_data
def load_metadata(metadata_path):
    with open(metadata_path, "r") as f:
        return json.load(f)


# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ App Settings")

artifact_dir = st.sidebar.text_input("Artifact folder", value="artifacts")
model_path = os.path.join(artifact_dir, "quora_duplicate_classifier.joblib")
metadata_path = os.path.join(artifact_dir, "metadata.json")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Expected files")
st.sidebar.code(
    "artifacts/\n"
    "├── quora_duplicate_classifier.joblib\n"
    "└── metadata.json"
)

# Default values in case metadata not found
default_embedding_model = "all-MiniLM-L6-v2"
default_threshold = 0.50
default_best_model_name = "Unknown"


# =========================
# LOAD ARTIFACTS
# =========================
metadata = None
classifier = None
embedder = None
artifact_error = False

try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = load_metadata(metadata_path)
    embedding_model_name = metadata.get("embedding_model_name", default_embedding_model)
    best_threshold = float(metadata.get("best_threshold", default_threshold))
    best_model_name = metadata.get("best_model_name", default_best_model_name)
    test_metrics = metadata.get("test_metrics", {})
    feature_summary = metadata.get("feature_summary", {})

    classifier = load_classifier(model_path)
    embedder = load_embedder(embedding_model_name)

except Exception as e:
    artifact_error = True
    embedding_model_name = default_embedding_model
    best_threshold = default_threshold
    best_model_name = default_best_model_name
    test_metrics = {}
    feature_summary = {}
    st.sidebar.error(f"Artifact loading error:\n{e}")


# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🔎 Quora Duplicate Question Detector</div>
    <div class="hero-subtitle">
        Semantic duplicate-question detection using SentenceTransformer embeddings,
        pairwise feature engineering, and a tuned machine learning classifier.
    </div>
</div>
""", unsafe_allow_html=True)

if artifact_error:
    st.markdown("""
    <div class="warning-box">
        <b>Artifacts not loaded.</b><br>
        Place your trained files inside <code>artifacts/</code>:
        <ul>
            <li><code>quora_duplicate_classifier.joblib</code></li>
            <li><code>metadata.json</code></li>
        </ul>
        Then rerun the app.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# =========================
# TOP METRICS
# =========================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Best Model</div>
        <div class="metric-value">{best_model_name}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Embedding Model</div>
        <div class="metric-value" style="font-size:1rem;">{embedding_model_name}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Threshold</div>
        <div class="metric-value">{best_threshold:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    f1_value = test_metrics.get("f1", None)
    display_f1 = f"{f1_value:.4f}" if f1_value is not None else "N/A"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Test F1</div>
        <div class="metric-value">{display_f1}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction", "📘 Project Details"])


# =========================
# TAB 1: SINGLE PREDICTION
# =========================
with tab1:
    st.markdown("### Ask the model whether two questions are duplicates")

    col_left, col_right = st.columns(2)

    with col_left:
        q1 = st.text_area(
            "Question 1",
            value="How can I learn Python quickly?",
            height=140
        )

    with col_right:
        q2 = st.text_area(
            "Question 2",
            value="What is the fastest way to learn Python?",
            height=140
        )

    col_btn1, col_btn2 = st.columns([1, 5])

    with col_btn1:
        run_pred = st.button("Predict", use_container_width=True)

    with col_btn2:
        custom_threshold = st.slider(
            "Prediction threshold",
            min_value=0.10,
            max_value=0.95,
            value=float(best_threshold),
            step=0.01
        )

    if run_pred:
        if not q1.strip() or not q2.strip():
            st.warning("Please enter both questions.")
        else:
            result = predict_duplicate(
                q1=q1,
                q2=q2,
                embedder=embedder,
                trained_model=classifier,
                threshold=custom_threshold
            )

            prob = result["duplicate_probability"]
            band = probability_band(prob)

            st.markdown("### Prediction Result")

            if result["prediction"] == 1:
                st.markdown(f"""
                <div class="success-box">
                    <h4 style="margin-bottom:0.4rem;">✅ {result["label"]}</h4>
                    <div class="small-note">
                        The model believes these two questions are semantically similar enough
                        to be treated as duplicates.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4 style="margin-bottom:0.4rem;">❌ {result["label"]}</h4>
                    <div class="small-note">
                        The model believes these questions are meaningfully different.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.progress(float(prob))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Duplicate Probability", f"{prob:.2%}")
            m2.metric("Confidence Band", band)
            m3.metric("Cosine Similarity", f"{result['cosine_similarity']:.4f}")
            m4.metric("Jaccard Similarity", f"{result['jaccard_similarity']:.4f}")

            st.markdown("### Feature Snapshot")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Q1 Word Count", result["q1_word_count"])
            d2.metric("Q2 Word Count", result["q2_word_count"])
            d3.metric("Token Overlap Ratio", f"{result['token_overlap_ratio']:.4f}")
            d4.metric("Applied Threshold", f"{custom_threshold:.2f}")

            with st.expander("Show raw prediction JSON"):
                st.json({
                    "question1": result["question1"],
                    "question2": result["question2"],
                    "duplicate_probability": round(result["duplicate_probability"], 6),
                    "prediction": result["prediction"],
                    "label": result["label"],
                    "cosine_similarity": round(result["cosine_similarity"], 6),
                    "jaccard_similarity": round(result["jaccard_similarity"], 6),
                    "token_overlap_ratio": round(result["token_overlap_ratio"], 6)
                })


# =========================
# TAB 2: BATCH PREDICTION
# =========================
with tab2:
    st.markdown("### Upload a CSV for batch scoring")
    st.markdown(
        """
        Your CSV must contain these columns:
        - `question1`
        - `question2`
        """
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            required_cols = {"question1", "question2"}
            if not required_cols.issubset(batch_df.columns):
                st.error("CSV must contain `question1` and `question2` columns.")
            else:
                st.success(f"File loaded successfully. Rows: {len(batch_df):,}")

                preview_rows = st.slider("Preview rows", 5, min(50, len(batch_df)), min(10, len(batch_df)))
                st.dataframe(batch_df.head(preview_rows), use_container_width=True)

                if st.button("Run Batch Prediction", use_container_width=True):
                    with st.spinner("Encoding text and generating predictions..."):
                        X_batch, lexical_batch, cosine_vals = build_batch_features(batch_df.copy(), embedder)
                        batch_prob = classifier.predict_proba(X_batch)[:, 1]
                        batch_pred = (batch_prob >= best_threshold).astype(int)

                        result_df = batch_df.copy()
                        result_df["duplicate_probability"] = batch_prob
                        result_df["prediction"] = batch_pred
                        result_df["label"] = np.where(batch_pred == 1, "Duplicate", "Not Duplicate")
                        result_df["cosine_similarity"] = cosine_vals
                        result_df["jaccard_similarity"] = lexical_batch["jaccard_similarity"].values
                        result_df["token_overlap_ratio"] = lexical_batch["token_overlap_ratio"].values

                    st.markdown("### Batch Prediction Output")

                    b1, b2, b3 = st.columns(3)
                    b1.metric("Total Rows", f"{len(result_df):,}")
                    b2.metric("Predicted Duplicates", int((result_df["prediction"] == 1).sum()))
                    b3.metric("Average Duplicate Probability", f"{result_df['duplicate_probability'].mean():.2%}")

                    st.dataframe(result_df, use_container_width=True)

                    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv_bytes,
                        file_name="quora_duplicate_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"Error while processing file: {e}")


# =========================
# TAB 3: PROJECT DETAILS
# =========================
with tab3:
    st.markdown("### Model and Feature Pipeline")

    st.markdown("""
    <div class="section-card">
        <b>Pipeline summary</b><br><br>
        1. Encode <code>question1</code> and <code>question2</code> separately using a SentenceTransformer.<br>
        2. Build pairwise semantic and lexical features.<br>
        3. Feed combined features into the trained classifier.<br>
        4. Convert probability to final class using the tuned threshold from validation.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("#### Features used")
        st.write("- Cosine similarity between question embeddings")
        st.write("- Word counts and character counts")
        st.write("- Word-count and character-count differences")
        st.write("- Jaccard similarity")
        st.write("- Token overlap ratio")
        st.write("- Absolute embedding difference")
        st.write("- Element-wise embedding product")

    with right:
        st.markdown("#### Saved metadata")
        st.json({
            "embedding_model_name": embedding_model_name,
            "best_model_name": best_model_name,
            "best_threshold": best_threshold,
            "feature_summary": feature_summary,
            "test_metrics": test_metrics
        })

    st.markdown("#### Notes")
    st.info(
        "This app must use the same feature engineering logic as the training notebook. "
        "If you change feature order or definitions, predictions will become incorrect."
    )