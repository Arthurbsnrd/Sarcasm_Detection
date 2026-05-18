"""Streamlit production app — Bi-LSTM + GloVe sarcasm headline detector."""

from __future__ import annotations

import os
import pickle
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from monitoring import get_health

# Reduce TensorFlow log noise at import time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "sarcasm_detector.keras"
TOKENIZER_PATH = BASE_DIR / "tokenizer_sarcasm.pkl"
MAX_LEN = 40
VOCAB_SIZE = 20_000
EMBED_DIM = 100
SARCASM_THRESHOLD = 0.5
WEIGHTS_ARCHIVE_ENTRY = "model.weights.h5"


@st.cache_resource(show_spinner=False)
def load_tokenizer():
    """Load pickled tokenizer (fast, ~1 MB)."""
    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(
            f"Tokenizer not found: `{TOKENIZER_PATH}`\n"
            "Place `tokenizer_sarcasm.pkl` at the project root."
        )
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)


def _build_architecture():
    """Recreate the Bi-LSTM architecture (matches training notebook)."""
    from tensorflow.keras import layers

    import tensorflow as tf

    return tf.keras.Sequential(
        [
            layers.Input(shape=(MAX_LEN,)),
            layers.Embedding(VOCAB_SIZE, EMBED_DIM, trainable=False),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )


@st.cache_resource(show_spinner="Loading neural network weights…")
def load_model():
    """Load weights from the .keras archive (avoids broken JSON config)."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: `{MODEL_PATH}`\n"
            "Place `sarcasm_detector.keras` in the `model/` folder."
        )

    model = _build_architecture()
    with zipfile.ZipFile(MODEL_PATH, "r") as archive:
        if WEIGHTS_ARCHIVE_ENTRY not in archive.namelist():
            raise ValueError(
                f"`{WEIGHTS_ARCHIVE_ENTRY}` missing inside `{MODEL_PATH}`."
            )
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive.extract(WEIGHTS_ARCHIVE_ENTRY, tmp_dir)
            weights_path = Path(tmp_dir) / WEIGHTS_ARCHIVE_ENTRY
            model.load_weights(weights_path)
    return model


def preprocess(text: str, tokenizer):
    """Tokenize headline and pad to MAX_LEN (post-padding)."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    sequence = tokenizer.texts_to_sequences([text.strip()])
    return pad_sequences(sequence, maxlen=MAX_LEN, padding="post")


def predict_proba(text: str, model, tokenizer) -> float:
    """Return P(sarcastic) in [0, 1]."""
    padded = preprocess(text, tokenizer)
    return float(model.predict(padded, verbose=0)[0][0])


def render_gauge(label: str, value: float) -> None:
    """Visual probability bar (0–100 %)."""
    pct = int(round(value * 100))
    st.caption(f"**{label}** — {pct}%")
    st.progress(value)


def main() -> None:
    st.set_page_config(
        page_title="Sarcasm Headline Detector",
        page_icon="📰",
        layout="centered",
        initial_sidebar_state="auto",
    )

    with st.sidebar:
        st.subheader("🩺 System health")
        health = get_health(model_loaded=False)
        if health["status"] == "ok":
            st.success("Status: **ok**")
        else:
            st.warning("Status: **degraded** — model files missing")
        st.json(health)
        st.caption(
            "MLOps monitoring payload. For a REST `/health` endpoint, "
            "run `python health_api.py` (Flask, optional)."
        )

    st.markdown(
        """
        # 📰 Sarcasm Headline Detector
        ### *Bi-LSTM + GloVe · News headline classifier*
        """
    )
    st.caption(
        "Paste a news headline below. The model estimates whether the tone is "
        "**sarcastic** (satirical) or **straight news**."
    )
    st.divider()

    # UI first — never block typing while the model loads
    headline = st.text_input(
        label="Headline",
        placeholder="e.g. Local man specialized in doing absolutely nothing receives national award",
        help="Enter a short news-style headline in English.",
    )

    analyze = st.button("Analyze headline", type="primary", use_container_width=True)

    if not analyze:
        st.info("👆 Enter a headline and click **Analyze headline** to run inference.")
        return

    if not headline or not headline.strip():
        st.warning("Please enter a headline before analyzing.")
        return

    # Load assets only when the user submits (lazy load)
    try:
        with st.spinner("Loading tokenizer…"):
            tokenizer = load_tokenizer()
        with st.spinner("Loading model (first run can take up to a minute)…"):
            model = load_model()
    except FileNotFoundError as exc:
        st.error(f"**Deployment assets missing**\n\n{exc}")
        return
    except Exception as exc:
        st.error(
            f"**Failed to load model or tokenizer**\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n"
            "Ensure `model/sarcasm_detector.keras` exists and matches your "
            "TensorFlow version (`tensorflow-cpu` in requirements.txt)."
        )
        return

    with st.spinner("Running inference…"):
        try:
            score = predict_proba(headline, model, tokenizer)
        except Exception as exc:
            st.error(f"Inference failed: `{type(exc).__name__}: {exc}`")
            return

    st.divider()
    st.subheader("Prediction")

    if score > SARCASM_THRESHOLD:
        st.error("### 🎭 **SARCASTIC**")
        st.caption(
            f"Model confidence: **{score:.1%}** sarcastic "
            f"(threshold {SARCASM_THRESHOLD:.0%})."
        )
        render_gauge("Sarcasm probability", score)
    else:
        st.success("### ✅ **REAL NEWS**")
        st.caption(
            f"Model confidence: **{(1 - score):.1%}** non-sarcastic "
            f"(raw sarcasm score: {score:.1%})."
        )
        render_gauge("Non-sarcasm confidence", 1.0 - score)

    with st.expander("Technical details"):
        st.write(f"- Raw output (P sarcastic): `{score:.6f}`")
        st.write(f"- Decision threshold: `{SARCASM_THRESHOLD}`")
        st.write(f"- Sequence length (padding): `{MAX_LEN}`")


if __name__ == "__main__":
    main()
