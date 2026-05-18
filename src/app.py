import os
import streamlit as st
import numpy as np
import pickle
from typing import Optional


st.set_page_config(page_title="Sarcasm Detector", layout="wide")


### Sidebar
with st.sidebar:
    st.title("Sarcasm Detector")
    st.markdown("Detect whether a news headline is sarcastic or real using a BiLSTM model.")
    st.write("---")
    st.header("Model")
    st.write("You can upload a trained Keras model (.h5) and tokenizer (.pkl) to use real predictions.")
    uploaded_model = st.file_uploader("Upload model (.h5)", type=['h5'])
    uploaded_tokenizer = st.file_uploader("Upload tokenizer (.pkl)", type=['pkl'])
    st.write("---")
    st.header("Settings")
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    examples = st.multiselect("Quick examples", [
        "Scientists discover sleeping helps when tired",
        "Yeah right, that was totally necessary...",
        "Local man buys new phone, life changed forever",
        "Oh great, another Monday"
    ], default=["Scientists discover sleeping helps when tired"])
    st.write("---")
    st.markdown("**About**\n\nBuilt with Streamlit. Fallback heuristic used if no model is loaded.")


### Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input")
    headline = st.text_area("Enter a news headline", value=examples[0] if examples else "", height=120)
    st.markdown("### Previews")
    for ex in examples:
        if st.button(f"Use: {ex}", key=f"example_{ex}"):
            headline = ex

    predict_btn = st.button("Predict", type='primary')

with col2:
    st.subheader("Model Status")
    model_status = "No model loaded"
    tokenizer_status = "No tokenizer loaded"

    MODEL_DIR = 'models'
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path: Optional[str] = None
    tokenizer_obj = None

    # Save uploaded files to models/ for reuse
    if uploaded_model is not None:
        model_path = os.path.join(MODEL_DIR, 'uploaded_model.h5')
        with open(model_path, 'wb') as f:
            f.write(uploaded_model.getbuffer())
        model_status = f'Uploaded: {os.path.basename(model_path)}'

    if uploaded_tokenizer is not None:
        tokenizer_path = os.path.join(MODEL_DIR, 'uploaded_tokenizer.pkl')
        with open(tokenizer_path, 'wb') as f:
            f.write(uploaded_tokenizer.getbuffer())
        tokenizer_status = f'Uploaded: {os.path.basename(tokenizer_path)}'

    # Show existing files if present
    if os.path.exists(os.path.join(MODEL_DIR, 'sarcasm_model.h5')):
        model_status = 'Found: sarcasm_model.h5'
        model_path = os.path.join(MODEL_DIR, 'sarcasm_model.h5')
    if os.path.exists(os.path.join(MODEL_DIR, 'tokenizer.pkl')):
        tokenizer_status = 'Found: tokenizer.pkl'
        tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.pkl')

    st.write(model_status)
    st.write(tokenizer_status)

    st.write("---")
    st.markdown("**Confidence**")
    conf_placeholder = st.empty()
    progress_placeholder = st.empty()


### Utilities: lightweight tokenizer + heuristic predictor
def clean_text(text: str) -> str:
    import re
    t = text.lower()
    t = re.sub(r"http\S+|www\S+|https\S+", "", t)
    t = re.sub(r"[^a-z0-9\s]", ' ', t)
    t = re.sub(r"\s+", ' ', t).strip()
    return t


class SimpleTokenizer:
    def __init__(self):
        self.word_index = {}

    def fit_on_texts(self, texts):
        idx = 1
        for t in texts:
            for w in t.split():
                if w not in self.word_index:
                    self.word_index[w] = idx
                    idx += 1

    def texts_to_sequences(self, texts):
        seqs = []
        for t in texts:
            seqs.append([self.word_index.get(w, 0) for w in t.split()])
        return seqs


def pad_sequences_manual(seqs, maxlen=100):
    import numpy as _np
    padded = _np.zeros((len(seqs), maxlen), dtype=_np.int32)
    for i, s in enumerate(seqs):
        s = s[:maxlen]
        padded[i, :len(s)] = s
    return padded


def load_keras_model(path: str):
    try:
        import tensorflow as _tf
        from tensorflow.keras.models import load_model as _load
        return _load(path)
    except Exception:
        return None


def load_tokenizer_obj(path: str):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


# Prepare defaults
model_obj = None
tokenizer_obj = None
if model_path:
    model_obj = load_keras_model(model_path)
if 'tokenizer_path' in locals():
    tokenizer_obj = load_tokenizer_obj(tokenizer_path)
if tokenizer_obj is None:
    tokenizer_obj = SimpleTokenizer()


def predict_with_model(text: str):
    clean = clean_text(text)
    try:
        # ensure tokenizer has vocab
        if not getattr(tokenizer_obj, 'word_index', None):
            tokenizer_obj.fit_on_texts([clean])
        seq = tokenizer_obj.texts_to_sequences([clean])
        if model_obj is not None:
            from tensorflow.keras.preprocessing.sequence import pad_sequences as _pad
            padded = _pad(seq, maxlen=100, padding='post')
            prob = float(model_obj.predict(padded)[0][0])
        else:
            padded = pad_sequences_manual(seq, maxlen=100)
            prob = 0.5
    except Exception:
        prob = 0.5
    return prob


def heuristic_predict(text: str):
    score = 0.5
    if '!' in text or '...' in text:
        score += 0.12
    lower = text.lower()
    for kw in ('yeah right', 'as if', 'sure', 'totally', 'obviously', 'amazing'):
        if kw in lower:
            score += 0.2
    return min(max(score, 0.0), 1.0)


### Prediction flow
if predict_btn:
    if not headline or headline.strip() == '':
        st.warning('Please enter a headline to predict.')
    else:
        # Prefer model if available
        prob = None
        if model_obj is not None:
            prob = predict_with_model(headline)
        else:
            prob = heuristic_predict(headline)

        label = 'Sarcastic' if prob >= threshold else 'Real'
        emoji = '😂' if prob >= threshold else '🙂'

        conf_placeholder.metric(label='Prediction', value=f"{label} {emoji}", delta=f"{prob*100:.1f}%")
        progress_placeholder.progress(int(prob*100))

        st.markdown("---")
        st.subheader('Details')
        st.write(f'Confidence: {prob*100:.2f}%')
        st.write(f'Model used: {os.path.basename(model_path) if model_path else "heuristic fallback"}')
        with st.expander('Preprocessing'):
            st.write('Cleaned input:')
            st.code(clean_text(headline))
        with st.expander('Notes'):
            st.markdown('- This demo uses a simple BiLSTM architecture when a model is provided.')
            st.markdown('- If you want a trained model, run the training pipeline and upload the resulting `.h5` and `.pkl` files.')

