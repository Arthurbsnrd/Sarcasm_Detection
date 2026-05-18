import os
import streamlit as st
import numpy as np
import pickle


st.set_page_config(page_title="Sarcasm Detector", layout="centered")
st.title("Sarcasm Detector")
st.write("Enter a news headline and get a sarcasm prediction.")

MODEL_PATH = os.path.join('models', 'sarcasm_model.h5')
TOKENIZER_PATH = os.path.join('models', 'tokenizer.pkl')
MAXLEN = 100

# Try to import TensorFlow lazily; if it fails, run a TF-free fallback
USE_TF = False
tf = None
load_model = None
try:
    import tensorflow as _tf
    from tensorflow.keras.models import load_model as _load_model
    tf = _tf
    load_model = _load_model
    USE_TF = True
except Exception:
    USE_TF = False


def simple_clean_text(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-z0-9\s]", ' ', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text


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


def pad_sequences_manual(seqs, maxlen=MAXLEN):
    padded = np.zeros((len(seqs), maxlen), dtype=np.int32)
    for i, s in enumerate(seqs):
        s = s[:maxlen]
        padded[i, :len(s)] = s
    return padded


def load_model_and_tokenizer():
    model = None
    tokenizer = None

    # Load model only if TF is available
    if USE_TF and load_model is not None and os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
        except Exception:
            model = None

    # Load tokenizer only if file exists
    if os.path.exists(TOKENIZER_PATH):
        try:
            with open(TOKENIZER_PATH, 'rb') as f:
                tokenizer = pickle.load(f)
        except Exception:
            tokenizer = None

    # Fallbacks
    if tokenizer is None:
        tokenizer = SimpleTokenizer()

    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()


def predict_sarcasm(text: str):
    clean = simple_clean_text(text)

    # ensure tokenizer has vocab for the input
    if not getattr(tokenizer, 'word_index', None) or len(tokenizer.word_index) == 0:
        try:
            tokenizer.fit_on_texts([clean])
        except Exception:
            pass

    seq = tokenizer.texts_to_sequences([clean])

    if USE_TF and model is not None:
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences as _pad
            padded = _pad(seq, maxlen=MAXLEN, padding='post')
            prob = float(model.predict(padded)[0][0])
        except Exception:
            # fall back to manual padding and dummy score
            padded = pad_sequences_manual(seq, maxlen=MAXLEN)
            prob = 0.5
    else:
        # simple heuristic fallback: punctuation and sarcasm keywords
        score = 0.5
        if '!' in text or '...' in text:
            score += 0.15
        lower = text.lower()
        for kw in ('yeah right', 'as if', 'sure', 'totally', 'obviously', 'amazing'):
            if kw in lower:
                score += 0.2
        prob = min(max(score, 0.0), 1.0)

    label = 'Sarcastic' if prob >= 0.5 else 'Real'
    emoji = '😂' if prob >= 0.5 else '🙂'
    return label, prob, emoji


text = st.text_input('Headline', 'Scientists discover sleeping helps when tired')

if st.button('Predict'):
    if text.strip() == '':
        st.warning('Please enter a headline.')
    else:
        label, prob, emoji = predict_sarcasm(text)
        st.markdown(f"{emoji} **{label}** — {prob*100:.0f}%")