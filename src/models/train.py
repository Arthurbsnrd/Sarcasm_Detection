import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing.tokenizer import TextPreprocessor
from src.models.model import create_model


def load_json_dataset(path):
    """Load JSON lines dataset (one JSON object per line) or CSV if provided."""
    if path.endswith('.csv'):
        return pd.read_csv(path)
    # assume json lines
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(pd.read_json(line, typ='series'))
            except Exception:
                # fallback: skip malformed lines
                continue
    if records:
        return pd.DataFrame(records)
    return pd.DataFrame()


def preprocess_data(df, preprocessor: TextPreprocessor, text_column: str = 'headline'):
    texts = df[text_column].astype(str).tolist()
    preprocessor.fit(texts)
    X = preprocessor.transform(texts)
    return X


def train_model_pipeline(data_path='data/Sarcasm_Headlines_Dataset.json',
                         model_path='models/sarcasm_model.h5',
                         tokenizer_path='models/tokenizer.pkl',
                         max_length=100,
                         epochs=3,
                         batch_size=128):
    df = load_json_dataset(data_path)
    if df.empty:
        raise FileNotFoundError(f"No data loaded from {data_path}")

    # Expecting 'headline' and 'is_sarcastic' columns in the dataset
    if 'headline' not in df.columns or 'is_sarcastic' not in df.columns:
        raise ValueError('Dataset must contain columns "headline" and "is_sarcastic"')

    pre = TextPreprocessor(max_len=max_length)
    X = preprocess_data(df, pre, text_column='headline')
    y = df['is_sarcastic'].astype(int).values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_model(vocab_size=10000, embedding_dim=50, max_length=max_length)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    # save tokenizer
    import pickle
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(pre.tokenizer, f)


if __name__ == '__main__':
    # default run (small epochs) - user can customize
    try:
        train_model_pipeline()
    except Exception as e:
        print('Training pipeline failed:', e)