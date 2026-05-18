from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import re

class TextPreprocessor:
    def __init__(self, max_words=10000, max_len=100):
        self.tokenizer = Tokenizer(num_words=max_words)
        self.max_len = max_len

    def clean_text(self, text):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        return text

    def fit(self, texts):
        cleaned_texts = [self.clean_text(text) for text in texts]
        self.tokenizer.fit_on_texts(cleaned_texts)

    def transform(self, texts):
        cleaned_texts = [self.clean_text(text) for text in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len)
        return padded_sequences

    def save_tokenizer(self, filepath):
        import pickle
        with open(filepath, 'wb') as file:
            pickle.dump(self.tokenizer, file)

    def load_tokenizer(self, filepath):
        import pickle
        with open(filepath, 'rb') as file:
            self.tokenizer = pickle.load(file)