import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tokenizer import Tokenizer  # Assuming tokenizer.py is in the same directory
from model import create_model  # Assuming model.py is in the same directory

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

def preprocess_data(data, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences(data['text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences

def train_model(model, X_train, y_train, epochs=5, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def save_model_and_tokenizer(model, tokenizer, model_path, tokenizer_path):
    model.save(model_path)
    tokenizer.save(tokenizer_path)

def main():
    data_path = 'data/processed/sarcasm_data.csv'  # Update with actual path
    model_path = 'models/sarcasm_model.h5'  # Update with actual path
    tokenizer_path = 'models/tokenizer.pickle'  # Update with actual path
    max_length = 100  # Define max length for padding

    # Load and preprocess data
    data = load_data(data_path)
    X = preprocess_data(data, Tokenizer(), max_length)
    y = data['label'].values  # Assuming 'label' column exists

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train model
    model = create_model(input_shape=(max_length,))
    model = train_model(model, X_train, y_train)

    # Save the trained model and tokenizer
    save_model_and_tokenizer(model, Tokenizer(), model_path, tokenizer_path)

if __name__ == "__main__":
    main()