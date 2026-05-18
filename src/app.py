from flask import Flask, request, jsonify
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.preprocessing.tokenizer import Tokenizer

# Load the trained model and tokenizer
model = load_model('path/to/your/model.h5')
tokenizer = Tokenizer('path/to/your/tokenizer.pickle')

def predict_sarcasm(text):
    # Preprocess the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
    prediction = model.predict(padded_sequences)
    return prediction[0][0]

# Streamlit app
st.title("Sarcasm Detection App")
st.write("Enter a sentence to check if it's sarcastic or not.")

user_input = st.text_area("Input Text")

if st.button("Predict"):
    if user_input:
        prediction = predict_sarcasm(user_input)
        if prediction > 0.5:
            st.success("The input text is sarcastic!")
        else:
            st.success("The input text is not sarcastic.")
    else:
        st.warning("Please enter some text.")