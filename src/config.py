# Configuration settings for the sarcasm detection application

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Hyperparameters
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
LSTM_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 10

# Model file names
MODEL_FILE_NAME = 'sarcasm_model.h5'
TOKENIZER_FILE_NAME = 'tokenizer.pickle'