from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

class SarcasmDetectorModel:
    def __init__(self, vocab_size, embedding_dim, max_length):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_length))
        model.add(Bidirectional(LSTM(32)))
        model.add(Dense(24, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                                 epochs=epochs, batch_size=batch_size, 
                                 callbacks=[early_stopping])
        return history

    def predict(self, X):
        padded_sequences = pad_sequences(X, maxlen=self.max_length, padding='post')
        return self.model.predict(padded_sequences)


def create_model(vocab_size: int = 10000, embedding_dim: int = 50, max_length: int = 100):
    """Utility function used by scripts and tests to get a compiled Keras model."""
    m = Sequential()
    m.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
    m.add(Bidirectional(LSTM(32)))
    m.add(Dense(24, activation='relu'))
    m.add(Dropout(0.3))
    m.add(Dense(1, activation='sigmoid'))
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m