def load_model(model_path):
    from tensorflow.keras.models import load_model
    return load_model(model_path)

def save_model(model, model_path):
    model.save(model_path)

def load_data(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def save_data(data, file_path):
    data.to_csv(file_path, index=False)