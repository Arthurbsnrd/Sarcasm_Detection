# Sarcasm Detection Application

This project is a sarcasm detection application built using Streamlit, TensorFlow, and Keras. The application aims to identify sarcastic comments in text data.

## Project Structure

```
sarcasm-detector
├── data
│   ├── raw
│   └── processed
├── src
│   ├── app.py
│   ├── config.py
│   ├── components
│   │   └── __init__.py
│   ├── models
│   │   ├── model.py
│   │   └── train.py
│   ├── preprocessing
│   │   └── tokenizer.py
│   └── utils
│       └── io.py
├── notebooks
│   └── exploration.ipynb
├── tests
│   └── test_model.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd sarcasm-detector
pip install -r requirements.txt
```

## Usage

To run the Streamlit application, execute the following command:

```bash
streamlit run src/app.py
```

Once the application is running, you can input text into the provided field and click the prediction button to see if the comment is sarcastic.

## Data

The raw dataset files should be placed in the `data/raw` directory. After preprocessing, the processed dataset files will be stored in the `data/processed` directory.

## Model

The model architecture is defined in `src/models/model.py`, and the training pipeline is implemented in `src/models/train.py`. The model is a Bidirectional LSTM designed to classify text as sarcastic or not.

## Preprocessing

Text data is preprocessed in `src/preprocessing/tokenizer.py`, which includes loading the dataset, cleaning the text, tokenizing, and padding sequences.

## Testing

Unit tests for the model can be found in `tests/test_model.py`. Ensure that the model functions as expected by running the tests.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.