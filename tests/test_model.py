import unittest
import numpy as np
from src.models.model import create_model


class TestSarcasmDetectionModel(unittest.TestCase):

    def setUp(self):
        # create a small model for testing
        self.model = create_model(vocab_size=1000, embedding_dim=16, max_length=100)

    def test_model_output_shape(self):
        # use integer input (token ids) compatible with Embedding layer
        input_data = np.random.randint(0, 1000, size=(1, 100))
        output = self.model.predict(input_data)
        self.assertEqual(output.shape, (1, 1), "Output shape should be (1, 1) for binary classification.")

    def test_model_compile(self):
        # model should already be compiled by create_model
        self.assertIsNotNone(self.model.optimizer, "Model should be compiled with an optimizer.")


if __name__ == '__main__':
    unittest.main()