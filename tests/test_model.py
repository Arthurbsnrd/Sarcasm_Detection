import unittest
from src.models.model import create_model

class TestSarcasmDetectionModel(unittest.TestCase):

    def setUp(self):
        self.model = create_model()

    def test_model_output_shape(self):
        input_shape = (1, 100)  # Example input shape
        output = self.model.predict(np.random.rand(*input_shape))
        self.assertEqual(output.shape, (1, 1), "Output shape should be (1, 1) for binary classification.")

    def test_model_compile(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.assertIsNotNone(self.model.optimizer, "Model should be compiled with an optimizer.")

if __name__ == '__main__':
    unittest.main()