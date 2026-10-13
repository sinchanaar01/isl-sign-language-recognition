import unittest

from src.isl_recognition.inference import PredictionSmoother


class PredictionSmootherTests(unittest.TestCase):
    def test_averages_recent_predictions(self):
        smoother = PredictionSmoother(window_size=2)
        self.assertEqual(
            smoother.update({"HELLO": 0.8, "YES": 0.2}),
            ("HELLO", 0.8),
        )
        label, confidence = smoother.update({"HELLO": 0.1, "YES": 0.9})
        self.assertEqual(label, "YES")
        self.assertAlmostEqual(confidence, 0.55)

    def test_rejects_empty_and_invalid_confidences(self):
        smoother = PredictionSmoother()
        with self.assertRaises(ValueError):
            smoother.update({})
        with self.assertRaises(ValueError):
            smoother.update({"HELLO": 1.1})


if __name__ == "__main__":
    unittest.main()
