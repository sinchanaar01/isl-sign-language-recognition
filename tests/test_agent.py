import io
import json
import unittest
from unittest.mock import patch

from src.isl_recognition.agent import Prediction, SentenceAgent


class SentenceAgentTests(unittest.TestCase):
    def test_empty_predictions_prompt_user_to_sign(self):
        response = SentenceAgent().generate([])
        self.assertFalse(response.needs_repeat)
        self.assertIn("Start signing", response.text)

    def test_low_confidence_requests_repeat(self):
        response = SentenceAgent(confidence_threshold=0.6).generate(
            [Prediction("WATER", 0.59)]
        )
        self.assertTrue(response.needs_repeat)
        self.assertIn("repeat", response.text)

    def test_fallback_suppresses_consecutive_duplicate(self):
        response = SentenceAgent(api_key="").generate(
            [Prediction("I", 0.8), Prediction("I", 0.9), Prediction("want", 0.9),
             Prediction("water", 0.9)]
        )
        self.assertFalse(response.used_llm)
        self.assertEqual(response.text, "I want water.")

    def test_invalid_confidence_is_rejected(self):
        with self.assertRaises(ValueError):
            SentenceAgent().generate([Prediction("HELP", 1.1)])

    def test_llm_response_is_parsed(self):
        response_body = {
            "choices": [{"message": {"content": "Please bring me water."}}]
        }
        fake_response = io.BytesIO(json.dumps(response_body).encode())
        with patch("urllib.request.urlopen", return_value=fake_response):
            response = SentenceAgent(api_key="test-only").generate(
                [Prediction("PLEASE", 0.9), Prediction("WATER", 0.9)]
            )
        self.assertTrue(response.used_llm)
        self.assertEqual(response.text, "Please bring me water.")


if __name__ == "__main__":
    unittest.main()
