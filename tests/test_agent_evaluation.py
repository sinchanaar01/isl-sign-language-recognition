import unittest

from tools.evaluate_agent import CASES


class HandwrittenAgentEvaluationTests(unittest.TestCase):
    def test_evaluation_has_thirty_handwritten_cases(self):
        # Fixed hand-authored agent smoke cases; not training data or sign metrics.
        self.assertEqual(len(CASES), 30)
        self.assertTrue(all(case["glosses"] and case["expected"] for case in CASES))
        self.assertEqual(sum(bool(case.get("low_confidence")) for case in CASES), 5)


if __name__ == "__main__":
    unittest.main()
