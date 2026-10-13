"""Validate the separately stored second agent-evaluation fixture."""

from __future__ import annotations

import unittest
from pathlib import Path

from tools.evaluate_agent_independent import load_cases


class IndependentAgentEvaluationTests(unittest.TestCase):
    def test_fixture_is_separate_and_contains_noisy_ambiguous_cases(self):
        fixture = Path("data/evaluation/agent_independent_cases.json")
        payload, raw = load_cases(fixture)
        cases = payload["cases"]

        self.assertGreaterEqual(len(cases), 30)
        self.assertGreater(sum(bool(case["ambiguous"]) for case in cases), 0)
        self.assertGreater(sum(bool(case["noisy"]) for case in cases), 0)
        self.assertIn(b"separately_curated", raw)


if __name__ == "__main__":
    unittest.main()
