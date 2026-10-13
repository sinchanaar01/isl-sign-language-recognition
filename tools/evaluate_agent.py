"""Evaluate offline agent behavior on hand-written gloss/sentence cases."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.isl_recognition.agent import Prediction
from src.isl_recognition.agent_loop import (
    AgentLoop,
    unsupported_content_words,
    vocabulary_from_file,
)

CASES: list[dict[str, Any]] = [
    {"glosses": ["HELLO"], "expected": "Hello!"},
    {"glosses": ["THANK YOU"], "expected": "Thank you."},
    {"glosses": ["PLEASE"], "expected": "Please."},
    {"glosses": ["SORRY"], "expected": "Sorry.", "low_confidence": True},
    {"glosses": ["YES"], "expected": "Yes."},
    {"glosses": ["NO"], "expected": "No."},
    {"glosses": ["HELP"], "expected": "Help."},
    {"glosses": ["WATER"], "expected": "Water."},
    {"glosses": ["FOOD"], "expected": "Food."},
    {"glosses": ["HOME"], "expected": "Home."},
    {"glosses": ["FAMILY"], "expected": "Family."},
    {"glosses": ["FRIEND"], "expected": "Friend."},
    {"glosses": ["LOVE"], "expected": "Love."},
    {"glosses": ["GOOD"], "expected": "Good."},
    {"glosses": ["BAD"], "expected": "Bad.", "low_confidence": True},
    {"glosses": ["STOP"], "expected": "Stop.", "low_confidence": True},
    {"glosses": ["WAIT"], "expected": "Wait.", "low_confidence": True},
    {"glosses": ["NAME"], "expected": "Name."},
    {"glosses": ["WHAT"], "expected": "What."},
    {"glosses": ["WHERE"], "expected": "Where.", "low_confidence": True},
    {"glosses": ["I", "WANT", "WATER"], "expected": "I want water."},
    {"glosses": ["I", "WANT", "FOOD"], "expected": "I want food."},
    {"glosses": ["I", "NEED", "HELP"], "expected": "I need help."},
    {"glosses": ["PLEASE", "HELP", "ME"], "expected": "Please help me."},
    {"glosses": ["WHAT", "YOUR", "NAME"], "expected": "What is your name?"},
    {"glosses": ["WHERE", "YOU", "LIVE"], "expected": "Where do you live?"},
    {"glosses": ["THANK", "YOU"], "expected": "Thank you."},
    {"glosses": ["GOOD", "MORNING"], "expected": "Good morning."},
    {"glosses": ["HELP", "WATER"], "expected": "Help water."},
    {"glosses": ["YES", "NO"], "expected": "Yes no."},
]


def evaluate(
    trace_directory: Path = Path("reports/agent_traces"),
    output_directory: Path = Path("reports/runs"),
) -> dict[str, Any]:
    vocabulary = vocabulary_from_file("data/target_glosses.txt")
    case_results = []
    repeats_true_positive = repeats_false_positive = repeats_false_negative = 0
    hallucination_cases = generated_cases = exact_matches = sentence_cases = 0
    for index, case in enumerate(CASES):
        expected_repeat = bool(case.get("low_confidence", False))
        confidence = 0.2 if expected_repeat else 0.95
        result = AgentLoop(
            [Prediction(gloss, confidence) for gloss in case["glosses"]],
            vocabulary,
            trace_directory=trace_directory,
        ).run()
        predicted_repeat = result.needs_repeat
        if predicted_repeat and expected_repeat:
            repeats_true_positive += 1
        elif predicted_repeat:
            repeats_false_positive += 1
        elif expected_repeat:
            repeats_false_negative += 1
        unsupported = (
            sorted(unsupported_content_words(result.text, case["glosses"]))
            if not predicted_repeat
            else []
        )
        if not predicted_repeat:
            generated_cases += 1
            sentence_cases += 1
            hallucination_cases += bool(unsupported)
            exact_matches += result.text == case["expected"]
        case_results.append(
            {
                "case_id": index + 1,
                "glosses": case["glosses"],
                "expected_sentence": case["expected"],
                "repeat_expected": expected_repeat,
                "repeat_predicted": predicted_repeat,
                "sentence": result.text,
                "exact_sentence_match": (
                    None if predicted_repeat else result.text == case["expected"]
                ),
                "unsupported_content_words": unsupported,
                "trace_path": result.trace_path,
            }
        )
    repeat_precision_denominator = repeats_true_positive + repeats_false_positive
    report = {
        "kind": "handwritten_offline_agent_evaluation_not_sign_recognition_metric",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_worktree_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "case_count": len(CASES),
        "llm_provider": None,
        "sentence_cases": sentence_cases,
        "exact_sentence_matches": exact_matches,
        "exact_sentence_match_rate": (
            exact_matches / sentence_cases if sentence_cases else None
        ),
        "hallucination_cases": hallucination_cases,
        "hallucination_rate": (
            hallucination_cases / generated_cases if generated_cases else None
        ),
        "repeat_true_positive": repeats_true_positive,
        "repeat_false_positive": repeats_false_positive,
        "repeat_false_negative": repeats_false_negative,
        "repeat_request_precision": (
            repeats_true_positive / repeat_precision_denominator
            if repeat_precision_denominator
            else None
        ),
        "cases": case_results,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = output_directory / f"agent_eval_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"}, indent=2
        )
    )
    print(f"Detailed run: {report_path}")
    return report


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
