"""Evaluate the agent on the separate ambiguity/noise case file."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.isl_recognition.agent import Prediction
from src.isl_recognition.agent_loop import (
    AgentLoop,
    provider_from_environment,
    unsupported_content_words,
    vocabulary_from_file,
)


def load_cases(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read and validate the independent, text-only evaluation fixture."""
    raw = path.read_bytes()
    payload = json.loads(raw)
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) < 30:
        raise ValueError("Independent evaluation must contain at least 30 cases.")
    identifiers: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each independent case must be a JSON object.")
        case_id = str(case.get("id", ""))
        glosses = case.get("glosses")
        confidences = case.get("confidences")
        if (
            not case_id
            or case_id in identifiers
            or not isinstance(glosses, list)
            or not glosses
            or not isinstance(confidences, list)
            or len(confidences) != len(glosses)
            or any(not isinstance(gloss, str) or not gloss.strip() for gloss in glosses)
            or any(
                not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1
                for confidence in confidences
            )
        ):
            raise ValueError(f"Malformed independent evaluation case: {case_id!r}")
        identifiers.add(case_id)
    return payload, raw


def evaluate(
    cases_path: Path = Path("data/evaluation/agent_independent_cases.json"),
    vocabulary_path: Path = Path("data/target_glosses.txt"),
    trace_directory: Path = Path("reports/agent_traces"),
    output_directory: Path = Path("reports/runs"),
) -> dict[str, Any]:
    """Run the configured provider if credentials exist, otherwise offline mode."""
    payload, raw_cases = load_cases(cases_path)
    provider = provider_from_environment()
    vocabulary = vocabulary_from_file(vocabulary_path)
    case_results: list[dict[str, Any]] = []
    repeat_tp = repeat_fp = repeat_fn = 0
    generated_count = hallucination_count = 0

    for case in payload["cases"]:
        predictions = [
            Prediction(gloss, float(confidence))
            for gloss, confidence in zip(
                case["glosses"], case["confidences"], strict=True
            )
        ]
        result = AgentLoop(
            predictions,
            vocabulary,
            provider=provider,
            trace_directory=trace_directory,
        ).run()
        expected_repeat = bool(case["should_repeat"])
        if result.needs_repeat and expected_repeat:
            repeat_tp += 1
        elif result.needs_repeat:
            repeat_fp += 1
        elif expected_repeat:
            repeat_fn += 1
        unsupported = (
            sorted(unsupported_content_words(result.text, case["glosses"]))
            if not result.needs_repeat
            else []
        )
        if not result.needs_repeat:
            generated_count += 1
            hallucination_count += bool(unsupported) or not result.verified
        case_results.append(
            {
                "id": case["id"],
                "ambiguous": bool(case.get("ambiguous")),
                "noisy": bool(case.get("noisy")),
                "expected_repeat": expected_repeat,
                "predicted_repeat": result.needs_repeat,
                "glosses": case["glosses"],
                "confidences": case["confidences"],
                "text": result.text,
                "verified": result.verified,
                "unsupported_content_words": unsupported,
                "trace_path": result.trace_path,
            }
        )

    predicted_repeat_denominator = repeat_tp + repeat_fp
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report: dict[str, Any] = {
        "kind": "separate_independent_agent_evaluation_not_sign_recognition",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_set": str(cases_path),
        "evaluation_set_sha256": hashlib.sha256(raw_cases).hexdigest(),
        "source_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or None,
        "source_worktree_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        ),
        "case_count": len(case_results),
        "ambiguous_case_count": sum(item["ambiguous"] for item in case_results),
        "noisy_case_count": sum(item["noisy"] for item in case_results),
        "provider": provider.name if provider else "deterministic-offline",
        "live_llm_provider_used": provider is not None,
        "generated_sentence_count": generated_count,
        "hallucination_cases": hallucination_count,
        "hallucination_rate": (
            hallucination_count / generated_count if generated_count else None
        ),
        "repeat_true_positive": repeat_tp,
        "repeat_false_positive": repeat_fp,
        "repeat_false_negative": repeat_fn,
        "repeat_request_precision": (
            repeat_tp / predicted_repeat_denominator
            if predicted_repeat_denominator
            else None
        ),
        "repeat_request_recall": (
            repeat_tp / (repeat_tp + repeat_fn) if repeat_tp + repeat_fn else None
        ),
        "hallucination_measurement": (
            "A generated case is flagged if the rule-based content-token checker "
            "finds unsupported words or the configured verification rejects it. "
            "This is not a human semantic-faithfulness judgment."
        ),
        "cases": case_results,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / f"agent_eval_independent_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"},
            indent=2,
        )
    )
    print(f"Detailed independent evaluation: {report_path}")
    return report


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
