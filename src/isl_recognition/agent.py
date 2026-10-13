"""Confidence-aware sign-to-sentence agent with optional OpenAI-compatible LLM."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


@dataclass(frozen=True)
class AgentResponse:
    text: str
    needs_repeat: bool
    used_llm: bool


class SentenceAgent:
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.confidence_threshold = confidence_threshold
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.endpoint = os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"
        )

    def generate(self, predictions: Iterable[Prediction]) -> AgentResponse:
        predictions = list(predictions)
        if not predictions:
            return AgentResponse("Start signing to build a sentence.", False, False)
        if any(
            not prediction.label.strip()
            or not 0 <= prediction.confidence <= 1
            for prediction in predictions
        ):
            raise ValueError("Predictions must have labels and confidence in [0, 1]")

        reliable = [
            prediction.label.strip()
            for prediction in predictions
            if prediction.confidence >= self.confidence_threshold
        ]
        if len(reliable) != len(predictions):
            return AgentResponse(
                "I wasn't sure about that sign. Please repeat it clearly.",
                True,
                False,
            )
        signs = self._deduplicate(reliable)
        if not signs:
            return AgentResponse("Start signing to build a sentence.", False, False)
        if not self.api_key:
            return AgentResponse(self._fallback_sentence(signs), False, False)
        return AgentResponse(self._generate_with_llm(signs), False, True)

    @staticmethod
    def _deduplicate(signs: list[str]) -> list[str]:
        result: list[str] = []
        for sign in signs:
            if not result or result[-1].casefold() != sign.casefold():
                result.append(sign)
        return result

    @staticmethod
    def _fallback_sentence(signs: list[str]) -> str:
        normalized = tuple(sign.casefold() for sign in signs)
        templates = {
            ("thank you",): "Thank you.",
            ("thank_you",): "Thank you.",
            ("hello",): "Hello!",
            ("please", "help", "me"): "Please help me.",
            ("i", "need", "help"): "I need help.",
            ("i", "want", "water"): "I want water.",
            ("i", "want", "food"): "I want food.",
            ("what", "your", "name"): "What is your name?",
            ("where", "you", "live"): "Where do you live?",
        }
        if normalized in templates:
            return templates[normalized]
        words = " ".join(signs)
        return words[0].upper() + words[1:] + "."

    def _generate_with_llm(self, signs: list[str]) -> str:
        body = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Turn the ordered recognized sign glosses into one short, "
                        "grammatical English sentence. Preserve their meaning. If "
                        "meaning is ambiguous, keep it literal; do not invent details. "
                        "Return only the sentence."
                    ),
                },
                {"role": "user", "content": "Recognized signs: " + " ".join(signs)},
            ],
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM sentence generation failed: {exc}") from exc
        try:
            sentence = payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise RuntimeError("LLM returned an unexpected response shape") from exc
        if not sentence:
            raise RuntimeError("LLM returned an empty sentence")
        return sentence
