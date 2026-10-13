"""Bounded tool-calling sentence agent with OpenAI/Anthropic adapters."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .agent import Prediction, SentenceAgent


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass(frozen=True)
class AgentTurn:
    tool_calls: list[ToolCall]
    final_text: str | None = None


@dataclass(frozen=True)
class AgentLoopResult:
    text: str
    needs_repeat: bool
    verified: bool
    used_llm: bool
    steps: int
    trace_path: str
    repeat_indices: tuple[int, ...] = ()


class ToolCallingProvider(Protocol):
    """Minimal provider interface to make the loop testable without network calls."""

    name: str

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AgentTurn: ...

    def draft_sentence(self, glosses: Sequence[str]) -> str: ...

    def judge_sentence(self, sentence: str, glosses: Sequence[str]) -> bool: ...


def _request_json(
    endpoint: str, payload: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Agent provider request failed: {exc}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("Agent provider returned a non-object JSON response")
    return decoded


class OpenAICompatibleProvider:
    """OpenAI Chat Completions-compatible provider."""

    name = "openai-compatible"

    def __init__(
        self,
        api_key: str,
        model: str,
        endpoint: str = "https://api.openai.com/v1/chat/completions",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AgentTurn:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": messages,
            "tools": [
                {"type": "function", "function": tool} for tool in tools
            ],
            "tool_choice": "auto",
        }
        result = _request_json(
            self.endpoint,
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            message = result["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenAI-compatible response has no assistant message") from exc
        calls = []
        for call in message.get("tool_calls", []):
            try:
                calls.append(ToolCall(
                    name=call["function"]["name"],
                    arguments=json.loads(call["function"]["arguments"]),
                    call_id=call["id"],
                ))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError("Malformed OpenAI-compatible tool call") from exc
        return AgentTurn(calls, message.get("content"))

    def draft_sentence(self, glosses: Sequence[str]) -> str:
        result = _request_json(
            self.endpoint,
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the ordered sign glosses as one concise, grammatical "
                            "English sentence. Preserve meaning; do not add facts. Return "
                            "only the sentence."
                        ),
                    },
                    {"role": "user", "content": " ".join(glosses)},
                ],
            },
            {"Authorization": f"Bearer {self.api_key}"},
        )
        return _openai_text(result)

    def judge_sentence(self, sentence: str, glosses: Sequence[str]) -> bool:
        result = _request_json(
            self.endpoint,
            {
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Judge whether the sentence adds any concrete content or "
                            "claim not supported by these sign glosses. Grammar words "
                            "may be added. Return JSON only: "
                            '{"supported": true} or {"supported": false}.'
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps({
                            "glosses": list(glosses), "sentence": sentence
                        }),
                    },
                ],
            },
            {"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            verdict = json.loads(result["choices"][0]["message"]["content"])
            supported = verdict["supported"]
            if not isinstance(supported, bool):
                raise ValueError("supported must be a JSON boolean")
            return supported
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise RuntimeError("OpenAI-compatible judge response is malformed") from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError("OpenAI-compatible judge returned invalid JSON") from exc


def _openai_text(result: dict[str, Any]) -> str:
    try:
        text = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenAI-compatible response has no text content") from exc
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("OpenAI-compatible provider returned empty text")
    return text.strip()


class AnthropicProvider:
    """Anthropic Messages API adapter using native tool-use blocks."""

    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        endpoint: str = "https://api.anthropic.com/v1/messages",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    @staticmethod
    def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            if message["role"] == "tool":
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    "content": message["content"],
                }
                if converted and converted[-1]["role"] == "user" and isinstance(
                    converted[-1]["content"], list
                ):
                    converted[-1]["content"].append(result_block)
                else:
                    converted.append({"role": "user", "content": [result_block]})
            elif message["role"] == "assistant" and message.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                for call in message["tool_calls"]:
                    blocks.append({
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "input": json.loads(call["function"]["arguments"]),
                    })
                converted.append({"role": "assistant", "content": blocks})
            else:
                converted.append({
                    "role": message["role"],
                    "content": message.get("content", ""),
                })
        return converted

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AgentTurn:
        result = _request_json(
            self.endpoint,
            {
                "model": self.model,
                "max_tokens": 1024,
                "system": (
                    "Use tools to inspect the sign buffer, check vocabulary, propose "
                    "one faithful sentence, verify it, then speak only after approval. "
                    "Never add unsupported content."
                ),
                "messages": self._messages(messages),
                "tools": [
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "input_schema": tool["parameters"],
                    }
                    for tool in tools
                ],
            },
            self._headers(),
        )
        blocks = result.get("content")
        if not isinstance(blocks, list):
            raise RuntimeError("Anthropic response has no content blocks")
        calls: list[ToolCall] = []
        final_text = []
        for block in blocks:
            if block.get("type") == "tool_use":
                calls.append(ToolCall(
                    name=block["name"],
                    arguments=block["input"],
                    call_id=block["id"],
                ))
            elif block.get("type") == "text":
                final_text.append(block.get("text", ""))
        text = "\n".join(value for value in final_text if value).strip()
        return AgentTurn(calls, text or None)

    def draft_sentence(self, glosses: Sequence[str]) -> str:
        result = _request_json(
            self.endpoint,
            {
                "model": self.model,
                "max_tokens": 256,
                "system": (
                    "Write one concise grammatical English sentence from the ordered "
                    "sign glosses. Preserve meaning; invent no facts. Return only it."
                ),
                "messages": [{"role": "user", "content": " ".join(glosses)}],
            },
            self._headers(),
        )
        return _anthropic_text(result)

    def judge_sentence(self, sentence: str, glosses: Sequence[str]) -> bool:
        result = _request_json(
            self.endpoint,
            {
                "model": self.model,
                "max_tokens": 64,
                "system": (
                    'Return only {"supported":true} or {"supported":false}. '
                    "Grammar words may be added; any unsupported content word means false."
                ),
                "messages": [{
                    "role": "user",
                    "content": json.dumps({
                        "glosses": list(glosses), "sentence": sentence
                    }),
                }],
            },
            self._headers(),
        )
        text = _anthropic_text(result).replace(" ", "").casefold()
        if text not in {'{"supported":true}', '{"supported":false}'}:
            raise RuntimeError("Anthropic sentence judge returned malformed JSON")
        return text == '{"supported":true}'


def _anthropic_text(result: dict[str, Any]) -> str:
    blocks = result.get("content")
    if not isinstance(blocks, list):
        raise RuntimeError("Anthropic response has no content blocks")
    text = " ".join(
        block.get("text", "") for block in blocks if block.get("type") == "text"
    ).strip()
    if not text:
        raise RuntimeError("Anthropic provider returned empty text")
    return text


def provider_from_environment() -> ToolCallingProvider | None:
    provider_name = os.getenv("LLM_PROVIDER", "openai").casefold()
    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return OpenAICompatibleProvider(
            api_key,
            os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            os.getenv(
                "OPENAI_BASE_URL",
                "https://api.openai.com/v1/chat/completions",
            ),
        )
    if provider_name == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        return AnthropicProvider(
            api_key,
            os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
            os.getenv(
                "ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/messages"
            ),
        )
    raise ValueError("LLM_PROVIDER must be 'openai' or 'anthropic'")


def vocabulary_from_file(path: str | Path) -> dict[str, str]:
    """Load candidate headwords as unverified entries, not attested ISL signs."""
    words = Path(path).read_text(encoding="utf-8").splitlines()
    return {
        line.strip().casefold(): "candidate headword; sign requires fluent-signer verification"
        for line in words
        if line.strip() and not line.lstrip().startswith("#")
    }


_FUNCTION_WORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
    "being", "do", "does", "did", "to", "from", "in", "on", "at", "for",
    "of", "with", "and", "or", "but", "i", "you", "we", "they", "he",
    "she", "it", "me", "us", "them", "my", "your", "our", "their",
}


def content_words(text: str) -> set[str]:
    words = {
        match.casefold()
        for match in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    }
    return words - _FUNCTION_WORDS


def unsupported_content_words(sentence: str, glosses: Sequence[str]) -> set[str]:
    source = content_words(" ".join(glosses))
    proposed = content_words(sentence)
    allowed_variants = set(source)
    for word in source:
        if len(word) > 3:
            if word.endswith("s"):
                allowed_variants.add(word[:-1])
            if word.endswith("ed"):
                allowed_variants.add(word[:-2])
            if word.endswith("ing"):
                allowed_variants.add(word[:-3])
    return proposed - allowed_variants


TOOL_SCHEMAS = [
    {
        "name": "get_sign_buffer",
        "description": "Return recent predicted signs and confidence scores.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "request_repeat",
        "description": "Ask the signer to repeat one low-confidence sign by zero-based index.",
        "parameters": {
            "type": "object",
            "properties": {"sign_index": {"type": "integer", "minimum": 0}},
            "required": ["sign_index"],
        },
    },
    {
        "name": "lookup_vocabulary",
        "description": "Check a recognized gloss against the reviewed vocabulary list.",
        "parameters": {
            "type": "object",
            "properties": {"word": {"type": "string"}},
            "required": ["word"],
        },
    },
    {
        "name": "propose_sentence",
        "description": "Draft a grammatical English sentence from ordered sign glosses.",
        "parameters": {
            "type": "object",
            "properties": {"glosses": {"type": "array", "items": {"type": "string"}}},
            "required": ["glosses"],
        },
    },
    {
        "name": "verify_sentence",
        "description": "Reject unsupported content words before any sentence is spoken.",
        "parameters": {
            "type": "object",
            "properties": {
                "sentence": {"type": "string"},
                "glosses": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["sentence", "glosses"],
        },
    },
    {
        "name": "speak",
        "description": "Queue a verified sentence for speech synthesis.",
        "parameters": {
            "type": "object",
            "properties": {"sentence": {"type": "string"}},
            "required": ["sentence"],
        },
    },
]


class AgentLoop:
    """Execute a bounded tool loop and append each tool/decision to a JSONL trace."""

    def __init__(
        self,
        sign_buffer: Sequence[Prediction],
        vocabulary: Mapping[str, str],
        provider: ToolCallingProvider | None = None,
        confidence_threshold: float = 0.6,
        max_steps: int = 5,
        trace_directory: str | Path = "reports/agent_traces",
        request_repeat_callback: Callable[[int], None] | None = None,
        speak_callback: Callable[[str], None] | None = None,
    ) -> None:
        if not 1 <= max_steps <= 5:
            raise ValueError("max_steps must be between one and five")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between zero and one")
        if any(not item.label.strip() or not 0 <= item.confidence <= 1
               for item in sign_buffer):
            raise ValueError("Sign buffer has an empty label or invalid confidence")
        self.sign_buffer = list(sign_buffer)
        self.vocabulary = {key.casefold(): value for key, value in vocabulary.items()}
        self.provider = provider
        self.confidence_threshold = confidence_threshold
        self.max_steps = max_steps
        self.request_repeat_callback = request_repeat_callback
        self.speak_callback = speak_callback
        self.trace_directory = Path(trace_directory)
        self.trace_path = self.trace_directory / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "_" + uuid.uuid4().hex + ".jsonl"
        )
        self.steps = 0
        self.repeat_indices: list[int] = []
        self.verified_sentence: str | None = None
        self.speech_requested = False
        self._fallback = SentenceAgent(api_key="")

    def _trace(self, event: str, **details: Any) -> None:
        self.trace_directory.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "step": self.steps,
            **details,
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _charge_step(self) -> bool:
        if self.steps >= self.max_steps:
            return False
        self.steps += 1
        return True

    def get_sign_buffer(self) -> list[dict[str, Any]]:
        return [asdict(prediction) for prediction in self.sign_buffer]

    def request_repeat(self, sign_index: int) -> dict[str, Any]:
        if not 0 <= sign_index < len(self.sign_buffer):
            raise ValueError(f"sign_index {sign_index} is outside the current buffer")
        prediction = self.sign_buffer[sign_index]
        if prediction.confidence >= self.confidence_threshold:
            return {
                "requested": False,
                "reason": "sign confidence is not below the repeat threshold",
                "sign_index": sign_index,
                "label": prediction.label,
            }
        if sign_index not in self.repeat_indices:
            self.repeat_indices.append(sign_index)
            if self.request_repeat_callback:
                self.request_repeat_callback(sign_index)
        return {
            "requested": True,
            "reason": "Please repeat this sign clearly.",
            "sign_index": sign_index,
            "label": prediction.label,
        }

    def lookup_vocabulary(self, word: str) -> dict[str, Any]:
        category = self.vocabulary.get(word.strip().casefold())
        return {
            "word": word,
            "known": category is not None,
            "category": category,
        }

    def verify_sentence(
        self, sentence: str, glosses: Sequence[str]
    ) -> dict[str, Any]:
        unsupported = sorted(unsupported_content_words(sentence, glosses))
        if unsupported:
            self.verified_sentence = None
            return {
                "supported": False,
                "reason": "unsupported content words",
                "unsupported_words": unsupported,
            }
        if self.provider and not self.provider.judge_sentence(sentence, glosses):
            self.verified_sentence = None
            return {
                "supported": False,
                "reason": "LLM judge found unsupported meaning",
                "unsupported_words": [],
            }
        self.verified_sentence = sentence.strip()
        return {"supported": True, "reason": "rule and configured judge accepted"}

    def speak(self, sentence: str) -> dict[str, Any]:
        if not self.verified_sentence or sentence.strip() != self.verified_sentence:
            return {
                "spoken": False,
                "reason": "sentence must pass verification before speech",
            }
        if self.speak_callback:
            self.speak_callback(sentence)
        self.speech_requested = True
        return {"spoken": True, "sentence": sentence}

    def _execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "get_sign_buffer":
            return {"signs": self.get_sign_buffer()}
        if name == "request_repeat":
            return self.request_repeat(int(arguments["sign_index"]))
        if name == "lookup_vocabulary":
            return self.lookup_vocabulary(str(arguments["word"]))
        if name == "propose_sentence":
            glosses = [str(item) for item in arguments["glosses"]]
            sentence = (
                self.provider.draft_sentence(glosses)
                if self.provider else self._fallback._fallback_sentence(glosses)
            )
            return {"sentence": sentence}
        if name == "verify_sentence":
            return self.verify_sentence(
                str(arguments["sentence"]),
                [str(item) for item in arguments["glosses"]],
            )
        if name == "speak":
            return self.speak(str(arguments["sentence"]))
        raise ValueError(f"Unknown agent tool: {name}")

    def _call_tool(
        self, name: str, arguments: dict[str, Any], call_id: str
    ) -> dict[str, Any] | None:
        if not self._charge_step():
            self._trace(
                "step_limit_reached",
                tool_name=name,
                tool_call_id=call_id,
                limit=self.max_steps,
            )
            return None
        try:
            result = self._execute(name, arguments)
        except (KeyError, TypeError, ValueError) as exc:
            result = {"error": str(exc)}
        self._trace(
            "tool_call",
            tool_name=name,
            tool_call_id=call_id,
            arguments=arguments,
            result=result,
        )
        return result

    def _result(
        self, text: str, needs_repeat: bool = False, verified: bool = True
    ) -> AgentLoopResult:
        return AgentLoopResult(
            text=text,
            needs_repeat=needs_repeat,
            verified=verified,
            used_llm=self.provider is not None,
            steps=self.steps,
            trace_path=str(self.trace_path),
            repeat_indices=tuple(self.repeat_indices),
        )

    def _fallback_result(self, reason: str) -> AgentLoopResult:
        glosses = [item.label for item in self.sign_buffer]
        safe_text = self._fallback._fallback_sentence(glosses)
        unsupported = unsupported_content_words(safe_text, glosses)
        if unsupported:
            safe_text = " ".join(glosses).capitalize() + "."
        self._trace(
            "safe_fallback",
            reason=reason,
            text=safe_text,
            unsupported_words=sorted(unsupported),
        )
        return self._result(safe_text, verified=not unsupported)

    def run(self) -> AgentLoopResult:
        if not self.sign_buffer:
            self._trace("empty_buffer")
            return self._result("Start signing to build a sentence.")

        low_confidence = [
            index for index, item in enumerate(self.sign_buffer)
            if item.confidence < self.confidence_threshold
        ]
        if low_confidence:
            index = low_confidence[0]
            result = self._call_tool(
                "request_repeat", {"sign_index": index}, "preflight-repeat"
            )
            message = (
                str(result["reason"])
                if result else "Please repeat the uncertain sign."
            )
            return self._result(message, needs_repeat=True, verified=False)

        if self.provider is None:
            return self._run_offline()

        glosses = [item.label for item in self.sign_buffer]
        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": (
                "Recognized signs and confidences: "
                + json.dumps(self.get_sign_buffer())
                + ". Use the available tools to check the glosses, propose one "
                "faithful sentence, verify it, and speak only after verification."
            ),
        }]
        for _ in range(self.max_steps):
            if self.steps >= self.max_steps:
                break
            turn = self.provider.complete(messages, TOOL_SCHEMAS)
            assistant_calls = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in turn.tool_calls
            ]
            messages.append({
                "role": "assistant",
                "content": turn.final_text,
                "tool_calls": assistant_calls,
            })
            self._trace(
                "provider_turn",
                provider=self.provider.name,
                tool_names=[call.name for call in turn.tool_calls],
                has_final_text=bool(turn.final_text),
            )
            if turn.final_text and not turn.tool_calls:
                verification = self._call_tool(
                    "verify_sentence",
                    {"sentence": turn.final_text, "glosses": glosses},
                    "final-verification",
                )
                if verification and verification.get("supported"):
                    self._call_tool(
                        "speak",
                        {"sentence": turn.final_text},
                        "verified-speech",
                    )
                    return self._result(turn.final_text)
                messages.append({
                    "role": "user",
                    "content": "The proposed sentence failed verification. "
                               "Use tools to produce a supported sentence.",
                })
                continue
            for call in turn.tool_calls:
                result = self._call_tool(call.name, call.arguments, call.call_id)
                if result is None:
                    break
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "name": call.name,
                    "content": json.dumps(result),
                })
                if call.name == "request_repeat" and result.get("requested"):
                    return self._result(
                        result["reason"], needs_repeat=True, verified=False
                    )
                if call.name == "speak" and result.get("spoken"):
                    return self._result(str(result["sentence"]))
            else:
                continue
            break
        return self._fallback_result("bounded agent step limit reached")

    def _run_offline(self) -> AgentLoopResult:
        glosses = [item.label for item in self.sign_buffer]
        calls = [
            ("get_sign_buffer", {}, "offline-buffer"),
            ("lookup_vocabulary", {"word": glosses[0]}, "offline-vocabulary"),
            ("propose_sentence", {"glosses": glosses}, "offline-proposal"),
        ]
        for name, arguments, call_id in calls:
            if self.steps >= self.max_steps - 2:
                break
            self._call_tool(name, arguments, call_id)
        if self.steps < self.max_steps:
            sentence = self._fallback._fallback_sentence(glosses)
            verification = self._call_tool(
                "verify_sentence",
                {"sentence": sentence, "glosses": glosses},
                "offline-verification",
            )
            if verification and verification.get("supported"):
                self._call_tool(
                    "speak", {"sentence": sentence}, "offline-speech"
                )
                return self._result(sentence)
        return self._fallback_result("offline step budget exhausted")
