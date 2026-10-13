import json
import io
import tempfile
import unittest
from unittest.mock import patch

from src.isl_recognition.agent import Prediction
from src.isl_recognition.agent_loop import (
    AgentLoop,
    AgentTurn,
    AnthropicProvider,
    OpenAICompatibleProvider,
    ToolCall,
    provider_from_environment,
    unsupported_content_words,
)


class MockProvider:
    name = "mock"

    def __init__(self, turns, draft="I want water.", judge=True):
        self.turns = iter(turns)
        self.draft = draft
        self.judge = judge
        self.complete_calls = 0
        self.judge_calls = 0

    def complete(self, messages, tools):
        self.complete_calls += 1
        return next(self.turns)

    def draft_sentence(self, glosses):
        return self.draft

    def judge_sentence(self, sentence, glosses):
        self.judge_calls += 1
        return self.judge


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary = {
            "i": "pronoun",
            "want": "verb",
            "water": "noun",
            "help": "verb",
        }

    def make_loop(self, *args, **kwargs):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return AgentLoop(
            *args, trace_directory=temporary.name, **kwargs
        )

    def test_low_confidence_requests_only_that_sign_again(self):
        requested = []
        loop = self.make_loop(
            [
                Prediction("I", 0.97),
                Prediction("WATER", 0.31),
                Prediction("PLEASE", 0.96),
            ],
            self.vocabulary,
            provider=MockProvider([]),
            request_repeat_callback=requested.append,
        )
        result = loop.run()
        self.assertTrue(result.needs_repeat)
        self.assertEqual(requested, [1])
        self.assertEqual(len(loop.sign_buffer), 3)
        self.assertEqual(result.repeat_indices, (1,))
        self.assertEqual(result.steps, 1)

    def test_hallucinated_content_is_rejected_before_valid_retry(self):
        provider = MockProvider([
            AgentTurn([ToolCall(
                "verify_sentence",
                {
                    "sentence": "I want a spaceship.",
                    "glosses": ["I", "WANT", "WATER"],
                },
                "verify-bad",
            )]),
            AgentTurn([ToolCall(
                "propose_sentence", {"glosses": ["I", "WANT", "WATER"]}, "draft"
            )]),
            AgentTurn([ToolCall(
                "verify_sentence",
                {"sentence": "I want water.", "glosses": ["I", "WANT", "WATER"]},
                "verify-good",
            )]),
            AgentTurn([ToolCall("speak", {"sentence": "I want water."}, "speak")]),
        ])
        spoken = []
        loop = self.make_loop(
            [Prediction("I", 0.95), Prediction("WANT", 0.95), Prediction("WATER", 0.95)],
            self.vocabulary,
            provider=provider,
            speak_callback=spoken.append,
        )
        result = loop.run()
        self.assertEqual(result.text, "I want water.")
        self.assertTrue(result.verified)
        self.assertEqual(spoken, ["I want water."])
        with open(result.trace_path, encoding="utf-8") as handle:
            trace_lines = [json.loads(line) for line in handle]
        rejected = [
            record for record in trace_lines
            if record.get("tool_name") == "verify_sentence"
        ][0]
        self.assertFalse(rejected["result"]["supported"])
        self.assertIn("spaceship", rejected["result"]["unsupported_words"])

    def test_agent_tool_loop_stops_at_five_steps(self):
        provider = MockProvider([
            AgentTurn([ToolCall("get_sign_buffer", {}, f"buffer-{index}")])
            for index in range(10)
        ])
        loop = self.make_loop(
            [Prediction("HELP", 0.95)],
            self.vocabulary,
            provider=provider,
            max_steps=5,
        )
        result = loop.run()
        self.assertEqual(result.steps, 5)
        self.assertLessEqual(provider.complete_calls, 5)
        self.assertIn("Help", result.text)
        with open(result.trace_path, encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]
        tool_steps = [record for record in records if record["event"] == "tool_call"]
        self.assertEqual(len(tool_steps), 5)

    def test_offline_fallback_verifies_and_speaks_deterministically(self):
        spoken = []
        loop = self.make_loop(
            [
                Prediction("I", 0.95),
                Prediction("WANT", 0.96),
                Prediction("WATER", 0.99),
            ],
            self.vocabulary,
            speak_callback=spoken.append,
        )
        result = loop.run()
        self.assertEqual(result.text, "I want water.")
        self.assertTrue(result.verified)
        self.assertFalse(result.used_llm)
        self.assertEqual(spoken, ["I want water."])
        self.assertTrue(result.trace_path.endswith(".jsonl"))

    def test_lexical_guard_rejects_unsupported_content(self):
        self.assertEqual(
            unsupported_content_words("I want a spaceship.", ["I", "WANT", "WATER"]),
            {"spaceship"},
        )

    def test_speak_is_denied_before_sentence_verification(self):
        spoken = []
        loop = self.make_loop(
            [Prediction("WATER", 0.9)],
            self.vocabulary,
            speak_callback=spoken.append,
        )
        self.assertFalse(loop.speak("Water.").get("spoken"))
        self.assertEqual(spoken, [])

    def test_llm_judge_can_reject_lexically_supported_sentence(self):
        provider = MockProvider([], judge=False)
        loop = self.make_loop(
            [Prediction("WATER", 0.9)],
            self.vocabulary,
            provider=provider,
        )
        result = loop.verify_sentence("Water.", ["WATER"])
        self.assertFalse(result["supported"])
        self.assertEqual(result["reason"], "LLM judge found unsupported meaning")
        self.assertIsNone(loop.verified_sentence)

    def test_openai_compatible_tool_call_parsing(self):
        response = io.BytesIO(json.dumps({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call-1",
                        "function": {
                            "name": "get_sign_buffer",
                            "arguments": "{}",
                        },
                    }]
                }
            }]
        }).encode())
        provider = OpenAICompatibleProvider("secret", "unit-test")
        with patch("urllib.request.urlopen", return_value=response):
            turn = provider.complete([{"role": "user", "content": "signs"}], [])
        self.assertEqual(turn.tool_calls[0].name, "get_sign_buffer")
        self.assertEqual(turn.tool_calls[0].arguments, {})

    def test_anthropic_tool_call_parsing(self):
        response = io.BytesIO(json.dumps({
            "content": [{
                "type": "tool_use",
                "id": "tool-1",
                "name": "get_sign_buffer",
                "input": {},
            }]
        }).encode())
        provider = AnthropicProvider("secret", "unit-test")
        with patch("urllib.request.urlopen", return_value=response):
            turn = provider.complete([{"role": "user", "content": "signs"}], [])
        self.assertEqual(turn.tool_calls[0].name, "get_sign_buffer")

    def test_provider_is_selected_from_environment(self):
        with patch.dict(
            "os.environ",
            {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test"},
            clear=True,
        ):
            self.assertIsInstance(provider_from_environment(), AnthropicProvider)
        with patch.dict("os.environ", {"LLM_PROVIDER": "openai"}, clear=True):
            self.assertIsNone(provider_from_environment())


if __name__ == "__main__":
    unittest.main()
