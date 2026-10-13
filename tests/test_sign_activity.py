"""Deterministic motion-state smoke tests; no sign-video samples are used."""

from __future__ import annotations

import unittest

import numpy as np
from src.isl_recognition.sign_activity import SignActivityDetector


class SignActivityDetectorTests(unittest.TestCase):
    def frame(self, marker: float) -> np.ndarray:
        return np.array([marker], dtype=np.float32)

    def test_requires_visible_wrist_motion_for_consecutive_onset_frames(self):
        detector = SignActivityDetector(
            motion_threshold=0.02,
            onset_frames=3,
            end_frames=2,
            pre_roll_frames=4,
        )
        self.assertEqual(
            detector.update(self.frame(0), [(0.1, 0.1)], False, 0).state,
            "ready",
        )
        detector.update(self.frame(1), [(0.1, 0.1)], True, 1)
        self.assertEqual(
            detector.update(self.frame(2), [(0.14, 0.1)], True, 2).state,
            "ready",
        )
        self.assertEqual(
            detector.update(self.frame(3), [(0.18, 0.1)], True, 3).state,
            "ready",
        )
        result = detector.update(self.frame(4), [(0.22, 0.1)], True, 4)

        self.assertEqual(result.state, "signing")
        self.assertIsNone(result.segment)

    def test_ends_after_still_frames_and_emits_segment_once(self):
        detector = SignActivityDetector(
            motion_threshold=0.02,
            onset_frames=2,
            end_frames=3,
            cooldown_seconds=0.5,
            pre_roll_frames=3,
        )
        wrists = [(0.1, 0.2)]
        detector.update(self.frame(0), wrists, True, 0.0)
        detector.update(self.frame(1), [(0.14, 0.2)], True, 0.1)
        self.assertEqual(
            detector.update(self.frame(2), [(0.18, 0.2)], True, 0.2).state,
            "signing",
        )
        detector.update(self.frame(3), [(0.18, 0.2)], True, 0.3)
        detector.update(self.frame(4), [(0.18, 0.2)], True, 0.4)
        ended = detector.update(self.frame(5), [(0.18, 0.2)], True, 0.5)

        self.assertEqual(ended.state, "processing")
        self.assertIsNotNone(ended.segment)
        self.assertEqual(ended.segment.frames[-1].tolist(), [5.0])
        self.assertAlmostEqual(ended.segment.started_at, 0.0)
        self.assertAlmostEqual(ended.segment.ended_at, 0.5)
        self.assertIsNone(detector.update(self.frame(6), wrists, True, 0.6).segment)

    def test_cooldown_waits_for_timeout_and_release_before_rearming(self):
        detector = SignActivityDetector(
            motion_threshold=0.02,
            onset_frames=1,
            end_frames=1,
            cooldown_seconds=0.5,
            pre_roll_frames=1,
        )
        detector.update(self.frame(0), [(0.1, 0.1)], True, 0.0)
        detector.update(self.frame(1), [(0.14, 0.1)], True, 0.1)
        ended = detector.update(self.frame(2), [(0.14, 0.1)], True, 0.2)
        self.assertEqual(ended.state, "processing")
        detector.finish_processing(0.2)
        self.assertEqual(detector.state, "cooldown")

        for frame_number, timestamp in ((3, 0.4), (4, 0.71)):
            update = detector.update(
                self.frame(frame_number), [(0.14, 0.1)], True, timestamp
            )
        self.assertEqual(update.state, "ready")

    def test_processing_must_be_explicitly_finished(self):
        detector = SignActivityDetector(onset_frames=1, end_frames=1)
        detector.update(self.frame(0), [(0.1, 0.1)], True, 0)
        detector.update(self.frame(1), [(0.14, 0.1)], True, 1)
        with self.assertRaisesRegex(RuntimeError, "completed segment"):
            detector.finish_processing(2)


if __name__ == "__main__":
    unittest.main()
