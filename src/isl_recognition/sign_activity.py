"""Motion-gated segment detection for isolated sign clips."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

DetectorState = Literal["ready", "signing", "processing", "cooldown"]
WristPosition = tuple[float, float]


@dataclass(frozen=True)
class SignSegment:
    frames: tuple[NDArray[np.float32], ...]
    started_at: float
    ended_at: float


@dataclass(frozen=True)
class SegmentUpdate:
    state: DetectorState
    motion: float
    segment: SignSegment | None = None


class SignActivityDetector:
    """Detect motion-started signs and finish them after sustained stillness."""

    def __init__(
        self,
        motion_threshold: float = 0.02,
        onset_frames: int = 3,
        end_frames: int = 12,
        cooldown_seconds: float = 0.8,
        pre_roll_frames: int = 8,
    ) -> None:
        if motion_threshold <= 0:
            raise ValueError("motion_threshold must be positive")
        if onset_frames < 1 or end_frames < 1 or pre_roll_frames < 1:
            raise ValueError(
                "onset_frames, end_frames, and pre_roll_frames must be positive"
            )
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")
        self.motion_threshold = motion_threshold
        self.onset_frames = onset_frames
        self.end_frames = end_frames
        self.cooldown_seconds = cooldown_seconds
        self._pre_roll: deque[tuple[NDArray[np.float32], float]] = deque(
            maxlen=pre_roll_frames
        )
        self._state: DetectorState = "ready"
        self._previous_wrists: tuple[WristPosition, ...] = ()
        self._active_count = 0
        self._still_count = 0
        self._segment_frames: list[NDArray[np.float32]] = []
        self._segment_started_at = 0.0
        self._cooldown_started_at = 0.0

    @property
    def state(self) -> DetectorState:
        return self._state

    def _motion(self, wrists: Sequence[WristPosition]) -> float:
        if not wrists or not self._previous_wrists:
            return 0.0
        count = min(len(wrists), len(self._previous_wrists))
        return max(
            float(
                np.hypot(
                    wrists[index][0] - self._previous_wrists[index][0],
                    wrists[index][1] - self._previous_wrists[index][1],
                )
            )
            for index in range(count)
        )

    def update(
        self,
        frame: NDArray[np.float32],
        wrists: Sequence[WristPosition],
        hands_visible: bool,
        timestamp: float | None = None,
    ) -> SegmentUpdate:
        """Consume one normalized landmark frame and return any completed segment."""
        now = time.monotonic() if timestamp is None else timestamp
        motion = self._motion(wrists) if hands_visible else 0.0
        moving = hands_visible and motion >= self.motion_threshold
        stable = not hands_visible or motion < self.motion_threshold
        copied_frame = np.asarray(frame, dtype=np.float32).copy()
        self._pre_roll.append((copied_frame, now))

        if self._state == "processing":
            self._previous_wrists = tuple(wrists) if hands_visible else ()
            return SegmentUpdate(self._state, motion)

        if self._state == "ready":
            self._active_count = self._active_count + 1 if moving else 0
            if self._active_count >= self.onset_frames:
                self._state = "signing"
                self._segment_frames = [item[0] for item in self._pre_roll]
                self._segment_started_at = self._pre_roll[0][1]
                self._still_count = 0
        elif self._state == "signing":
            self._segment_frames.append(copied_frame)
            self._still_count = self._still_count + 1 if stable else 0
            if self._still_count >= self.end_frames:
                segment = SignSegment(
                    frames=tuple(self._segment_frames),
                    started_at=self._segment_started_at,
                    ended_at=now,
                )
                self._state = "processing"
                self._active_count = 0
                self._still_count = 0
                self._previous_wrists = tuple(wrists) if hands_visible else ()
                return SegmentUpdate(self._state, motion, segment)
        elif self._state == "cooldown":
            self._still_count = self._still_count + 1 if stable else 0
            if (
                now - self._cooldown_started_at >= self.cooldown_seconds
                and self._still_count >= self.end_frames
            ):
                self._state = "ready"
                self._active_count = 0
                self._still_count = 0
                self._pre_roll.clear()

        self._previous_wrists = tuple(wrists) if hands_visible else ()
        return SegmentUpdate(self._state, motion)

    def finish_processing(self, timestamp: float | None = None) -> None:
        """Enter cooldown after the completed segment has been classified."""
        if self._state != "processing":
            raise RuntimeError("finish_processing requires a completed segment")
        self._state = "cooldown"
        self._cooldown_started_at = time.monotonic() if timestamp is None else timestamp
        self._still_count = 0
        self._active_count = 0
        self._previous_wrists = ()
