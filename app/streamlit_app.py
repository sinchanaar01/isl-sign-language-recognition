"""Streamlit + WebRTC webcam demo for trained landmark models."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque

import numpy as np
import streamlit as st

from src.isl_recognition.agent import Prediction, SentenceAgent
from src.isl_recognition.inference import PredictionSmoother, load_landmark_model
from src.isl_recognition.preprocessing import (
    FEATURES_PER_FRAME,
    normalize_frame,
    normalize_sequence,
)

st.set_page_config(page_title="ISL Recognition", page_icon="🤟", layout="wide")
st.title("🤟 Real-Time Indian Sign Language Recognition")
st.caption(
    "A research prototype: sign glosses are not a substitute for a qualified ISL "
    "interpreter. Recognition quality depends on the dataset and signer coverage."
)

checkpoint_path = os.getenv("ISL_MODEL_PATH", "models/best_landmark.pt")
if not os.path.isfile(checkpoint_path):
    st.warning(
        f"No trained landmark checkpoint found at `{checkpoint_path}`. "
        "Train a model first; camera display may work, but predictions are disabled."
    )

try:
    import cv2
    import mediapipe as mp
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
except ImportError as exc:
    st.error(f"Install app dependencies from requirements.txt: {exc}")
    st.stop()


class SignVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence: deque[np.ndarray] = deque(maxlen=96)
        self._smoother = PredictionSmoother(window_size=5)
        self._recent_predictions: deque[Prediction] = deque(maxlen=20)
        self._last_emitted_label = ""
        self._last_emitted_at = 0.0
        self._latest_label = "Waiting for a sign"
        self._latest_confidence = 0.0
        self._model = None
        self._classes: list[str] = []
        self._sequence_length = 32
        self._holistic = None
        if os.path.isfile(checkpoint_path):
            self._model, self._classes, self._sequence_length = load_landmark_model(
                checkpoint_path
            )
            self._sequence = deque(maxlen=self._sequence_length)
            self._holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                refine_face_landmarks=False,
            )

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        if self._holistic is not None:
            results = self._holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            landmarks = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
            if results.pose_landmarks:
                for index, point in enumerate(results.pose_landmarks.landmark):
                    landmarks[index * 3:index * 3 + 3] = (point.x, point.y, point.z)
                    landmarks[99 + index] = point.visibility
            for hand, start in (
                (results.left_hand_landmarks, 132),
                (results.right_hand_landmarks, 195),
            ):
                if hand:
                    for index, point in enumerate(hand.landmark):
                        landmarks[start + index * 3:start + index * 3 + 3] = (
                            point.x, point.y, point.z
                        )
            self._sequence.append(normalize_frame(landmarks))
            if len(self._sequence) == self._sequence_length:
                batch = normalize_sequence(np.stack(self._sequence), self._sequence_length)
                import torch
                with torch.inference_mode():
                    logits = self._model(torch.from_numpy(batch).unsqueeze(0))
                    probabilities = torch.softmax(logits[0], dim=0).tolist()
                prediction = {
                    label: float(probability)
                    for label, probability in zip(self._classes, probabilities)
                }
                label, confidence = self._smoother.update(prediction)
                now = time.monotonic()
                if (
                    label.casefold() != self._last_emitted_label.casefold()
                    and now - self._last_emitted_at >= 0.5
                ):
                    with self._lock:
                        self._recent_predictions.append(Prediction(label, confidence))
                    self._last_emitted_label = label
                    self._last_emitted_at = now
                with self._lock:
                    self._latest_label = label
                    self._latest_confidence = confidence
        with self._lock:
            label = self._latest_label
            confidence = self._latest_confidence
        cv2.putText(
            image,
            f"{label}  {confidence:.0%}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 0),
            2,
        )
        return type(frame).from_ndarray(image, format="bgr24")

    def snapshot(self):
        with self._lock:
            return self._latest_label, self._latest_confidence, list(self._recent_predictions)

    def clear_predictions(self):
        with self._lock:
            self._recent_predictions.clear()


webrtc_ctx = webrtc_streamer(
    key="isl-webcam",
    video_processor_factory=SignVideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)


@st.fragment(run_every=0.5)
def live_status():
    if webrtc_ctx.video_processor:
        label, confidence, predictions = webrtc_ctx.video_processor.snapshot()
        st.metric("Current sign", label, f"{confidence:.0%} confidence")
        st.caption("Recent stable signs: " + " → ".join(p.label for p in predictions))
    else:
        st.info("Allow camera access and start the stream to begin recognition.")


live_status()
if webrtc_ctx.video_processor:
    _, _, recent_predictions = webrtc_ctx.video_processor.snapshot()
else:
    recent_predictions = []

if st.button("Generate sentence", disabled=not recent_predictions):
    webrtc_ctx.video_processor.clear_predictions()
    try:
        response = SentenceAgent().generate(recent_predictions)
        if response.needs_repeat:
            st.warning(response.text)
        else:
            st.success(response.text)
            st.caption(
                "Generated with the configured LLM." if response.used_llm else
                "Rule-based fallback; set OPENAI_API_KEY for LLM sentence generation."
            )
            safe_text = json.dumps(response.text).replace("<", "\\u003c")
            st.components.v1.html(
                f"<script>window.speechSynthesis.cancel();"
                f"window.speechSynthesis.speak(new SpeechSynthesisUtterance({safe_text}));"
                f"</script>",
                height=0,
            )
    except RuntimeError as exc:
        st.error(f"Could not generate the sentence: {exc}")
elif not recent_predictions:
    st.caption("Low-confidence predictions are passed to the agent, which asks you to repeat.")
