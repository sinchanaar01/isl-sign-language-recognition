"""Streamlit webcam and uploaded-video demo for trained landmark models."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import streamlit as st

from src.isl_recognition.agent import Prediction
from src.isl_recognition.agent_loop import (
    AgentLoop,
    provider_from_environment,
    vocabulary_from_file,
)
from src.isl_recognition.inference import (
    PredictionSmoother,
    load_landmark_model,
    predict_video_file,
)
from src.isl_recognition.preprocessing import (
    FEATURES_PER_FRAME,
    normalize_frame,
    normalize_sequence,
    select_modality,
)

st.set_page_config(page_title="ISL Recognition", page_icon="🤟", layout="wide")
st.title("🤟 Real-Time Indian Sign Language Recognition")
st.caption(
    "Research prototype only. ISL is a complete language; gloss predictions and "
    "generated English are not a substitute for a qualified interpreter."
)

checkpoint_path = Path(os.getenv("ISL_MODEL_PATH", "models/best_landmark.pt"))
if not checkpoint_path.is_file():
    st.warning(
        f"No trained landmark checkpoint at `{checkpoint_path}`. Upload/webcam "
        "predictions require real trained weights; no random model is substituted."
    )

try:
    import cv2
    import mediapipe as mp
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
except ImportError as exc:
    st.error(f"Install app dependencies from requirements.txt: {exc}")
    st.stop()


@st.cache_resource
def cached_landmark_model(
    path: str, modified_time: float
) -> tuple[object, list[str], int, str]:
    del modified_time
    return load_landmark_model(path)


class SignVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._smoother = PredictionSmoother(window_size=5)
        self._recent_predictions: deque[Prediction] = deque(maxlen=20)
        self._last_emitted_label = ""
        self._last_emitted_at = 0.0
        self._latest_label = "Waiting for a sign"
        self._latest_confidence = 0.0
        self._last_frame_latency_ms: float | None = None
        self._fps = 0.0
        self._started_at = time.perf_counter()
        self._frame_count = 0
        self._model = None
        self._classes: list[str] = []
        self._sequence_length = 32
        self._modality = "hands_pose"
        self._sequence: deque[np.ndarray] = deque(maxlen=self._sequence_length)
        self._holistic = None
        if checkpoint_path.is_file():
            (
                self._model,
                self._classes,
                self._sequence_length,
                self._modality,
            ) = cached_landmark_model(
                str(checkpoint_path.resolve()), checkpoint_path.stat().st_mtime
            )
            self._sequence = deque(maxlen=self._sequence_length)
            self._holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                refine_face_landmarks=False,
            )

    def recv(self, frame):
        started = time.perf_counter()
        image = frame.to_ndarray(format="bgr24")
        if self._holistic is not None and self._model is not None:
            results = self._holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            landmarks = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
            if results.pose_landmarks:
                for index, point in enumerate(results.pose_landmarks.landmark):
                    landmarks[index * 3:index * 3 + 3] = (
                        point.x, point.y, point.z
                    )
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
                sequence = normalize_sequence(
                    np.stack(self._sequence), self._sequence_length
                )
                sequence = select_modality(sequence, self._modality)
                import torch

                with torch.inference_mode():
                    logits = self._model(torch.from_numpy(sequence).unsqueeze(0))
                    probabilities = torch.softmax(logits[0], dim=0).tolist()
                label, confidence = self._smoother.update({
                    name: float(probability)
                    for name, probability in zip(self._classes, probabilities)
                })
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

        elapsed = time.perf_counter() - started
        with self._lock:
            self._frame_count += 1
            self._last_frame_latency_ms = elapsed * 1000
            total_elapsed = time.perf_counter() - self._started_at
            self._fps = self._frame_count / total_elapsed if total_elapsed > 0 else 0.0
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
            return {
                "label": self._latest_label,
                "confidence": self._latest_confidence,
                "buffer": list(self._recent_predictions),
                "fps": self._fps,
                "end_to_end_latency_ms": self._last_frame_latency_ms,
            }

    def clear_predictions(self) -> None:
        with self._lock:
            self._recent_predictions.clear()

    def on_ended(self) -> None:
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None


@st.fragment(run_every=0.5)
def show_live_status(context) -> None:
    processor = context.video_processor
    if processor is None:
        st.info("Allow camera access and start the stream to begin recognition.")
        return
    snapshot = processor.snapshot()
    st.metric(
        "Current sign",
        snapshot["label"],
        f"{snapshot['confidence']:.0%} confidence",
    )
    st.progress(float(max(0.0, min(1.0, snapshot["confidence"]))))
    fps, latency = st.columns(2)
    fps.metric("Processing FPS", f"{snapshot['fps']:.1f}")
    latency_value = snapshot["end_to_end_latency_ms"]
    latency.metric(
        "Frame end-to-end latency",
        "—" if latency_value is None else f"{latency_value:.1f} ms",
    )
    st.caption(
        "Stable sign buffer: "
        + (" → ".join(item.label for item in snapshot["buffer"]) or "empty")
    )


webcam_buffer: list[Prediction] = []
processor = None
webcam_tab, upload_tab = st.tabs(["Webcam", "Record and test"])
with webcam_tab:
    context = webrtc_streamer(
        key="isl-webcam",
        video_processor_factory=SignVideoProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
    )
    show_live_status(context)
    processor = context.video_processor
    if processor is not None:
        webcam_buffer = processor.snapshot()["buffer"]

with upload_tab:
    st.write(
        "Upload a short video to run MediaPipe plus the trained classifier. The "
        "temporary upload is deleted after inference; it is not added to the dataset "
        "automatically."
    )
    upload = st.file_uploader(
        "Upload a video clip",
        type=["mp4", "mov", "avi", "mkv"],
        key="video-upload",
    )
    if upload is not None and checkpoint_path.is_file():
        if st.button("Run clip inference", key="run-upload-inference"):
            temporary_path: Path | None = None
            try:
                suffix = Path(upload.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                    temporary_path = Path(handle.name)
                    handle.write(upload.getvalue())
                model, classes, sequence_length, modality = cached_landmark_model(
                    str(checkpoint_path.resolve()), checkpoint_path.stat().st_mtime
                )
                with st.spinner("Extracting landmarks and classifying the clip..."):
                    result = predict_video_file(
                        temporary_path, model, classes, sequence_length, modality
                    )
                st.session_state["upload_prediction"] = result
                st.session_state["upload_buffer"] = [
                    Prediction(result.label, result.confidence)
                ]
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(f"Clip inference failed: {exc}")
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
    elif upload is not None:
        st.info("Train a model and set ISL_MODEL_PATH before running clip inference.")

    upload_prediction = st.session_state.get("upload_prediction")
    if upload_prediction is not None:
        st.subheader("Uploaded clip result")
        st.metric(
            upload_prediction.label, f"{upload_prediction.confidence:.1%} confidence"
        )
        st.progress(float(max(0.0, min(1.0, upload_prediction.confidence))))
        model_latency, pipeline_latency, frames_count = st.columns(3)
        model_latency.metric(
            "Model only", f"{upload_prediction.model_only_latency_ms:.1f} ms"
        )
        pipeline_latency.metric(
            "MediaPipe + model",
            f"{upload_prediction.end_to_end_latency_ms:.1f} ms",
        )
        frames_count.metric("Extracted frames", upload_prediction.extracted_frame_count)
        st.dataframe(
            [
                {"sign": label, "confidence": confidence}
                for label, confidence in sorted(
                    upload_prediction.probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5]
            ],
            hide_index=True,
        )
        if st.button("Add uploaded prediction to agent buffer"):
            st.session_state.setdefault("upload_buffer", []).append(
                Prediction(upload_prediction.label, upload_prediction.confidence)
            )

upload_buffer = st.session_state.get("upload_buffer", [])
agent_buffer = list(webcam_buffer) + list(upload_buffer)
st.header("Agentic sentence")
st.caption(
    "The agent verifies content before enabling speech. Low-confidence predictions "
    "request a repeat without removing other signs."
)
if agent_buffer:
    st.code(
        " → ".join(
            f"{prediction.label} ({prediction.confidence:.0%})"
            for prediction in agent_buffer
        ),
        language="text",
    )
if st.button("Generate and verify sentence", disabled=not agent_buffer):
    try:
        session_state = st.session_state
        loop = AgentLoop(
            agent_buffer,
            vocabulary_from_file("data/target_glosses.txt"),
            provider=provider_from_environment(),
            speak_callback=lambda sentence: session_state.update({
                "agent_tts_text": sentence
            }),
        )
        result = loop.run()
        session_state["agent_response"] = result
        session_state["agent_trace_path"] = result.trace_path
        if result.needs_repeat:
            session_state["repeat_notice"] = result.text
        else:
            session_state.pop("repeat_notice", None)
    except RuntimeError as exc:
        st.error(f"Agent execution failed: {exc}")

response = st.session_state.get("agent_response")
if response is not None:
    if response.needs_repeat:
        st.warning(response.text)
    elif not response.verified:
        st.warning("Sentence could not be verified; speech is disabled.")
    else:
        st.subheader("Verified sentence")
        st.success(response.text)
        st.caption(
            f"Verified: {response.verified} · "
            f"Provider: {'LLM' if response.used_llm else 'offline fallback'} · "
            f"Agent tool steps: {response.steps}"
        )
        safe_text = json.dumps(response.text, ensure_ascii=True).replace("<", "\\u003c")
        st.components.v1.html(
            f"<button onclick='window.speechSynthesis.cancel();"
            f"window.speechSynthesis.speak(new SpeechSynthesisUtterance({safe_text}));'>"
            "Speak sentence</button>",
            height=48,
        )

trace_path = st.session_state.get("agent_trace_path")
with st.expander("Agent trace (latest run)"):
    if trace_path and Path(trace_path).is_file():
        trace_lines = Path(trace_path).read_text(encoding="utf-8").splitlines()
        st.code("\n".join(trace_lines[-20:]), language="json")
    else:
        st.info("No agent run has been recorded in this session.")
