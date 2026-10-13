# Engineering decisions and assumptions

## Data and language

- **Do not substitute ASL for ISL.** WLASL is retained only as an auditable source of ASL annotations; its video media was not downloaded, and it cannot substantiate an ISL model.
- **First-party acquisition is a blocker.** INCLUDE's data/license endpoint and Kaggle's exact dataset/license could not be independently verified in this environment. A metadata JSON or public mirror is not permission to reuse video.
- **Vocabulary is a candidate list, not a claim of verified dictionary coverage.** The ISLRTC dictionary launch page is cited, but individual headword entries could not be retrieved. A fluent ISL signer must check every candidate sign and regional variant before collecting/labeling clips.
- **Keep live-demo clips out of training.** The webcam recorder stores `video_path,label,signer_id` under `data/raw/own/`, provides clip review and deletion, and does not upload samples. Those clips are reserved for held-out live-demo checks. Training data must come from a separate, appropriately licensed external source under `data/raw/downloaded/`. The proposed collection target (20 candidate words, 30 clips each, at least three consenting signers) is a plan, not current coverage.
- **Signer IDs define the split unit.** Clips from one signer must never appear across train/validation/test. Because signer grouping may leave a class without train support, the manifest/trainer fail instead of silently leaking signers.
- **Keep local recordings held out.** `tools.record_clips` writes only under `data/raw/own/`; those clips are reserved for live-demo checks and never enter training manifests. Licensed external training data remains isolated under `data/raw/downloaded/`.

## Models and experiment design

- Use compact landmark-sequence models (bidirectional LSTM/GRU, small Transformer, TCN) as the temporal comparison; landmark features lower the input size but omit appearance and depend on MediaPipe quality.
- Keep MobileNetV2 as a distinct static-image baseline, not a directly comparable temporal model. Static image data also needs signer-prefixed filenames so leakage can be detected.
- The full default grid varies hand-only vs. pose+hands, training augmentation on/off, sequence lengths 16/32/48, and three seeds. These are experimental settings, not results. The best deployable checkpoint is selected on validation accuracy; the held-out test partition is used once for final reporting.
- Python 3.11.9 was available for supported-version tests; the project `.venv` uses Python 3.14 and is outside the declared support range. Neither active environment has the CUDA/PyTorch/OpenCV/MediaPipe stack needed for training or camera inference, and the machine has Intel UHD 620 graphics. The Colab notebook is a documented training path, not evidence that training ran.

## Sentence agent and safety

- Keep the agent loop bounded to at most five tool actions. Each sign is a `Prediction(label, confidence)`; low confidence requests repeating only that sign and preserves the rest of the buffer.
- Support OpenAI-compatible Chat Completions and Anthropic Messages via environment-selected provider adapters. Keys stay in environment/host secrets.
- Apply a token-level, rule-based unsupported-content check before an optional LLM judge. This is a conservative guard, not a semantic proof. Failed verification prevents TTS.
- Preserve a deterministic offline fallback so the app remains demonstrable without API keys. Its authored-case evaluation is not an ISL translation evaluation.
- Write tool decisions to local JSONL traces, but ignore traces in Git because gloss sequences can reveal sensitive content.

## Application and deployment

- Offer both WebRTC webcam streaming and uploaded-video inference. The upload path deletes its temporary input in a `finally` block and reports measured per-clip pipeline/model latency when run.
- Keep speech synthesis in the user's browser. This avoids an additional server-side voice service; browser voices and availability vary.
- Use Hugging Face Docker Spaces metadata and port 7860. Do not publish trained weights until derived-weight permissions and signer consent are established. Without a checkpoint, app inference is disabled rather than replaced with untrained/random output.
- No deployment was attempted because HF credentials, a trained model, and a distribution decision are unavailable.
- The local OpenCV live loop uses defaults of 3 consecutive moving-wrist frames to start, 12 still/no-hand frames to end, 32 resampled frames per classification, and a 0.8-second cooldown with release gating. These are implementation defaults, not empirically tuned values; no camera measurement has been made.
