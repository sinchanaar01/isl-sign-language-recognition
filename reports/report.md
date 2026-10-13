# Technical report: Real-Time ISL Recognition with an Agentic Sentence Layer

## Abstract

This repository implements a reproducible prototype pipeline for isolated sign recognition from webcam video and confidence-aware sentence generation. Its planned in-domain source is the INCLUDE Indian Sign Language dataset, with optional comparison against static images and the out-of-domain WLASL American Sign Language corpus. As of this report, only WLASL's annotation JSON has been downloaded; no media dataset was prepared and no model was trained. Accordingly, no recognition score or latency result is reported.

## Method

MediaPipe Holistic provides 33 pose landmarks (xyz and visibility) and 21 landmarks for each hand (xyz), yielding 258 features per frame. Coordinates are centered on the shoulder midpoint and scaled by shoulder width; clips are linearly resampled to a fixed 32-frame sequence. Seeded landmark noise and hand dropout are applied in the training dataset only. The split groups entire signer IDs into train, validation, and test partitions.

The temporal comparison comprises two-layer bidirectional LSTM and GRU classifiers, a small two-layer Transformer with learned positional embeddings, and a temporal convolutional network (TCN). Ablations vary hand-only wrist/palm-normalized features versus pose+hands, augmentation on/off, sequence length, and random seed. The full default grid has not been executed. A MobileNetV2 image classifier provides a separate static-sign baseline and requires signer IDs encoded in filenames. Runs log training parameters, validation curves, checkpoint artifacts, test metrics, and device information to MLflow. Best checkpoint selection is by validation accuracy. Held-out reports include accuracy, per-class precision/recall/F1, a confusion matrix, top confusion pairs, and model-only latency; test-clip MediaPipe+model latency is measured when source videos are available.

The app maintains a short moving average of class probabilities. A confidence-aware, maximum-five-step agent uses OpenAI-compatible or Anthropic native tool calling. It can inspect the sign buffer, request a repeat for an individual low-confidence sign, look up candidate vocabulary, draft a sentence, verify it with a lexical content-word guard and optional LLM judge, and queue TTS only after acceptance. With no provider key, the offline fallback follows the same rule-based verification. Every tool decision is logged to an ignored local JSONL trace.

## Results

| Model | Test accuracy | Macro F1 | Mean model latency |
| --- | ---: | ---: | ---: |
| MobileNetV2 static | Not measured | Not measured | Not measured |
| BiLSTM landmarks | Not measured | Not measured | Not measured |
| BiGRU landmarks | Not measured | Not measured | Not measured |
| Transformer landmarks | Not measured | Not measured | Not measured |
| TCN landmarks | Not measured | Not measured | Not measured |

The WLASL v0.3 JSON was retrieved from the official GitHub repository and parsed successfully: 2,000 gloss categories and 21,083 annotated instances. These are **metadata counts**, not locally available video counts or model samples. WLASL is ASL and its upstream Computational Use of Data Agreement must be reviewed and accepted before acquiring/using the videos. INCLUDE first-party media/license could not be verified from this environment; Kaggle discovery was blocked and no specific uploader/license was selected. Therefore, a meaningful ISL metric cannot yet be computed.

The hand-written offline-agent evaluation is saved at `reports/runs/agent_eval_20261013T164359Z.json`. It covers thirty authored gloss cases, of which twenty-five were eligible to produce text and five were intentionally low-confidence repeat cases. On this fixed set, the deterministic fallback exactly matched all eligible expected sentences; the mechanical unsupported-content-word detector flagged none; and repeat-request precision was 1.0 on five true-positive requests. These measurements validate only the code's behavior on the specified authored strings; they are not recognition quality, human linguistic evaluation, or LLM-provider quality. The evaluator must be rerun after code/data changes.

## Error analysis

No model predictions exist yet, so no empirical confusion/error analysis is possible. Once data is acquired and training has completed, inspect per-class recall/F1 and the saved confusion matrices for confused handshapes, low-support classes, and signer-dependent failures. Record the selected split, class counts, signer counts, and evaluation device beside each run.

## Limitations

The implementation classifies isolated labeled signs; it does not decode continuous signing or represent non-manual grammar and facial expression. The target English-gloss list is a draft, not an ISL lexicon. Landmark-only models discard appearance and may fail under occlusion, unusual camera placement, or pose estimation errors. A fluent-signer-verified label map and signer-diverse ISL test set are essential. LLM sentence rewriting can alter meaning; the literal/offline fallback and confidence gate reduce but do not remove this risk. Browser TTS and hosted WebRTC reliability depend on client, HTTPS, and TURN configuration.

## Next steps

1. Obtain licensed, signer-annotated INCLUDE/ISL samples or record consented clips; verify the target vocabulary with fluent ISL signers.
2. Capture/verify strict signer-independent splits with class coverage in training and evaluation.
3. Run all four temporal baselines and static MobileNetV2 if signer-labeled images are available; inspect test errors and class metrics, record latency and device, and replace the results table with measured outputs.
4. Test end-to-end webcam performance on representative signers; calibrate the repeat threshold and review consent, privacy, and accessibility with ISL users.
