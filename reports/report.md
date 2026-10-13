# Technical report: Real-Time ISL Recognition with an Agentic Sentence Layer

## Abstract

This repository implements a reproducible prototype pipeline for isolated sign recognition from webcam video and confidence-aware sentence generation. Its planned in-domain source is the INCLUDE Indian Sign Language dataset, with optional comparison against static images and the out-of-domain WLASL American Sign Language corpus. As of this report, only WLASL's annotation JSON has been downloaded; no media dataset was prepared and no model was trained. Accordingly, no recognition score or latency result is reported.

## Method

MediaPipe Holistic provides 33 pose landmarks (xyz and visibility) and 21 landmarks for each hand (xyz), yielding 258 features per frame. Coordinates are centered on the shoulder midpoint and scaled by shoulder width; clips are linearly resampled to a fixed 32-frame sequence. Seeded landmark noise and hand dropout are applied in the training dataset only. The split groups entire signer IDs into train, validation, and test partitions.

The planned temporal comparison comprises two-layer bidirectional LSTM and GRU classifiers and a small two-layer Transformer with learned positional embeddings. A MobileNetV2 image classifier provides a separate static-sign baseline. Runs log training parameters, validation curves, checkpoint artifacts, test metrics, and device information to MLflow. Best temporal model selection is by validation accuracy. Held-out reports include accuracy, per-class precision/recall/F1, a confusion matrix, and mean model-forward latency.

The app maintains a short moving average of class probabilities. A confidence-aware agent requests a repeated sign below a configurable threshold; above the threshold, it sends ordered glosses to an OpenAI-compatible chat completion API when a key is configured. Without a key it uses deterministic text fallback. Speech is emitted through browser speech synthesis.

## Results

| Model | Samples | Test accuracy | Macro F1 | Mean model latency |
| --- | ---: | ---: | ---: | ---: |
| MobileNetV2 static | 0 | Not measured | Not measured | Not measured |
| LSTM landmarks | 0 | Not measured | Not measured | Not measured |
| GRU landmarks | 0 | Not measured | Not measured | Not measured |
| Transformer landmarks | 0 | Not measured | Not measured | Not measured |

The WLASL v0.3 JSON was retrieved from the official GitHub repository and parsed successfully: 2,000 gloss categories and 21,083 annotated instances. These are **metadata counts**, not locally available video counts or model samples. WLASL is ASL and its upstream Computational Use of Data Agreement must be reviewed and accepted before acquiring/using the videos. INCLUDE first-party media/license could not be verified from this environment; Kaggle discovery was blocked and no specific uploader/license was selected. Therefore, a meaningful ISL metric cannot yet be computed.

## Error analysis

No model predictions exist yet, so no empirical confusion/error analysis is possible. Once data is acquired and training has completed, inspect per-class recall/F1 and the saved confusion matrices for confused handshapes, low-support classes, and signer-dependent failures. Record the selected split, class counts, signer counts, and evaluation device beside each run.

## Limitations

The implementation classifies isolated labeled signs; it does not decode continuous signing or represent non-manual grammar and facial expression. The target English-gloss list is a draft, not an ISL lexicon. Landmark-only models discard appearance and may fail under occlusion, unusual camera placement, or pose estimation errors. A fluent-signer-verified label map and signer-diverse ISL test set are essential. LLM sentence rewriting can alter meaning; the literal/offline fallback and confidence gate reduce but do not remove this risk. Browser TTS and hosted WebRTC reliability depend on client, HTTPS, and TURN configuration.

## Next steps

1. Obtain licensed, signer-annotated INCLUDE/ISL samples; resolve the exact label mapping for the target vocabulary.
2. Capture/verify strict signer-independent splits with class coverage in training and evaluation.
3. Run all four baselines, inspect test errors and class metrics, record latency and device, and replace the results table with measured outputs.
4. Test end-to-end webcam performance on representative signers; calibrate the repeat threshold and review consent, privacy, and accessibility with ISL users.
