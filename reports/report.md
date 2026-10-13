# Real-Time Indian Sign Language Recognition with an Agentic Sentence Layer

## Abstract

This report describes an end-to-end prototype for isolated Indian Sign Language (ISL) recognition from short video clips, followed by a confidence-aware agent that can turn predicted glosses into a candidate English sentence. The implementation includes MediaPipe landmark extraction, signer-exclusive data splitting, four temporal model families, experiment logging, evaluation artifact generation, an interactive webcam/upload app, and a bounded tool-calling layer with an offline fallback.

The key experimental limitation is data, not a missing score hidden by this report: no licensed ISL videos or consented signer clips are available in the workspace, so no recognition model has been trained. The downloaded WLASL artifact is annotation metadata for American Sign Language (ASL), not video and not ISL training data. Recognition accuracy, macro-F1, ablations, confusion analysis, and live inference performance are therefore explicitly **not measured**. A separate saved, authored-case check exercises only the offline sentence-agent behavior. All reported outcomes below identify the artifact from which they were measured.

## 1. Motivation and problem definition

ISL is a natural language used by Deaf communities in India. A practical vision system could support accessible interfaces and communication tools, but a label classifier is not a translator. Sign languages have their own grammars and use spatial and non-manual information; isolated English glosses and hand landmarks do not encode all of this. An English sentence generated from a sequence of predicted labels is consequently a fallible prototype output, not a faithful interpretation.

This project scopes recognition to a closed set of isolated signs. The input is a short clip; the recognition model predicts one label from the classes observed during training. A separate agent consumes temporally ordered labels and confidence values. It may request a repeat for an uncertain item, draft a grammatical English sentence, check for unsupported content, and offer text-to-speech (TTS) only after its verification gate passes. Continuous-sign segmentation, translation into ISL, and safety-critical use are out of scope.

The design favors a transparent, testable pipeline and honest data provenance over reporting proxy performance. A WLASL metadata audit provides a reproducible data-source check, and fixed authored examples exercise the sentence-agent path. Neither can stand in for evaluation on consented, signer-held-out ISL video.

## 2. Related work

Word-level sign recognition has benefited from larger video corpora and standardized comparisons. WLASL introduced a large word-level ASL dataset and compared recognition methods [1]. It is useful for method development, but it differs from ISL in language and cannot establish ISL generalization. This repository downloaded only its official annotation JSON, not video media. The reported category and annotation counts refer solely to that metadata file.

MediaPipe Holistic combines pose, face, and hand landmark components for real-time perception [2]. The project consumes pose and hand coordinates while omitting face landmarks. This lowers model input size and provides explicit spatial signals, at the cost of detector dependency and lost visual detail. The report does not claim that this representation is superior to pixels; that requires a controlled experiment.

For sequence modeling, LSTM networks were introduced to address long-range dependencies in recurrent learning [3]. The Transformer replaced recurrence with attention for sequence transduction [4]. This project includes bidirectional LSTM and GRU models, a compact Transformer, and a temporal convolutional baseline to compare different inductive biases under identical data splits. No architecture is declared best without completed runs.

The agent follows the general reasoning-and-action pattern in which a language model selects tools and observes their results [5]. Here, tool use is constrained by a five-action limit and local validation rules. This provides a bounded control flow, not a guarantee against language-model errors or semantic hallucinations.

## 3. Data and provenance

### 3.1 Sources audited

The WLASL v0.3 annotation JSON was fetched from a pinned revision of the upstream repository. The saved audit artifact, `reports/runs/wlasl_metadata_audit.json`, includes the URL, SHA-256 digest, timestamp, parsed counts, language, and explicit `video_media_downloaded: false` and `training_samples: 0` fields. The audit found 2,000 gloss categories, 21,083 annotation instances, 119 distinct signer identifiers, and 21,083 video identifiers in the metadata. These are counts of ASL annotations/IDs, not locally available recordings or training examples. WLASL's Computational Use of Data Agreement must be read and accepted before any media use; no such use is claimed here.

The INCLUDE ISL dataset was considered as the preferred in-domain source, but first-party media access and license terms could not be confirmed from this environment. No Kaggle dataset was selected because an uploader-specific license and signer provenance could not be verified. `data/sources.md` records these decisions. Candidate target words in `data/target_glosses.txt` cite the ISLRTC dictionary landing page, but the corresponding individual entries could not be verified. Each sign, label, and regional variant needs review by a fluent ISL signer.

### 3.2 Human collection and split

`tools/record_clips.py` records consented 2–3 second webcam clips, stores a pseudonymous signer ID, and allows review or deletion before continuing. `tools/make_manifest.py` verifies CSV fields, file readability, at least three distinct signers, non-empty partitions, and training coverage for every label. It assigns signer-exclusive train, validation, and test groups; it does not split adjacent clips of a signer across partitions. The proposed target of 20 words, 30 clips per word, and at least three signers is a collection plan, not observed data.

If a class does not occur for a training signer, the manifest reports the coverage issue instead of putting a signer into multiple partitions. The split must be inspected along with class-by-split counts before training. This is particularly important for small signer counts, where a nominally valid group split can leave an evaluation class unsupported.

## 4. Method

### 4.1 Landmark extraction and normalization

`src/isl_recognition/data.py` decodes video with OpenCV and processes selected frames with MediaPipe Holistic. A pose frame contributes 33 xyz locations and visibility values; the two hands contribute 21 xyz locations each, for 258 scalar features per frame in the current representation. Missing landmarks remain zero-valued. Pose coordinates are normalized around the shoulder midpoint and shoulder width. The `hands` ablation then uses only hand coordinates, centering each hand on its wrist and scaling by wrist-to-middle-metacarpophalangeal distance. Pose-plus-hands keeps the full feature vector.

Each example is resampled to a fixed temporal length. The trainer supports 16, 32, and 48 frames; the processing script extracts up to 96 source frames before model-side normalization. Training-only augmentation adds seeded coordinate noise and hand dropout. Validation and test examples are not augmented. Random seeds are recorded, and split overlap is checked before a run proceeds.

### 4.2 Models and training

The temporal model interface accepts sequences of landmark vectors and predicts one class. The candidate families are:

- **BiLSTM:** two-layer bidirectional LSTM, followed by classification.
- **BiGRU:** a corresponding two-layer bidirectional GRU.
- **Transformer:** a compact Transformer encoder with learned positional embeddings.
- **TCN:** temporal convolutions with dilations and residual connections.

The static image path uses MobileNetV2 and is separate because static images are not temporally equivalent to the video inputs. It requires filenames that carry signer IDs so signer separation can be checked. No image corpus is present.

By default, landmark runs vary four architectures, two modalities, augmentation on/off, three sequence lengths, and three seeds. Validation performance selects checkpoints; test data is reserved for final run metrics. MLflow records parameters and metrics, while JSON/CSV artifacts contain per-run results and aggregate ablations. The grid has not run in this workspace. There is no best trained model to distribute; `models/MODEL_CARD.md` says so.

### 4.3 Agent and app

The `SentenceAgent` consumes a sign buffer of label/confidence predictions and exposes six tools: `get_sign_buffer`, `request_repeat`, `lookup_vocabulary`, `propose_sentence`, `verify_sentence`, and `speak`. The loop is capped at five actions. A low-confidence item requests that single sign again rather than discarding the sentence. Providers are selected through environment variables for OpenAI-compatible Chat Completions or Anthropic Messages. Keys are not stored in the repository.

Sentence verification first applies a rule-based content-token check against the supplied glosses; an optional LLM judge provides an additional check. TTS is unavailable until verification succeeds. When no API key is configured, a deterministic offline fallback remains available. It is intentionally conservative and does not purport to translate ISL grammar. Tool calls are saved to local JSONL traces, ignored by Git because gloss streams may be sensitive.

The Streamlit interface has webcam/WebRTC and upload modes, smoothing/confidence display, sign buffer, agent trace, final sentence, and browser TTS. In the absence of a trained checkpoint, it disables model prediction rather than emitting fabricated or random labels. The upload path provides an inference UI without requiring camera permission or WebRTC reachability, but it still requires a real checkpoint for prediction.

## 5. Experiments and results

### 5.1 Recognition

No videos were available to preprocess, and no landmark model or image model was trained. Consequently, no signer-independent recognition result exists.

| Model/configuration | Accuracy | Macro-F1 | Per-class metrics | Confusion analysis | Latency |
| --- | --- | --- | --- | --- | --- |
| MobileNetV2 static images | Not measured | Not measured | Not measured | Not available | Not measured |
| BiLSTM landmarks | Not measured | Not measured | Not measured | Not available | Not measured |
| BiGRU landmarks | Not measured | Not measured | Not measured | Not available | Not measured |
| Transformer landmarks | Not measured | Not measured | Not measured | Not available | Not measured |
| TCN landmarks | Not measured | Not measured | Not measured | Not available | Not measured |

All architecture/modality/augmentation/sequence-length comparisons and three-seed mean/standard-deviation values remain unmeasured. There are no confusion pairs to analyze without test predictions. No real-time FPS or camera-to-sentence latency was recorded.

### 5.2 Offline authored-agent check

The recorded run `reports/runs/agent_eval_20261013T164559Z.json` contains a run timestamp and source Git commit. The deterministic fallback exactly matched the expected strings for 25/25 authored text-producing cases. No unsupported content words were flagged in those same examples. Repeat-request precision was 1.0 for five authored low-confidence cases. The other five of the thirty examples are repeat cases, not sentence matches.

These quantities describe a finite, hand-written set and deterministic local control flow only. They are not model-recognition metrics, human judgments of fluency or semantic fidelity, signer studies, or tests of a live LLM provider. In particular, the token guard can miss paraphrases and cannot establish that an English sentence preserves the intent of an ISL utterance.

### 5.3 Ablations

| Factor | Planned comparison | Result |
| --- | --- | --- |
| Input modality | Hands vs. pose + hands | Not run |
| Augmentation | On vs. off | Not run |
| Temporal length | 16 vs. 32 vs. 48 frames | Not run |
| Model family | BiLSTM vs. BiGRU vs. Transformer vs. TCN | Not run |
| Random seed | 42, 43, and 44 for each configuration | Not run |

The list of seed values describes the training configuration only. No mean, standard deviation, or test estimate is implied.

## 6. Error analysis

There are no model predictions or confusion-matrix images, so empirical classifier error analysis cannot be performed. After collecting data, inspect per-class precision/recall/F1, the confusion matrix, false positives under held-out signers, and sample-level confidence. Distinguish data coverage problems from visually similar signs and MediaPipe failures. Report signer and class counts with every split, and use only the held-out test results in the final performance table.

The authored agent set is small and hand-designed; its strings may overfit the fallback's templates. The lexical rejection rule is deliberately testable but is not a validated hallucination detector. Human ISL users and qualified interpreters should review both vocabulary and sentences before any user-facing release.

## 7. Limitations, ethics, and future work

The primary limitation is absence of an in-domain dataset and trained model. Further limitations include closed-vocabulary isolated classification, omission of facial expression and full non-manual grammar, sensitivity to occlusion/camera conditions, an uncalibrated confidence threshold, and the difference between predicted English glosses and ISL meaning. Public hosting adds questions of consent, biometric privacy, data retention, provider retention, and derived-weight distribution.

Before recording, obtain informed consent from each signer, avoid bystanders, use pseudonymous IDs, limit access to raw video, and establish deletion/retention terms. Consent to research collection does not automatically authorize public media release, public model weights, or third-party LLM processing. Never send sensitive sign streams to an API without reviewing its terms. For consequential communication, use a qualified interpreter; this tool is not a replacement.

Next, verify a real ISL lexicon with fluent signers; acquire data under explicit terms or run the provided recorder with consent; create signer-exclusive splits; and execute the fixed-seed grid on suitable hardware. Publish measured metrics, latency/device information, and real confusion-based error analysis only after those runs. Conduct usability and bias reviews with Deaf ISL users, calibrate repeat requests on held-out signers, and expand to continuous signing only with a suitable sequence dataset and evaluation protocol.

## References

1. D. Li, C. Rodriguez, X. Yu, and H. Li. “Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison.” *IEEE Winter Conference on Applications of Computer Vision (WACV)*, 2020. [Paper and citation](https://openaccess.thecvf.com/content_WACV_2020/html/Li_Word-Level_Deep_Sign_Language_Recognition_From_Video_A_New_Large-Scale_Dataset_and_Methods_Comparison_WACV_2020_paper.html).
2. Google AI Edge. “Holistic Landmarker guide.” [MediaPipe documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker).
3. S. Hochreiter and J. Schmidhuber. “Long Short-Term Memory.” *Neural Computation*, 9(8):1735–1780, 1997. [DOI: 10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735).
4. A. Vaswani et al. “Attention Is All You Need.” *Advances in Neural Information Processing Systems*, 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
5. S. Yao et al. “ReAct: Synergizing Reasoning and Acting in Language Models.” *International Conference on Learning Representations*, 2023. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629).
