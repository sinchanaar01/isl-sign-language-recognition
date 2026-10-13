# Interview notes

1. **Why use landmarks instead of raw video pixels?**  
   Landmarks provide a compact representation of hand and body geometry and make temporal models smaller. They trade away appearance/context and inherit pose-estimation failures; this project has no measured accuracy advantage to claim.

2. **Why insist on a signer-independent split?**  
   Consecutive clips from the same person share signing style and appearance. Keeping each signer in exactly one partition reduces identity leakage and better tests generalization to an unseen signer.

3. **Why include pose with hands?**  
   Pose can supply arm/body context for signs. The hands-only ablation wrist-centers each hand and scales it by wrist-to-middle-MCP distance so it does not use shoulder coordinates as a hidden pose feature.

4. **Why BiLSTM and BiGRU?**  
   Both recurrent networks aggregate temporal context in both directions over a short isolated clip. GRU has a simpler gated state; this repository has not yet run a real-data comparison.

5. **Why a Transformer, and did it beat the LSTM?**  
   Self-attention is a different way to model relationships across sampled frames and can be compared under the same signer split. No training data/checkpoint is available, so there is no measured winner.

6. **Why a TCN?**  
   Dilated temporal convolutions provide a convolutional sequence baseline with a broad temporal receptive field. Its inclusion lets the experiment compare recurrent, attention, and convolutional approaches.

7. **Why include MobileNetV2?**  
   It is a practical static-image baseline for sign images. It is not a fair direct comparison with video sequence models, and it requires explicit signer-prefixed image filenames to validate split separation.

8. **What does the preprocessing do?**  
   MediaPipe Holistic produces pose/hand landmarks. Frames are shoulder-centered and scaled, then uniformly resampled to a fixed length. Hands-only mode re-centers on each wrist and scales by a palm span.

9. **How are confidence and repeats handled?**  
   Recent class probabilities are averaged. A below-threshold prediction triggers `request_repeat(sign_index)` for one item, retaining the rest of the ordered sign buffer. The default threshold is a design setting, not calibrated yet.

10. **What tools can the agent call?**  
    `get_sign_buffer`, `request_repeat`, `lookup_vocabulary`, `propose_sentence`, `verify_sentence`, and `speak`. The loop is bounded to at most five tool actions; TTS is refused until sentence verification passes.

11. **How do you detect hallucinations?**  
    A lexical rule blocks content words absent from the source glosses. If enabled, an LLM judge adds a second check. Function words may be inserted for grammar. Neither mechanism guarantees semantic preservation or a valid ISL translation.

12. **Which LLM providers are supported?**  
    OpenAI-compatible Chat Completions and Anthropic Messages with provider/model/key selected from environment variables. Mock tests cover response parsing; live calls are not available without credentials.

13. **What does the offline evaluation mean?**  
    It runs a deterministic fallback on thirty authored gloss cases, including five low-confidence repeat cases. It measures exact string agreement and the local token guard on that fixed set—not recognition, human interpretation quality, or provider quality.

14. **What is the main limitation right now?**  
    There is no licensed ISL video corpus or consented recorded sample, so there is no trained recognizer, sign accuracy, per-class score, ablation result, or real camera latency result.

15. **What is required before deployment?**  
    Obtain consented/licensed data, verify vocabulary with fluent ISL signers, train and evaluate on signer-held-out samples, establish whether derived weights may be distributed, supply a trained checkpoint, configure a private HF Space and any provider secrets, then conduct a camera/upload/TTS usability review.
