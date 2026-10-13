# Project status and human actions

**Current state:** implementation and test scaffolding are present, but there is no licensed/consented ISL media, no trained checkpoint, no real recognition result, and no public deployment. Recognition and latency numbers are intentionally not reported. WLASL audit data is ASL annotation metadata only; it is not local training data.

| Phase | Status | Evidence / blocker |
| --- | --- | --- |
| 1. Data | **BLOCKED - needs human** | Recorder and validator are implemented. A dataset license/access route is unverified, target words need fluent-signer verification, and no consented clips are available. |
| 2. Models and training | **BLOCKED - needs human** | Preprocessing, four temporal baselines, static baseline, seeded grid, MLflow, and Colab path are implemented. No real clips were processed and no model was trained. |
| 3. Agent | **DONE** | Bounded tool loop, two provider adapters, offline fallback, JSONL traces, mocked tests, and the saved authored-case evaluation are present. No live provider call was made. |
| 4. App | **BLOCKED - needs human** | Webcam and upload UI are implemented. Cannot validate predictions, real FPS/latency, or end-to-end behavior without a checkpoint, test clips, and a human camera session. |
| 5. Deployment | **BLOCKED - needs human** | Docker Space metadata/configuration and click-by-click guide are present. No Hugging Face credential, trained model, or permission to distribute derived weights is available. |
| 6. Documentation | **DONE** | README, technical report, decisions, deployment instructions, and interview notes document actual evidence and limitations. |
| 7. Quality gate | **DONE** | Repository-wide Ruff and Black checks pass; all tests pass on supported Python 3.11.9 with a saved-workspace coverage report; GitHub Actions workflow added. New `NDArray[np.float32]` and `ArrayLike` annotations are validated against existing call sites. PyTorch factory/latency model signatures remain unresolved because their optional, lazily imported runtime types cannot be named safely without changing import behavior. |

## Exact steps to unblock data and model evaluation

1. **Verify source rights and language labels.** Contact the INCLUDE maintainers for the current access procedure/license, or identify a specific Kaggle publisher and inspect that dataset's license and signer provenance. Ask a fluent ISL signer to verify every word and regional variant in `data/target_glosses.txt`. Record decisions in `data/sources.md`; do not substitute ASL.
2. **Install on Python 3.11 and record consented clips.** Use an actual signer, informed consent, a private machine, and pseudonymous IDs. Repeat the recorder for every signer and each label; the current plan is 20 verified signs, 30 clips per label, and at least three signers. Review or delete every clip in the recorder UI.

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -e ".[app,train,dev]"
   python -m tools.record_clips --signer-id signer_01
   # Repeat with distinct pseudonymous IDs, e.g. signer_02 and signer_03.
   ```

3. **Validate and preprocess real recordings.** Fix every corrupt/missing clip or class-coverage issue reported before training.

   ```powershell
   python -m tools.make_manifest --clips data/raw/clips.csv
   python -m src.isl_recognition.data --source-csv data/raw/clips.csv --output-dir data/processed
   ```

4. **Train/evaluate.** On hardware with compatible dependencies, run the full grid below. It is a substantial experiment. If local compute or install compatibility prevents training, run `notebooks/train_colab.ipynb` on Colab with the real consented dataset in private Drive storage. Save the generated per-run logs, ablation table, model card, and checkpoint; do not copy private clips into the repository.

   ```powershell
   python -m src.isl_recognition.train_landmarks --manifest data/processed/manifest.csv --architectures lstm gru transformer tcn --seeds 42 43 44 --modalities hands hands_pose --augmentations on off --sequence-lengths 16 32 48 --epochs 40
   ```

5. **Test the application manually.** Start `streamlit run app/streamlit_app.py`; first use a consented upload clip, then webcam after granting permission. Review errors/confidence, repeat requests, TTS, and the agent trace. Record measurements only from a real run and its actual device.
6. **Deploy only after a rights review.** Follow [`DEPLOY.md`](DEPLOY.md), create a private HF Docker Space, and configure provider keys only as host secrets. Include a checkpoint only after the source license and signer consent explicitly permit derived-weight distribution. If not, a secure authenticated artifact retrieval path must be implemented first.

## Exact quality-check commands

```powershell
python -m pip install -e ".[dev]"
python -m ruff check src tools app tests
python -m black --check src tools app tests
python -m pytest
```

The pytest coverage table prints to the terminal and is also written to `reports/coverage.xml`. Coverage is a code-path statistic for the current test suite, not recognition accuracy or an experimental model result.
