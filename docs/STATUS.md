# Project status and human actions

**Current state:** the data-acquisition attempt and live segmentation path are implemented, but no licensed ISL video was acquired, no model was trained, and no public deployment exists. WLASL audit data is ASL annotation metadata only. Recognition accuracy, classifier latency, and camera FPS are intentionally not reported. The user's own recordings are reserved for a separate held-out live-demo set.

| Phase | Status | Evidence / blocker |
| --- | --- | --- |
| 1. Data | **BLOCKED - needs human** | `tools.download_data` tried INCLUDE but the host refused the request before media, size, or license could be inspected; Kaggle CLI/config are absent. No media was acquired and `data/target_glosses.txt` remains an unverified candidate vocabulary. The actual acquisition audit is `reports/runs/data_acquisition_20261013T2244+0530.json`. |
| 2. Models and training | **BLOCKED - needs human** | Preprocessing, MobileNetV2 and four temporal baselines, the seeded grid, MLflow, and Colab path exist. Preprocessing was attempted against the downloaded-manifest path and failed because no `data/raw/downloaded/clips.csv` exists. No real clips were processed and no model was trained. |
| 3. Agent | **DONE (offline path only)** | Bounded tool loop, OpenAI-compatible/Anthropic adapters, offline fallback, JSONL traces, and mocked tests are present. The independent 30-case run yielded 0/20 flagged generated sentences, repeat precision 10/10, and recall 10/10 using deterministic offline mode; no live LLM provider call was made. This is not a semantic translation evaluation. |
| 4. App | **BLOCKED - needs human** | The local OpenCV motion-segmented app is implemented and the explicit missing-checkpoint guard was exercised. The Streamlit webcam/upload app is secondary. No checkpoint/camera run exists, so predictions, FPS, camera-to-displayed-word latency, and full app integration remain unvalidated. |
| 5. Deployment | **BLOCKED - needs human** | Docker Space metadata/configuration and click-by-click guide are present. No Hugging Face credential, trained model, or permission to distribute derived weights is available. |
| 6. Documentation | **DONE** | README, technical report, decisions, deployment instructions, and interview notes document actual evidence and limitations. |
| 7. Quality gate | **DONE** | 44 tests passed on Python 3.11.9 (35% measured test coverage); Ruff passed on Python 3.11.9 and Black passed in the installed dev-tool environment. The GitHub Actions workflow is present. PyTorch factory/latency model signatures remain unresolved because their optional, lazily imported runtime types cannot be named safely without changing import behavior. |

## Exact steps to unblock data and model evaluation

1. **Acquire licensed external ISL data.** Contact the INCLUDE maintainers for an official media URL, access instructions, license, and signer metadata. Alternatively install/configure Kaggle CLI credentials, verify a specific dataset card/license/signer provenance, and run the downloader:

   ```powershell
   python -m pip install kaggle
   # Configure ~/.kaggle/kaggle.json following Kaggle's official instructions.
   python -m tools.download_data --kaggle-slug OWNER/VERIFIED-DATASET-SLUG
   ```

   If the source supplies clips manually, place the provider-approved files under `data/raw/downloaded/` and create `data/raw/downloaded/clips.csv` with `video_path,label,signer_id`. Do not guess signer IDs. Ask a fluent ISL signer to verify every word and regional variant in `data/target_glosses.txt` before training.
2. **Validate and preprocess only the licensed external dataset.** Fix every corrupt/missing clip, missing signer ID, or class-coverage issue. Never use `data/raw/own/clips.csv` here.

   ```powershell
   python -m tools.make_manifest --clips data/raw/downloaded/clips.csv --output data/processed/manifest.csv --report reports/data_manifest.json
   python -m src.isl_recognition.data --source-csv data/raw/downloaded/clips.csv --output-dir data/processed
   ```

3. **Train/evaluate.** On hardware with compatible dependencies, run the full grid below. It is a substantial experiment. If local compute or install compatibility prevents training, run `notebooks/train_colab.ipynb` on Colab with the real licensed dataset in private storage. Save the generated per-run logs, ablation table, model card, and checkpoint; do not copy private clips into the repository.

   ```powershell
   python -m src.isl_recognition.train_landmarks --manifest data/processed/manifest.csv --architectures lstm gru transformer tcn --seeds 42 43 44 --modalities hands hands_pose --augmentations on off --sequence-lengths 16 32 48 --epochs 40
   ```

4. **Create an isolated demo set if desired.** Use actual signer consent and pseudonymous IDs; these clips are strictly for manual live-demo testing and are never included in the training manifest:

   ```powershell
   python -m tools.record_clips --signer-id signer_demo --manifest data/raw/own/clips.csv
   python -m src.isl_recognition.live
   ```

5. **Test the application manually.** First run the local OpenCV demo above, then use `streamlit run app/streamlit_app.py` for the secondary upload/webcam/agent/TTS UI. Record FPS and camera-to-displayed-word latency only from an actual camera run with the trained checkpoint.
6. **Deploy only after a rights review.** Follow [`DEPLOY.md`](DEPLOY.md), create a private HF Docker Space, and configure provider keys only as host secrets. Include a checkpoint only after the source license and signer consent explicitly permit derived-weight distribution. If not, a secure authenticated artifact retrieval path must be implemented first.

## Exact quality-check commands

```powershell
python -m pip install -e ".[dev]"
python -m ruff check src tools app tests
python -m black --check src tools app tests
python -m pytest
```

The pytest coverage table prints to the terminal and is also written to `reports/coverage.xml`. Coverage is a code-path statistic for the current test suite, not recognition accuracy or an experimental model result.
