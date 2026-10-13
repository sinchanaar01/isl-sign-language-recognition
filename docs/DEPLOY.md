# Deploy to Hugging Face Spaces (Docker)

## Current deployment state

**BLOCKED - needs human.** No Hugging Face token is configured, no trained checkpoint exists, and the data licenses/sign consent do not permit any model distribution decision yet. The Space can be built from this Docker configuration, but without a licensed checkpoint both webcam and upload modes explicitly disable predictions. Do not deploy it as a working recognizer until the checklist below is completed.

## Before publishing

1. Collect or obtain real, signer-consented ISL data; record the license and permissions in `data/sources.md`.
2. Train/evaluate the models and inspect `models/MODEL_CARD.md`, `models/class_map.json`, and the per-run records in `reports/runs/`.
3. Ask the dataset provider/licensor whether distributing this derived checkpoint publicly is permitted. Participant consent for public model hosting is separately required.
4. If redistribution is **not** allowed, do not add weights to the Space Git repository. Implement an authenticated private artifact download to a persistent/private storage location before deployment; no such downloader is implemented yet, and a public Space with no weights must remain a UI-only preview.

## Click-by-click Docker Space setup

1. Sign in to [Hugging Face](https://huggingface.co/) and open [Create a new Space](https://huggingface.co/new-space).
2. Choose an owner and unique Space name.
3. Select **Docker** as the Space SDK and choose **Private** while reviewing the dataset and model license/consent. Public visibility is a deliberate later decision.
4. Create the Space. Its initial `README.md` must retain the Docker Space YAML metadata (`sdk: docker`, `app_port: 7860`); this repository already contains the required header.
5. On your machine, clone the newly created Space repository. Add this project as a remote or copy its reviewed source files; do not copy `data/raw`, private clips, API keys, MLflow runs, or raw WLASL videos.
6. If and only if the model license and signer permissions allow weight redistribution, add the trained checkpoint with Git LFS:

   ```powershell
   git lfs install
   git lfs track "models/*.pt"
   git add .gitattributes
   git add -f models/best_landmark.pt
   git add README.md Dockerfile .dockerignore pyproject.toml requirements.txt src app data/target_glosses.txt
   git commit -m "Deploy licensed ISL recognition demo"
   git push
   ```

   Include the matching model card and class-map JSON. Verify the Space commit contains no media, signer identifiers, secrets, or unlicensed weights before pushing. The normal project `.gitignore` deliberately excludes model weights.
7. Open the Space's **Settings** tab. Under **Variables and secrets**, add only the provider key you use:
   - Secret `OPENAI_API_KEY`, variable `LLM_PROVIDER=openai`, optional `OPENAI_MODEL`.
   - Or secret `ANTHROPIC_API_KEY`, variable `LLM_PROVIDER=anthropic`, optional `ANTHROPIC_MODEL`.
   
   Never place a key in Docker build args, README, source, or a public repository. The offline fallback needs no secret.
8. Select appropriate Space hardware in **Settings → Hardware**. A CPU Space is enough for the UI but may have slow MediaPipe inference; select GPU only after measuring the actual workload and reviewing cost.
9. Wait for the Docker build in the Space **Logs** tab. It should serve Streamlit on port `7860` and expose the `/_stcore/health` endpoint. Resolve build/runtime errors from the logs, not by ignoring failed health checks.
10. Open the Space through its HTTPS URL. First test **Record and test** with a short consenting test video and the licensed checkpoint. Then test camera permission, WebRTC reachability, sign-buffer display, repeat request, agent trace, and browser TTS. If WebRTC is blocked by network policy, use the upload mode or configure a compliant TURN service.
11. Check the Space's storage, logs, and provider terms. Raw uploaded video is held temporarily for inference by the application and removed from its local temporary file after processing; hosted infrastructure may have separate operational logs/retention. Do not upload sensitive video without consent and a retention review.

## No checkpoint allowed in public Git

If the derived weights cannot be redistributed, leave model weights out of Git and do not set a public URL for them. A private artifact downloader, access-control plan, and lifecycle for user-uploaded clips must be implemented and reviewed first. Setting only `ISL_MODEL_PATH` does not fetch a model; it selects a file already present in the container.
