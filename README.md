# Real-Time Indian Sign Language Recognition with an Agentic Sentence Layer

An end-to-end, research-oriented starter for webcam-based Indian Sign Language (ISL) recognition. The pipeline extracts hand and body landmarks, classifies short sign clips, and turns confident predictions into readable English with an optional LLM and speech output.

> **Status:** Project scaffold and implementation are under construction. No dataset has yet been downloaded and no model has been trained; this repository does not claim recognition metrics or a working pretrained checkpoint.

## Quick start

Use Python 3.10–3.12 for current MediaPipe/PyTorch wheel availability:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[app,train,dev]"
```

See the data instructions, training and deployment documentation below once the pipeline is complete. API keys belong in environment variables, never in source control.
