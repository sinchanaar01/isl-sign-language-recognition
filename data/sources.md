# Dataset sources, status, and licensing

The official WLASL v0.3 **annotation JSON** was downloaded from the upstream repository on 2026-10-13 into the ignored local file `data/raw/WLASL_v0.3.json` (11,932,637 bytes). Its SHA-256 and a rerunnable metadata audit are saved under `reports/runs/`. It contains labels and video URLs/IDs, not the videos needed to train a model. No video or image samples are included in Git. INCLUDE and Kaggle samples were not downloaded; the first-party download/license could not be verified, and Kaggle dataset discovery was blocked in this environment. These corpora have distinct access conditions and/or third-party video rights. Use the original provider's current terms and obtain any required access before training or redistribution.

| Dataset | Language/use | Source | What is available here | License / access note |
| --- | --- | --- | --- | --- |
| INCLUDE | Indian Sign Language (ISL) word-level signs; preferred in-domain source | [INCLUDE paper/project search](https://scholar.google.com/scholar?q=INCLUDE+Indian+Sign+Language+dataset) | Not downloaded: the upstream landing/download URL and redistribution license could not be verified from this environment. Acquire from the authors' current official distribution and record its license and signer annotations. | **BLOCKED - needs human:** obtain from the authors and verify license, consent, and signer metadata. Do not assume open redistribution from a paper or third-party mirror. |
| WLASL | **American** Sign Language (ASL), not ISL; optional cross-lingual engineering baseline only | [Official WLASL repository](https://github.com/dxli94/WLASL), [downloaded v0.3 metadata](https://raw.githubusercontent.com/dxli94/WLASL/ac00e6be631c1a2a486621b65f202219f3964d6b/start_kit/WLASL_v0.3.json) | The 11.9 MB upstream v0.3 metadata JSON is present locally (ignored by Git). Original video files still need to be acquired from the upstream sources; unavailable videos require the maintainers' request process. | The official repository includes the **Computational Use of Data Agreement (C-UDA 1.0)** and states academic/computational use only; read and agree to the upstream terms before downloading/using videos. Each video may retain separate copyright. WLASL is not ISL training data. |
| Kaggle ISL datasets | Candidate ISL alphabet/word image/video corpora; dataset identity varies by uploader | [Kaggle dataset search](https://www.kaggle.com/datasets?search=indian+sign+language) | Not downloaded: Kaggle's dataset discovery/download endpoints were blocked in this environment, and no exact uploader, dataset card, or license could be verified. | Choose a specific dataset only after checking the uploader, license, signer metadata, and whether labels map to ISL. |
| ISLRTC dictionary | ISL learning/reference resource, not an ML training corpus | [ISLRTC dictionary launch page](https://islrtc.nic.in/video/indian-sign-language-dictionary-launched-on-website/) | The official dictionary launch page was reachable, but an entry-level headword/media catalogue could not be accessed in a verifiable form here. Candidate lookup terms are listed in `data/target_glosses.txt`; no claim that all entries were individually verified. | **BLOCKED - needs human:** confirm every headword and sign in the official dictionary with a fluent ISL signer. English glosses are not ASL-equivalent signs. |

## Add a dataset to this pipeline

For labeled clips, create a CSV with `video_path,label,signer_id`. Paths may be absolute or relative to that CSV. The signer ID is mandatory: do not derive one from class names or filenames unless the provider documents that mapping. Run:

```powershell
python -m src.isl_recognition.data --source-csv data/raw/clips.csv --output-dir data/processed
```

For static images, arrange images under `data/raw/static/{train,val,test}/{class_name}/`. The split must be signer-independent; `ImageFolder` does not enforce this for you. Keep source attribution, the exact license text, download date, and mapping/split provenance alongside private data. Keep all raw media and derived data out of Git.

For a private Kaggle dataset where its license permits use, install/configure the Kaggle CLI yourself, then download the exact verified dataset slug:

```powershell
kaggle datasets download -d OWNER/VERIFIED-DATASET-SLUG -p data/raw/kaggle --unzip
```

The repository intentionally does not guess a Kaggle slug or automate acceptance of dataset terms.

## Consent-based data collection

The local webcam recorder and manifest validator support consented collection. Record only with informed, documented consent; use pseudonymous signer IDs; do not capture bystanders or upload clips without permission. Collection target (20 candidate words, 30 clips per word, at least 3 signers) is a plan, not an achieved count. A fluent ISL signer must verify each sign and variant before collection.

```powershell
python -m tools.record_clips --signer-id signer_01
# Enter one label per clip. After recording: r=replay, d=delete, q=quit.
python -m tools.make_manifest --clips data/raw/clips.csv
```

The validator checks that videos can be opened/read by OpenCV, requires at least three signers, makes a signer-exclusive train/validation/test split, and fails if any class is absent from train. Inspect `reports/data_manifest.json` and the generated manifest before preprocessing.
