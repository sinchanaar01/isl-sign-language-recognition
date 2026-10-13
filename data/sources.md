# Dataset sources, status, and licensing

The official WLASL v0.3 **annotation JSON** was downloaded from the upstream repository on 2026-10-13 into the ignored local file `data/raw/WLASL_v0.3.json` (11,932,637 bytes). Its SHA-256 and a rerunnable metadata audit are saved under `reports/runs/`. It contains labels and video URLs/IDs, not the videos needed to train a model. No video or image samples are included in Git. On 2026-10-13, `tools/download_data.py` attempted the candidate IIT Madras INCLUDE page; the connection was refused before the page, dataset size, media URLs, or license could be verified. Kaggle was not attempted because the Kaggle CLI and `~/.kaggle/kaggle.json` are absent. No INCLUDE/Kaggle media was downloaded. These corpora have distinct access conditions and/or third-party video rights. Use the original provider's current terms and obtain any required access before training or redistribution.

| Dataset | Language/use | Source | What is available here | License / access note |
| --- | --- | --- | --- | --- |
| INCLUDE | Indian Sign Language (ISL) word-level signs; preferred in-domain source | Candidate official host: [IIT Madras INCLUDE page](https://www.iitm.ac.in/donlab/INCLUDE); paper/project discovery [search](https://scholar.google.com/scholar?q=INCLUDE+Indian+Sign+Language+dataset) | Not downloaded. Automated request was refused by the remote endpoint; no page, file size, downloadable media URL, or license was verified. | **BLOCKED - needs human:** contact the dataset authors/maintainers for the current official URL and written access/license terms; verify consent and signer metadata. Do not assume open redistribution from a paper or third-party mirror. |
| WLASL | **American** Sign Language (ASL), not ISL; optional cross-lingual engineering baseline only | [Official WLASL repository](https://github.com/dxli94/WLASL), [downloaded v0.3 metadata](https://raw.githubusercontent.com/dxli94/WLASL/ac00e6be631c1a2a486621b65f202219f3964d6b/start_kit/WLASL_v0.3.json) | The 11.9 MB upstream v0.3 metadata JSON is present locally (ignored by Git). Original video files still need to be acquired from the upstream sources; unavailable videos require the maintainers' request process. | The official repository includes the **Computational Use of Data Agreement (C-UDA 1.0)** and states academic/computational use only; read and agree to the upstream terms before downloading/using videos. Each video may retain separate copyright. WLASL is not ISL training data. |
| Kaggle ISL datasets | Candidate ISL alphabet/word image/video corpora; dataset identity varies by uploader | [Kaggle dataset search](https://www.kaggle.com/datasets?search=indian+sign+language) | Not attempted by CLI: neither `kaggle` nor `~/.kaggle/kaggle.json` is present. | **BLOCKED - needs human:** configure API credentials, review a specific dataset card/license and signer metadata, then rerun the acquisition tool. |
| ISLRTC dictionary | ISL learning/reference resource, not an ML training corpus | [ISLRTC dictionary launch page](https://islrtc.nic.in/video/indian-sign-language-dictionary-launched-on-website/) | The official dictionary launch page was reachable, but an entry-level headword/media catalogue could not be accessed in a verifiable form here. Candidate lookup terms are listed in `data/target_glosses.txt`; no claim that all entries were individually verified. | **BLOCKED - needs human:** confirm every headword and sign in the official dictionary with a fluent ISL signer. English glosses are not ASL-equivalent signs. |

## Add a dataset to this pipeline

For licensed external training clips, create `data/raw/downloaded/clips.csv` with `video_path,label,signer_id`. Paths may be absolute or relative to that CSV. The signer ID is mandatory: do not derive one from class names or filenames unless the provider documents that mapping. Keep the user's own webcam clips under `data/raw/own/` as a separate live-demo test set; do not feed those recordings into preprocessing or training. Run:

```powershell
python -m tools.make_manifest --clips data/raw/downloaded/clips.csv --output data/processed/manifest.csv --report reports/data_manifest.json
python -m src.isl_recognition.data --source-csv data/raw/downloaded/clips.csv --output-dir data/processed
```

For static images, arrange images under `data/raw/static/{train,val,test}/{class_name}/`. The split must be signer-independent; `ImageFolder` does not enforce this for you. Keep source attribution, the exact license text, download date, and mapping/split provenance alongside private data. Keep all raw media and derived data out of Git.

For a private Kaggle dataset where its license permits use, install/configure the Kaggle CLI yourself, then download the exact verified dataset slug:

```powershell
kaggle datasets download -d OWNER/VERIFIED-DATASET-SLUG -p data/raw/kaggle --unzip
```

Automated acquisition can be attempted with `python -m tools.download_data`. It checks the INCLUDE page first, then searches Kaggle only when the API credentials and CLI are available. For Kaggle, only recognized permissive licenses and class-specific files matching `target_glosses.txt` are eligible; the tool does not download monolithic archives or accept click-through terms. It checks advertised size against available disk space before transfer. Any generated `clips.csv` leaves signer IDs blank unless canonical `video_path,label,signer_id` metadata explicitly supplies them. Blank signer IDs must be resolved with the source provider; they are never guessed from a filename.

Downloaded data is isolated under `data/raw/downloaded/`. The existing `data/raw/own/` recordings remain a separate live-demo test set and must not be mixed into model train/validation/test splits. The current `data/target_glosses.txt` remains a candidate vocabulary because no usable dataset was acquired; only successfully indexed classes may replace its entries on a future acquisition.

## Consent-based data collection

The local webcam recorder creates a separate held-out live-demo set. Record only with informed, documented consent; use pseudonymous signer IDs; do not capture bystanders or upload clips without permission. Collection target (20 candidate words, 30 clips per word, at least 3 signers) is a plan, not an achieved count. A fluent ISL signer must verify each sign and variant before collection.

```powershell
python -m tools.record_clips --signer-id signer_01
# Enter one label per clip. After recording: r=replay, d=delete, q=quit.
```

The default output is `data/raw/own/` and `data/raw/own/clips.csv`. Do not run this live-demo CSV through the manifest validator or training pipeline.

## Automated acquisition attempt: 2026-10-13T17:22:30.678531+00:00

- Free disk before attempt: 71574605824 bytes.
- INCLUDE: **blocked**.
  Source page: https://www.iitm.ac.in/donlab/INCLUDE
  Result: Official source request failed: <urlopen error [WinError 10061] No connection could be made because the target machine actively refused it>
- Kaggle: **blocked**.
  Result: Missing `~/.kaggle/kaggle.json`. Configure Kaggle API credentials following https://github.com/Kaggle/kaggle-api#api-credentials.
- Manual action: Contact INCLUDE maintainers for an official media URL and license, or configure Kaggle API credentials, verify an exact dataset card/license and signer metadata, then rerun `python -m tools.download_data --kaggle-slug OWNER/SLUG`. See data/sources.md.
