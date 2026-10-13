# Dataset sources, status, and licensing

The official WLASL v0.3 **annotation JSON** was downloaded from the upstream repository into the ignored local file `data/raw/WLASL_v0.3.json` (11,932,637 bytes). It contains labels and video URLs/IDs, not the videos needed to train a model. No video or image samples are included in Git. INCLUDE and Kaggle samples were not downloaded; the first-party download/license could not be verified, and Kaggle dataset discovery was blocked in this environment. These corpora have distinct access conditions and/or third-party video rights. Use the original provider's current terms and obtain any required access before training or redistribution.

| Dataset | Language/use | Source | What is available here | License / access note |
| --- | --- | --- | --- | --- |
| INCLUDE | Indian Sign Language (ISL) word-level signs; preferred in-domain source | [INCLUDE paper/project search](https://scholar.google.com/scholar?q=INCLUDE+Indian+Sign+Language+dataset) | Not downloaded: the upstream landing/download URL and redistribution license could not be verified from this environment. Acquire from the authors' current official distribution and record its license and signer annotations. | Do not assume open redistribution from a paper or a third-party mirror. Dataset copies may have separate terms. |
| WLASL | **American** Sign Language (ASL), not ISL; optional cross-lingual engineering baseline only | [Official WLASL repository](https://github.com/dxli94/WLASL), [downloaded v0.3 metadata](https://raw.githubusercontent.com/dxli94/WLASL/ac00e6be631c1a2a486621b65f202219f3964d6b/start_kit/WLASL_v0.3.json) | The 11.9 MB upstream v0.3 metadata JSON is present locally (ignored by Git). Original video files still need to be acquired from the upstream sources; unavailable videos require the maintainers' request process. | The official repository includes the **Computational Use of Data Agreement (C-UDA 1.0)** and states academic/computational use only; read and agree to the upstream terms before downloading/using videos. Each video may retain separate copyright. WLASL is not ISL training data. |
| Kaggle ISL datasets | Candidate ISL alphabet/word image/video corpora; dataset identity varies by uploader | [Kaggle dataset search](https://www.kaggle.com/datasets?search=indian+sign+language) | Not downloaded: Kaggle's dataset discovery/download endpoints were blocked in this environment, and no exact uploader, dataset card, or license could be verified. | Choose a specific dataset only after checking the uploader, license, signer metadata, and whether labels map to ISL. |

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
