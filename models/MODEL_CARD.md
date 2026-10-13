# Model card — not trained

**Status: BLOCKED - needs human.** There is no trained checkpoint in this repository. No accuracy, F1, latency, or model generalization claim is available.

The full-grid trainer writes this file after it has completed real signer-independent trials and selected the checkpoint with the best validation accuracy. The selected checkpoint and `class_map.json` are local artifacts; do not publish either until the dataset license and signer consent permit derived weights to be distributed.

To continue, collect/license clips, verify the vocabulary with a fluent ISL signer, run `tools.make_manifest`, extract features, then run the Colab or local experiment grid described in `README.md`.
