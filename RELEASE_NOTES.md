BloodMNIST 128 project: DenseNet121 features plus standard GA selection.

Measured SVM test accuracy: 98.74% with
241 selected features, versus 98.92%
with all 1,024 features, on the same 3,421 test images.

The self-contained notebook includes real outputs and automatic downloads.
Default mode verifies the shared checkpoint and runs inference. Optional modes
rerun GA, extract from raw images, or retrain the full project.

Exact training/validation image overlaps were excluded; the official test set
is unchanged. Selection used validation data only. Dataset: BloodMNIST from
MedMNIST+, CC BY 4.0. The dataset asset is a byte-identical project mirror of
the official Zenodo archive; see DATA_ATTRIBUTION.md.
