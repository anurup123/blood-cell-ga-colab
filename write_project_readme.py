"""Write the project handoff from measured outputs, never from expected scores."""
import importlib.metadata
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
summary = json.loads((ROOT / 'run/run_complete.json').read_text())
ga = json.loads((ROOT / 'run/ga_result.json').read_text())
comparison = pd.read_csv(ROOT / 'run/classifier_comparison.csv')
test = comparison.query("split == 'test'").pivot(index='classifier', columns='features', values='accuracy')
rows = []
for name in ['SVM', 'KNN', 'RandomForest', 'NeuralHead']:
    a, b = test.loc[name, ['all', 'ga_selected']]
    rows.append(f'| {name} | {100*a:.2f}% | {100*b:.2f}% | {100*(b-a):+.2f} |')
table = '\n'.join(rows)
text = f'''# Blood-cell classification with DenseNet and Genetic Algorithm feature selection

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anurup123/blood-cell-ga-colab/blob/main/Blood_Cell_GA_Colab_Kaggle.ipynb)

**Project result: {100*summary['svm_ga_selected_test_accuracy']:.2f}% test accuracy
using {summary['features_after']} of {summary['features_before']} deep features.**
The matched all-feature SVM scores {100*summary['svm_all_test_accuracy']:.2f}%.

Share [Blood_Cell_GA_Colab_Kaggle.ipynb](Blood_Cell_GA_Colab_Kaggle.ipynb).
It contains all code, numbered instructions, and actual executed outputs.
The trained model, fitted classifiers, feature mask, scaler, and supporting data
download automatically from a public release. No companion Python files or
GitHub login are needed. This is an educational project using standard methods;
no research novelty is claimed.

## What the project does

An image of one blood cell enters DenseNet121. Its global pooling layer produces
1,024 deep features. A standard binary genetic algorithm selects a subset, and
an SVM predicts one of eight blood-cell types. The classes are basophils,
eosinophils, erythroblasts, immature granulocytes, lymphocytes, monocytes,
neutrophils, and platelets.

The feature mask reduces the downstream classifier's input dimensions by
**{summary['feature_reduction_percent']:.2f}%**. DenseNet still computes all 1,024 features;
this is not a claim of equivalent end-to-end inference speedup.

## Run it

1. Open the notebook in Colab, or import it into Kaggle and enable Internet.
2. Leave `RUN_MODE = "verify"` and use **Run all** for checkpoint verification
   and example image inference. A GPU is optional for this mode.
3. If package installation asks for a kernel/session restart, restart once
   and run all again.

For full training, select a GPU and set these values in notebook step 2:

```python
RUN_MODE = "train"
EXPERIMENT_NAME = "my_blood_training_001"
```

Use a new experiment name for a fresh run, restart the kernel, and run all.
The notebook downloads the roughly 569 MB dataset, trains the model from ImageNet
initialization, extracts aligned features, runs GA, and evaluates the classifiers.
**Step 20 shows the new experiment's results.** Earlier tables verify the shared
checkpoint. Fresh training can give different scores.

Other modes are `ga` (rerun selection and classifiers from saved features) and
`reextract` (use the checkpoint to extract from images and rerun selection).
After an experiment, the image-prediction helper uses that new model and mask.
Rerun the optional image-upload cell to use it.

## Measured test results

All conditions use the same **3,421 test images**. The GA-selected SVM predicts
**{summary['svm_ga_selected_correct']} correctly**, versus
**{summary['svm_all_correct']}** with all features.

| Classifier | All features | GA-selected features | Change (percentage points) |
| --- | ---: | ---: | ---: |
{table}

The original trained DenseNet image classifier scores
**{100*summary['densenet_test_accuracy']:.2f}%**. Its original head's predictions on
extracted features were checked against image-model predictions before selection.
SVM is the primary comparison because SVM validation accuracy supplies GA fitness.
The other classifiers are additional matched comparisons. Balanced accuracy,
macro F1, per-class results, and confusion matrices are included in the outputs.

This is one held-out benchmark result. GA is not guaranteed to improve every
classifier or repeat training run. Scores from BloodMNIST and skin-lesion data
are different tasks and are not a direct comparison of medical model quality.

## Dataset and evaluation

The project uses BloodMNIST 128 from [MedMNIST+](https://zenodo.org/records/10519652).
The complete archive has 17,092 images. Integrity checking found exact image
overlaps in the provided splits: 3 train/validation, 5 train/test, and 1
validation/test. The loader removes the corresponding 8 training images and
1 validation image, preserving the official test set unchanged.

The resulting split is **11,951 train / 1,711 validation / 3,421 test**, with no
exact pixel duplicates shared across splits. Patient IDs are not provided in
this benchmark; these scores do not establish generalization to unseen patients.
The project classifies cell types and is not a clinical diagnosis system.

Scaling is fitted on training features only. Model checkpoints and SVM C are
chosen using validation data. GA receives training/validation arrays only, and
its mask is saved and hashed before the final test comparisons.

## Standard method and configuration

- DenseNet121 with ImageNet initialization, standard ImageNet normalization,
  random flips/rotations, a 256-unit head and 8-class softmax.
- Two head-warmup epochs, then at most 20 fine-tuning epochs, with validation-based
  early stopping and best-checkpoint restoration. BatchNorm statistics stay frozen.
- SVM RBF kernel; C selected from 0.1, 1 and 10 using validation accuracy.
  The chosen C was **{ga['svm_C']}** and is identical for the matched comparisons.
- Standard GA: 18 candidates, 20 generations, tournament selection, uniform
  crossover, bit mutation and two elites. The all-feature candidate is included.
- Fitness: `0.99 × validation accuracy + 0.01 × fraction of features removed`.
- The published search evaluated **{ga['unique_masks_evaluated']} distinct masks**.
  The selected mask's validation accuracy was **{100*ga['validation_accuracy']:.2f}%**.

## Files and saving your work

The notebook includes every function. `pipeline.py` contains the same code for
local development. `results/` holds measured tables and verification records.
The GitHub Release holds the model bundle and a checksum-verified dataset mirror.
The official dataset source remains Zenodo.

In Colab, files are under `/content/blood_cell_ga`; in Kaggle, under
`/kaggle/working/blood_cell_ga`. At the end, the notebook creates downloadable ZIPs
for verified results and, when requested, the new experiment's checkpoint and
classifiers. Save them before the cloud session ends.

The notebook outputs are produced in fresh Jupyter kernels on the project server.
Hosted Colab/Kaggle execution is not claimed. The notebook provides installation,
GPU checks and paths for both platforms.

## Attribution

BloodMNIST is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md).
MedMNIST: Yang et al., Scientific Data 10, 41 (2023), and Yang, Shi and Ni,
ISBI (2021). Source images: Acevedo et al., Data in Brief 30, 105474 (2020),
[DOI 10.1016/j.dib.2020.105474](https://doi.org/10.1016/j.dib.2020.105474).
'''
(ROOT / 'README.md').write_text(text)
notes = f'''BloodMNIST 128 project: DenseNet121 features plus standard GA selection.

Measured SVM test accuracy: {100*summary['svm_ga_selected_test_accuracy']:.2f}% with
{summary['features_after']} selected features, versus {100*summary['svm_all_test_accuracy']:.2f}%
with all 1,024 features, on the same 3,421 test images.

The self-contained notebook includes real outputs and automatic downloads.
Default mode verifies the shared checkpoint and runs inference. Optional modes
rerun GA, extract from raw images, or retrain the full project.

Exact training/validation image overlaps were excluded; the official test set
is unchanged. Selection used validation data only. Dataset: BloodMNIST from
MedMNIST+, CC BY 4.0. The dataset asset is a byte-identical project mirror of
the official Zenodo archive; see DATA_ATTRIBUTION.md.
'''
(ROOT / 'RELEASE_NOTES.md').write_text(notes)
packages = ['tensorflow', 'keras', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'Pillow', 'joblib', 'nbformat', 'nbclient', 'ipykernel']
(ROOT / 'requirements.txt').write_text('\n'.join(f'{name}=={importlib.metadata.version(name)}' for name in packages) + '\n')
print('Wrote README, release notes, and tested requirements.')
