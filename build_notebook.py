"""Create a self-contained, cell-by-cell project notebook from the tested code."""
import ast
import hashlib
import json
from pathlib import Path
from pprint import pformat
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + '\n'))


def code(text, tag=None):
    cell = nbf.v4.new_code_cell(textwrap.dedent(text).strip() + '\n')
    if tag:
        cell.metadata['tags'] = [tag]
    cells.append(cell)


def source_functions(filename, names):
    source = (ROOT / filename).read_text()
    found = {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    return '\n\n\n'.join(found[name] for name in names)


summary = json.loads((ROOT / 'run/run_complete.json').read_text())
config = json.loads((ROOT / 'run/protocol.json').read_text())
asset_hash = hashlib.sha256((ROOT / 'assets_manifest.json').read_bytes()).hexdigest()
metadata = json.loads((ROOT / 'release_assets/model_bundle/model_metadata.json').read_text())
before = 100 * summary['svm_all_test_accuracy']
after = 100 * summary['svm_ga_selected_test_accuracy']
width = summary['features_before']
selected = summary['features_after']
change = summary['svm_change_percentage_points']

md(f"""
# Blood-cell classification with DenseNet and Genetic Algorithm feature selection

**Measured test accuracy: {after:.2f}% after GA selection, using {selected} of {width} deep features.**
The same SVM with all features scores **{before:.2f}%** on the same **3,421 test images**.
The difference is **{change:+.2f} percentage points**. These are actual project results,
not promised scores for every new training run.

This is a practical project using an existing CNN, a standard binary genetic
algorithm, and standard classifiers. It does not propose a new method.

**Workflow:** blood-cell image → DenseNet121 → 1,024 deep features → GA feature mask → SVM cell-type prediction.

Everything needed is in this file: code, instructions, and executed outputs.
Checkpoints and supporting files download automatically from the public GitHub
release. No companion Python files or GitHub token are needed.

## Start here

1. Open this file in **Google Colab**, or import it into **Kaggle Notebooks**.
2. On Kaggle, enable **Internet**. Select a GPU for full training; inference can also run on CPU.
3. Keep `RUN_MODE = "verify"` below and choose **Run all**, or run cells in order.
4. If package installation asks for a restart, restart once and run all again.
5. Read the computed results and try the example images. Retraining is optional.

The published outputs were executed on the project server in a fresh Jupyter
kernel. Hosted Colab/Kaggle execution is not claimed. The dependency setup and
file paths support both platforms.
""")
md("""
## 1. Install the tested packages

Run this first in a fresh session. An installation can take several minutes.
If an updated package was already imported by the notebook service, restart the
session/kernel once when asked, then run all again.
""")
code("""
import importlib.metadata as package_metadata
import shutil
import subprocess
import sys

required = {
    "tensorflow": "2.20.0", "keras": "3.13.2", "numpy": "2.2.6",
    "pandas": "2.3.3", "scipy": "1.16.3", "scikit-learn": "1.6.1",
    "Pillow": "11.3.0", "joblib": "1.6.0",
}
module_names = {"scikit-learn": "sklearn", "Pillow": "PIL"}
changed, imported = [], []
for package, wanted in required.items():
    try:
        current = package_metadata.version(package)
    except package_metadata.PackageNotFoundError:
        current = None
    if current != wanted:
        spec = f"{package}=={wanted}"
        if package == "tensorflow" and sys.platform.startswith("linux") and shutil.which("nvidia-smi"):
            spec = f"tensorflow[and-cuda]=={wanted}"
        changed.append(spec)
        if module_names.get(package, package) in sys.modules:
            imported.append(package)
if changed:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *changed])
    if imported:
        raise RuntimeError("Packages installed. Restart the session/kernel ONCE, then Run all again. Updated imports: " + ", ".join(imported))
print("Packages ready:")
for package in required:
    print(package, package_metadata.version(package))
""", 'setup')
md("""
## 2. Choose what to run

**Keep `verify` for your first run.** To retrain the whole project, use `train`,
give the experiment a new name, restart the kernel, and run all.

| Mode | What it does |
| --- | --- |
| `verify` | Download the trained checkpoint, recompute its SVM test scores, and demonstrate inference. |
| `ga` | Also rerun GA and the classifiers using saved, aligned deep features. |
| `reextract` | Download the images, re-extract features with the shared model, then rerun GA. |
| `train` | Download the images, train a fresh DenseNet from ImageNet initialization, extract features, and run GA. |

The first result table always verifies the **shared model**. If you select an
experiment mode, the later **New experiment results** section shows that run's
results separately. A new model may produce a different accuracy.
""")
code("""
import os
from pathlib import Path

RUN_MODE = "verify"  # Options: "verify", "ga", "reextract", "train"
EXPERIMENT_NAME = "my_blood_run_001"  # Use a new name for a fresh experiment.
BATCH_SIZE = 64  # Reduce to 32 or 16 if your GPU runs out of memory during training.

assert RUN_MODE in {"verify", "ga", "reextract", "train"}
assert EXPERIMENT_NAME and Path(EXPERIMENT_NAME).name == EXPERIMENT_NAME
if Path("/kaggle/working").exists():
    WORK_DIR = Path("/kaggle/working/blood_cell_ga")
elif Path("/content").exists():
    WORK_DIR = Path("/content/blood_cell_ga")
else:
    WORK_DIR = Path.cwd() / "blood_cell_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR = WORK_DIR / "verified_results"
RESULT_DIR.mkdir(exist_ok=True)
RUN_DIR = WORK_DIR / "experiments" / EXPERIMENT_NAME
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
print("Mode:", RUN_MODE)
print("Working folder:", WORK_DIR)
""")
md("## 3. Import libraries and configure the runtime")
code("""
import hashlib
import json
import stat
import time
import urllib.request
import zipfile

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import display, FileLink, Image as DisplayImage
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
""")
code('CLASS_NAMES = ' + pformat(metadata['class_names']) + '\n\nDEFAULT_CONFIG = ' + pformat(config) +
     '\n\nCONFIG = dict(DEFAULT_CONFIG, batch_size=BATCH_SIZE)\n')
code(source_functions('pipeline.py', ['file_digest', 'configure_runtime']) + '\n\nconfigure_runtime(CONFIG["seed"], CONFIG["cpu_threads"])\n')
md("""
## 4. Download helpers

These helpers are included in the notebook. They verify download hashes, resume
partial downloads, and reuse correct cached files. No edits are needed here.
""")
code(source_functions('cloud_io.py', ['sha256_file', 'download']))
code(source_functions('cloud_io.py', ['safe_extract', 'release_url', 'fetch_bundle']))
md("""
## 5. Download the checkpoint and project files

This downloads the model bundle, not the full image dataset. It contains the
trained DenseNet, fitted SVMs, scaler, GA mask, aligned features and labels,
example images, and measured results.
""")
code(f'''
REPOSITORY = "anurup123/blood-cell-ga-colab"
RELEASE = "v1.0.0"
manifest_url = f"https://raw.githubusercontent.com/{{REPOSITORY}}/{{RELEASE}}/assets_manifest.json"
manifest_path = download(manifest_url, WORK_DIR / "assets_manifest.json", "{asset_hash}")
asset_manifest = json.loads(manifest_path.read_text())
assert asset_manifest["repository"] == REPOSITORY and asset_manifest["release"] == RELEASE
BUNDLE_DIR = fetch_bundle(asset_manifest, WORK_DIR)
model_info = json.loads((BUNDLE_DIR / "model_metadata.json").read_text())
reference = model_info["summary"]
print("Dataset:", model_info["dataset"])
print("Backbone:", model_info["model"])
print("Selected features:", reference["features_after"], "of", reference["features_before"])
''')
md("""
## 6. Understand the dataset and split

BloodMNIST contains 17,092 microscopy images of eight blood-cell types. This
project uses the 128×128 images. The supplied benchmark splits contained exact
duplicate images across partitions, so the loader excludes **8 training rows
and 1 validation row**. The **3,421-image test set remains unchanged**.

The experiment therefore uses **11,951 training**, **1,711 validation**, and
**3,421 test** images, with zero exact pixel duplicates shared across splits.
Patient identifiers are not provided here; this is an image-level benchmark.
""")
code("""
audit = json.loads((BUNDLE_DIR / "data_audit.json").read_text())
split_manifest = pd.read_csv(BUNDLE_DIR / "split_manifest.csv")
included = split_manifest.loc[split_manifest["included"]]
display(pd.crosstab(included.class_name, included.split))
print("Remaining exact cross-split image overlaps:", audit["remaining_exact_image_duplicates_across_splits"])
print("Official test set unchanged:", audit["official_test_set_unchanged"])
""")
md("""
## 7. Load the aligned features, scaler and classifiers

Each feature row stays paired with its label and image ID. The scaler was fitted
on training features only. GA selected its mask using training/validation data;
the test set was reserved for the reported comparison.
""")
code("""
with np.load(BUNDLE_DIR / "aligned_features.npz", allow_pickle=False) as archive:
    saved_arrays = {key: archive[key] for key in archive.files}
saved_mask = np.load(BUNDLE_DIR / "ga_mask.npy", allow_pickle=False).astype(bool)
scaler = joblib.load(BUNDLE_DIR / "feature_scaler.joblib")
svm_all = joblib.load(BUNDLE_DIR / "svm_all.joblib")
svm_ga = joblib.load(BUNDLE_DIR / "svm_ga_selected.joblib")
inference_mask = saved_mask
assert saved_mask.shape == (reference["features_before"],) and saved_mask.sum() == reference["features_after"]
for split in ["train", "val", "test"]:
    assert len(saved_arrays[f"X_{split}"]) == len(saved_arrays[f"y_{split}"]) == len(saved_arrays[f"ids_{split}"])
    print(split, saved_arrays[f"X_{split}"].shape, "features; labels and image IDs aligned")
""")
md("""
## 8. Recompute the shared model's test accuracy

This cell makes predictions from the saved classifiers and computes the scores.
Both columns use the same test images and SVM settings; only the feature mask differs.
""")
code("""
test_scaled = np.asarray(scaler.transform(saved_arrays["X_test"]), dtype=np.float32)
rows = []
saved_test_predictions = {}
with threadpool_limits(limits=CONFIG["cpu_threads"]):
    for subset, classifier, selection in [("all", svm_all, np.ones_like(saved_mask)), ("ga_selected", svm_ga, saved_mask)]:
        prediction = classifier.predict(test_scaled[:, selection])
        labels = saved_arrays["y_test"]
        saved_test_predictions[subset] = prediction
        accuracy = float(accuracy_score(labels, prediction))
        np.testing.assert_allclose(accuracy, reference[f"svm_{subset}_test_accuracy"], rtol=0, atol=1e-12)
        rows.append({"Features": subset, "Feature count": int(selection.sum()), "Test images": len(labels),
                     "Correct": int((prediction == labels).sum()), "Accuracy (%)": 100 * accuracy,
                     "Balanced accuracy (%)": 100 * balanced_accuracy_score(labels, prediction),
                     "Macro F1": f1_score(labels, prediction, average="macro", zero_division=0)})
verification = pd.DataFrame(rows)
verification.to_csv(RESULT_DIR / "verified_svm_results.csv", index=False)
display(verification.round(4))
print(f"VERIFIED GA + SVM TEST ACCURACY: {verification.iloc[1]['Accuracy (%)']:.2f}%")
print(f"Feature reduction: {100 * (1 - saved_mask.mean()):.2f}%")
print(f"Accuracy change: {verification.iloc[1]['Accuracy (%)'] - verification.iloc[0]['Accuracy (%)']:+.2f} percentage points")
""", 'measured-results')
md("""
## 9. Per-class results and the complete measured comparison

The class report is recomputed from the selected SVM predictions. The four-classifier
table below it is the recorded full experiment; optional experiment cells can rerun
those models. SVM is the primary comparison because it supplies the GA fitness.
""")
code("""
report = classification_report(saved_arrays["y_test"], saved_test_predictions["ga_selected"],
                               labels=np.arange(8), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
display(pd.DataFrame(report).T.round(4))
comparison = pd.read_csv(BUNDLE_DIR / "classifier_comparison.csv")
comparison_table = comparison.query("split == 'test'").pivot(index="classifier", columns="features", values="accuracy") * 100
comparison_table["change (percentage points)"] = comparison_table["ga_selected"] - comparison_table["all"]
display(comparison_table.round(2))
print(f"Recorded original DenseNet test accuracy: {100 * reference['densenet_test_accuracy']:.2f}%")
print("GA removes inputs to the downstream classifier; DenseNet still computes all 1,024 deep features.")
""")
md("""
## 10. Define and load the trained DenseNet

ImageNet normalization is built into the model. Inputs are RGB images with values
from 0 to 255; do not divide by 255 again outside this model. The image augmentations
are active only during training.
""")
code(source_functions('pipeline.py', ['build_model']))
code("""
image_model, _ = build_model(image_size=128, weights=None, seed=CONFIG["seed"])
image_model.load_weights(BUNDLE_DIR / "densenet121.weights.h5")
feature_model = tf.keras.Model(image_model.input, image_model.get_layer("deep_features").output)
feature_model.jit_compile = False
print("Checkpoint loaded; ready for inference.")
print("Image shape:", image_model.input_shape, "Feature shape:", feature_model.output_shape)
""")
md("""
## 11. Predict from actual example images

The examples are unchanged test-image pixel arrays, one per class. This demonstrates
the full image → deep features → scaling → GA mask → SVM path. The overall test
accuracy above is based on all 3,421 images, not just these examples.
""")
code("""
def predict_images(paths):
    paths = [Path(path) for path in paths]
    rows = []
    for start in range(0, len(paths), 32):
        group = paths[start:start + 32]
        pixels = []
        for path in group:
            image = tf.keras.utils.load_img(path, color_mode="rgb", target_size=(128, 128), interpolation="bilinear")
            pixels.append(tf.keras.utils.img_to_array(image))
        batch = np.stack(pixels)
        features = feature_model.predict(batch, batch_size=32, verbose=0)
        scaled = np.asarray(scaler.transform(features), dtype=np.float32)
        with threadpool_limits(limits=CONFIG["cpu_threads"]):
            ga_prediction = svm_ga.predict(scaled[:, inference_mask])
            all_prediction = svm_all.predict(scaled)
        dense_prediction = image_model.predict(batch, batch_size=32, verbose=0).argmax(axis=1)
        for path, selected_label, all_label, dense_label in zip(group, ga_prediction, all_prediction, dense_prediction):
            rows.append({"Image": path.name, "GA + SVM": CLASS_NAMES[int(selected_label)],
                         "All-feature SVM": CLASS_NAMES[int(all_label)], "DenseNet": CLASS_NAMES[int(dense_label)]})
    return pd.DataFrame(rows)

example_paths = [BUNDLE_DIR / "examples" / item["filename"] for item in model_info["examples"]]
example_predictions = predict_images(example_paths)
example_predictions.insert(1, "Dataset label", [item["class_name"] for item in model_info["examples"]])
example_predictions.to_csv(RESULT_DIR / "example_predictions.csv", index=False)
display(example_predictions)
display(DisplayImage(filename=str(example_paths[0]), width=160))
""", 'image-inference')
md("""
## 12. Optional: use your own image

Use a cropped image of one blood cell. Upload it with the notebook's file panel
and enter its path. In Colab, you can instead enable the upload button below.
The default skips the dialog so Run all completes without waiting for input.
""")
code("""
MY_IMAGE_PATH = ""  # Example: "/content/blood_cell.png"
UPLOAD_WITH_COLAB_BUTTON = False
custom_paths = [Path(MY_IMAGE_PATH)] if MY_IMAGE_PATH else []
if UPLOAD_WITH_COLAB_BUTTON:
    try:
        from google.colab import files
    except ImportError as error:
        raise RuntimeError("On Kaggle, use its file panel and set MY_IMAGE_PATH.") from error
    custom_paths = [Path(name) for name in files.upload()]
if custom_paths:
    display(predict_images(custom_paths))
else:
    print("Optional upload skipped. Shared-checkpoint verification and example inference are complete.")
""")

# Optional experiment cells are appended below. All functions remain in the file.
md("""
## 13. Data-loading functions for optional experiments

The original image-level splits are retained, with the documented exact-overlap
rows excluded from training/validation. Row IDs remain attached to the data.
The image dataset shuffles paired images and labels only during training.
""")
code(source_functions('pipeline.py', ['load_data']))
code(source_functions('pipeline.py', ['image_dataset']))
md("""
## 14. Optional: download the full dataset

Only `reextract` and `train` need this roughly 569 MB download. The project mirror
is byte-identical to the official BloodMNIST 128 archive and is checked by SHA-256
and the official MD5. The original source is
[MedMNIST+ on Zenodo](https://zenodo.org/records/10519652), under CC BY 4.0.
""")
code("""
data = None
if RUN_MODE != "verify":
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    signature = {"mode": RUN_MODE, "config": CONFIG, "source_weights_sha256": model_info["exported_weights_sha256"]}
    signature_path = RUN_DIR / "experiment_config.json"
    if signature_path.exists():
        assert json.loads(signature_path.read_text()) == signature, "Choose a new EXPERIMENT_NAME for changed settings or a different mode."
    else:
        signature_path.write_text(json.dumps(signature, indent=2) + "\\n")
    (RUN_DIR / "protocol.json").write_text(json.dumps(CONFIG, indent=2) + "\\n")
if RUN_MODE in {"reextract", "train"}:
    info = asset_manifest["dataset"]
    dataset_path = download(release_url(asset_manifest, info["name"]), WORK_DIR / "data" / info["name"], info["sha256"], info["size"])
    data, experiment_audit = load_data(dataset_path, RUN_DIR)
else:
    print("Full dataset download skipped for mode:", RUN_MODE)
""")
md("""
## 15. Optional: train the original DenseNet architecture

The classifier head is warmed up for 2 epochs, then DenseNet is fine-tuned for
up to 20 epochs with early stopping. Checkpoints are selected by validation accuracy.
Training can take substantially longer on a cloud GPU than on the project server.
For a fresh run, use a new experiment name. An interrupted run is never silently
treated as completed training.
""")
code(source_functions('pipeline.py', ['train_model']))
code("""
experiment_model = image_model
if RUN_MODE == "train":
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Select a GPU, restart the kernel, and Run all before full training.")
    experiment_model = train_model(data, RUN_DIR, CONFIG)
elif RUN_MODE in {"ga", "reextract"}:
    shutil.copy2(BUNDLE_DIR / "densenet121.weights.h5", RUN_DIR / "best_model.weights.h5")
    print("Using the shared trained checkpoint.")
else:
    print("Backbone training skipped.")
""")
md("""
## 16. Optional: extract features and verify the original head

Extraction never shuffles. Features, labels, and image IDs are saved together.
The original head must give matching predictions from extracted features and
from the complete image model. This checks that extraction preserves the
trained model's predictions.
""")
code(source_functions('pipeline.py', ['extract_features']))
code("""
arrays = saved_arrays
if RUN_MODE in {"reextract", "train"}:
    arrays = extract_features(experiment_model, data, RUN_DIR, CONFIG["batch_size"])
elif RUN_MODE == "ga":
    for name in ["aligned_features.npz", "densenet_test_predictions.npz", "data_audit.json", "split_manifest.csv"]:
        shutil.copy2(BUNDLE_DIR / name, RUN_DIR / name)
    print("Using saved, verified features; no image extraction is needed.")
else:
    print("Optional feature extraction skipped.")
""")
md("""
## 17. Fit the scaler and choose SVM regularization on validation data

The scaler is fitted on training features only. SVM C is chosen from 0.1, 1, and
10 using validation accuracy. The same selected C is used before and after GA.
""")
code(source_functions('pipeline.py', ['fit_scaler_and_choose_svm']))
code("""
if RUN_MODE != "verify":
    experiment_scaler, scaled, chosen_c = fit_scaler_and_choose_svm(arrays, RUN_DIR, CONFIG)
    print("Validation-selected SVM C:", chosen_c)
else:
    print("Validation search skipped; shared classifiers are already fitted.")
""")
md("""
## 18. Standard Genetic Algorithm

Each bit indicates whether a deep feature is kept. This uses ordinary tournament
selection, uniform crossover, bit mutation, and elitism. The all-feature candidate
is included in the initial population. The fitness is:

`0.99 × validation accuracy + 0.01 × fraction of features removed`.

Accuracy is prioritized, with a small reward for fewer features. The GA receives
training and validation arrays only. Every population, including the final one,
is evaluated. This is a standard project implementation, not a novelty claim.
""")
code(source_functions('pipeline.py', ['run_binary_ga']))
code(source_functions('pipeline.py', ['lock_selection']))
md("## 19. Optional: run GA and lock the selected mask")
code("""
if RUN_MODE != "verify":
    selected_mask, ga_result, ga_history = run_binary_ga(scaled["train"], arrays["y_train"], scaled["val"], arrays["y_val"], chosen_c, CONFIG)
    lock_selection(selected_mask, ga_result, ga_history, RUN_DIR, CONFIG)
    display(ga_history.tail())
    print(json.dumps(ga_result, indent=2))
else:
    print("GA rerun skipped. The saved feature mask has been verified.")
""")
md("""
## 20. New experiment results: compare before and after GA

**This is where your new run's results appear** when `RUN_MODE` is `ga`, `reextract`,
or `train`. Each classifier is fitted on all features and on the selected subset
using matching settings. The primary comparison is SVM; KNN, random forest, and
the neural head are additional project comparisons.

The new neural heads keep their own learned weights. No unrelated model weights
are mixed after training. The test set is evaluated after the GA mask is locked.
With `verify`, this optional fitting is skipped because steps 8–11 already verify
and demonstrate the shared trained model.
""")
code(source_functions('pipeline.py', ['build_feature_head']))
code(source_functions('pipeline.py', ['evaluate_classifiers']))
code("""
new_summary = None
if RUN_MODE != "verify":
    new_comparison, new_summary = evaluate_classifiers(arrays, scaled, selected_mask, RUN_DIR, chosen_c, CONFIG)
    new_table = new_comparison.query("split == 'test'").pivot(index="classifier", columns="features", values="accuracy") * 100
    display(new_table.round(2))
    print(f"YOUR NEW RUN: GA + SVM test accuracy = {100 * new_summary['svm_ga_selected_test_accuracy']:.2f}%")
    print("Your run's files:", RUN_DIR)
    # Make the existing prediction helper use this new experiment from now on.
    image_model = experiment_model
    feature_model = tf.keras.Model(image_model.input, image_model.get_layer("deep_features").output)
    feature_model.jit_compile = False
    scaler = experiment_scaler
    inference_mask = selected_mask
    svm_all = joblib.load(RUN_DIR / "svm_all.joblib")
    svm_ga = joblib.load(RUN_DIR / "svm_ga_selected.joblib")
    print("Inference now uses YOUR new experiment. Rerun step 12 to predict your own image with it.")
else:
    print("No new training experiment requested. The shared-checkpoint scores are in step 8.")
""")
md("""
## 21. Save results and checkpoints

Download your results before ending the cloud session. In Colab, use the file
panel; on Kaggle, files under `/kaggle/working` are available as notebook outputs.
The verified-results ZIP contains the shared-model scores computed in this run.
An experiment mode also creates a separate ZIP with its trained weights, scaler,
SVMs, feature mask, and result tables.

These are benchmark results for blood-cell type classification. Higher accuracy
on BloodMNIST does not establish that it is a better medical model than the skin
classifier; the tasks and datasets differ. GA is not guaranteed to improve every
classifier or every fresh training run.
""")
code("""
verification_summary = {
    "dataset": "BloodMNIST 128", "test_images": len(saved_arrays["y_test"]),
    "shared_svm_all_accuracy": float(verification.iloc[0]["Accuracy (%)"] / 100),
    "shared_svm_ga_accuracy": float(verification.iloc[1]["Accuracy (%)"] / 100),
    "shared_selected_features": int(saved_mask.sum()), "mode": RUN_MODE,
}
(RESULT_DIR / "verification_summary.json").write_text(json.dumps(verification_summary, indent=2) + "\\n")
verified_zip = shutil.make_archive(str(WORK_DIR / "verified_results"), "zip", RESULT_DIR)
print("Shared-model verification complete:", verification_summary)
display(FileLink(verified_zip))
if new_summary is not None:
    experiment_zip = WORK_DIR / f"{EXPERIMENT_NAME}_checkpoint_and_results.zip"
    names = ["best_model.weights.h5", "svm_all.joblib", "svm_ga_selected.joblib", "feature_scaler.joblib",
             "ga_mask.npy", "selected_feature_indices.npy", "run_complete.json", "protocol.json",
             "selection_locked.json", "data_audit.json", "classifier_comparison.csv", "training_history.csv",
             "ga_history.csv", "experiment_config.json"]
    with zipfile.ZipFile(experiment_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name in names:
            if (RUN_DIR / name).exists():
                archive.write(RUN_DIR / name, name)
    print("Your new experiment's checkpoint and results:")
    display(FileLink(str(experiment_zip)))
""", 'completion')
md("""
## References and troubleshooting

- Dataset and license: [MedMNIST+ on Zenodo](https://zenodo.org/records/10519652),
  [MedMNIST project](https://medmnist.com/), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Yang et al., *MedMNIST v2*, Scientific Data 10, 41 (2023), and Yang, Shi and Ni,
  *MedMNIST Classification Decathlon*, ISBI (2021).
- Source microscopy data: Acevedo et al., *A dataset of microscopic peripheral
  blood cell images for development of automatic recognition systems*, Data in
  Brief 30, 105474 (2020), [DOI](https://doi.org/10.1016/j.dib.2020.105474).
- Download error: enable Internet, then rerun the download cell. Correct cached files are reused.
- Package error after installation: restart the session/kernel once, then run from step 1.
- GPU memory error: use `BATCH_SIZE = 32` or `16` and a fresh experiment name.
- Incomplete training checkpoint: choose a new experiment name for a fresh run;
  the notebook does not claim that an interrupted model finished training.
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'},
    'colab': {'name': 'Blood_Cell_GA_Colab_Kaggle.ipynb', 'provenance': [], 'toc_visible': True},
    'accelerator': 'GPU',
})
nbf.validate(notebook)
for index, cell in enumerate(cells):
    if cell.cell_type == 'code':
        compile(cell.source, f'cell_{index + 1}', 'exec')
destination = ROOT / 'Blood_Cell_GA_Colab_Kaggle.ipynb'
nbf.write(notebook, destination)
print('Built', destination, 'with', len(cells), 'cells')
