"""Package only measured results and the exact model used to obtain them."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import zipfile

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import tensorflow as tf

from pipeline import CLASS_NAMES, build_model, configure_runtime, file_digest, load_data

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run'
ASSETS = ROOT / 'release_assets'
BUNDLE = ASSETS / 'model_bundle'
RESULTS = ROOT / 'results'
for directory in [BUNDLE / 'examples', RESULTS]:
    directory.mkdir(parents=True, exist_ok=True)
summary = json.loads((RUN / 'run_complete.json').read_text())
training = json.loads((RUN / 'training_complete.json').read_text())
configure_runtime(42, 2)
model, _ = build_model(weights=None)
model.load_weights(RUN / 'best_model.weights.h5')
model.save_weights(BUNDLE / 'densenet121.weights.h5')

names = ['aligned_features.npz', 'feature_scaler.joblib', 'svm_all.joblib', 'svm_ga_selected.joblib',
         'ga_mask.npy', 'selected_feature_indices.npy', 'ga_result.json', 'ga_history.csv',
         'training_history.csv', 'training_complete.json', 'run_complete.json', 'classifier_comparison.csv',
         'split_manifest.csv', 'data_audit.json', 'feature_parity.json', 'selection_locked.json',
         'svm_validation_search.csv', 'confusion_SVM_ga_selected.csv', 'report_SVM_ga_selected.json',
         'densenet_test_predictions.npz']
for name in names:
    shutil.copy2(RUN / name, BUNDLE / name)
    if name.endswith(('.json', '.csv')):
        shutil.copy2(RUN / name, RESULTS / name)

with np.load(RUN / 'aligned_features.npz', allow_pickle=False) as archive:
    arrays = {key: archive[key] for key in archive.files}
mask = np.load(RUN / 'ga_mask.npy')
scaler = joblib.load(RUN / 'feature_scaler.joblib')
all_model = joblib.load(RUN / 'svm_all.joblib')
ga_model = joblib.load(RUN / 'svm_ga_selected.joblib')
scaled_test = np.asarray(scaler.transform(arrays['X_test']), dtype=np.float32)
table = pd.read_csv(RUN / 'classifier_comparison.csv')
for row in table.itertuples():
    with np.load(RUN / f'predictions_{row.classifier}_{row.features}_{row.split}.npz') as saved:
        np.testing.assert_array_equal(saved['labels'], arrays[f'y_{row.split}'])
        np.testing.assert_array_equal(saved['image_ids'], arrays[f'ids_{row.split}'])
        actual = [accuracy_score(saved['labels'], saved['predictions']),
                  balanced_accuracy_score(saved['labels'], saved['predictions']),
                  f1_score(saved['labels'], saved['predictions'], average='macro', zero_division=0)]
        np.testing.assert_allclose(actual, [row.accuracy, row.balanced_accuracy, row.macro_f1], rtol=0, atol=1e-12)
for label, classifier, selection in [('all', all_model, np.ones_like(mask)), ('ga_selected', ga_model, mask)]:
    prediction = classifier.predict(scaled_test[:, selection])
    with np.load(RUN / f'predictions_SVM_{label}_test.npz') as reference:
        np.testing.assert_array_equal(prediction, reference['predictions'])
    np.testing.assert_allclose(accuracy_score(arrays['y_test'], prediction), summary[f'svm_{label}_test_accuracy'], atol=1e-12)

data, audit = load_data(ROOT / 'data/bloodmnist_128.npz')
examples, indices = [], []
for label, name in enumerate(CLASS_NAMES):
    index = int(np.flatnonzero(data['y_test'] == label)[0])
    filename = str(data['ids_test'][index]) + '.png'
    Image.fromarray(data['images_test'][index]).save(BUNDLE / 'examples' / filename)
    examples.append({'filename': filename, 'label': label, 'class_name': name})
    indices.append(index)
feature_model = tf.keras.Model(model.input, model.get_layer('deep_features').output)
feature_model.jit_compile = False
example_features = feature_model.predict(data['images_test'][indices], batch_size=8, verbose=0)
example_scaled = np.asarray(scaler.transform(example_features), dtype=np.float32)
np.testing.assert_array_equal(ga_model.predict(example_scaled[:, mask]), ga_model.predict(scaled_test[indices][:, mask]))
metadata = {
    'dataset': 'BloodMNIST 128', 'class_names': CLASS_NAMES, 'image_size': 128,
    'model': 'DenseNet121', 'feature_layer': 'deep_features', 'raw_input_range': [0, 255],
    'preprocessing': 'RGB; 1/255 scaling and ImageNet normalization are built into the model',
    'best_training_epoch': training['best_epoch'], 'best_validation_accuracy': training['best_validation_accuracy'],
    'source_checkpoint_sha256': file_digest(RUN / 'best_model.weights.h5'),
    'exported_weights_sha256': file_digest(BUNDLE / 'densenet121.weights.h5'),
    'examples': examples, 'summary': summary,
    'versions': {name: importlib.metadata.version(name) for name in
                 ['tensorflow', 'keras', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'Pillow', 'joblib']},
}
(BUNDLE / 'model_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
notice = '''BloodMNIST data attribution

MedMNIST and MedMNIST+ by Jiancheng Yang and colleagues.
Yang et al., MedMNIST v2, Scientific Data 10, 41 (2023).
Yang, Shi and Ni, MedMNIST Classification Decathlon, ISBI (2021).
Official dataset: https://zenodo.org/records/10519652
Project: https://medmnist.com/

Source blood-cell images: Acevedo et al., A dataset of microscopic peripheral
blood cell images for development of automatic recognition systems,
Data in Brief 30, 105474 (2020). DOI: 10.1016/j.dib.2020.105474.

BloodMNIST license: Creative Commons Attribution 4.0 International.
License: https://creativecommons.org/licenses/by/4.0/

The dataset release file is a byte-identical project mirror of the official
128-pixel archive. The example PNGs contain the dataset pixel arrays. Our
training/validation loader excludes 8 training rows and 1 validation row that
duplicate images in another split; the official test set is unchanged. Features,
models and comparisons are outputs of this educational project. This is an
image-level benchmark, not a validation for diagnosis or unseen patients.
'''
(ROOT / 'DATA_ATTRIBUTION.md').write_text(notice)
(BUNDLE / 'DATA_ATTRIBUTION.txt').write_text(notice)
checksums = {str(path.relative_to(BUNDLE)): file_digest(path) for path in sorted(BUNDLE.rglob('*'))
             if path.is_file() and path.name != 'checksums.json'}
(BUNDLE / 'checksums.json').write_text(json.dumps(checksums, indent=2) + '\n')
bundle_zip = ASSETS / 'blood_cell_model_bundle_v1.zip'
with zipfile.ZipFile(bundle_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
    for path in sorted(BUNDLE.rglob('*')):
        if path.is_file():
            archive.write(path, path.relative_to(BUNDLE))
dataset = ROOT / 'data/bloodmnist_128.npz'
manifest = {
    'repository': 'anurup123/blood-cell-ga-colab', 'release': 'v1.0.0',
    'model_bundle': {'name': bundle_zip.name, 'size': bundle_zip.stat().st_size, 'sha256': file_digest(bundle_zip)},
    'dataset': {'name': dataset.name, 'size': dataset.stat().st_size, 'sha256': file_digest(dataset),
                'official_url': 'https://zenodo.org/records/10519652/files/bloodmnist_128.npz?download=1',
                'official_md5': 'adace1e0ed228fccda1f39692059dd4c'},
}
(ROOT / 'assets_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
verification = {'classifier_metric_rows_recomputed': 16, 'prediction_labels_and_ids_match_saved_features': True,
                'exported_svms_reproduce_reported_predictions': True,
                'raw_example_ga_predictions_match_saved_feature_predictions': True,
                'model_bundle_size_bytes': bundle_zip.stat().st_size, **summary}
(RESULTS / 'export_validation.json').write_text(json.dumps(verification, indent=2) + '\n')
print('ASSET EXPORT COMPLETE', json.dumps(verification), flush=True)
