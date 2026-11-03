"""Focused checks for the feature alignment and GA fitness used by this project."""
import contextlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from pipeline import DEFAULT_CONFIG, configure_runtime, extract_features, run_binary_ga

ROOT = Path(__file__).resolve().parent
configure_runtime(42, 2)
checks = {}
rng = np.random.default_rng(42)
centers = rng.normal(size=(8, 24)) * 5
train_y = np.repeat(np.arange(8), 12)
val_y = np.repeat(np.arange(8), 4)
train = centers[train_y] + rng.normal(size=(len(train_y), 24)) * 0.1
val = centers[val_y] + rng.normal(size=(len(val_y), 24)) * 0.1
config = dict(DEFAULT_CONFIG, ga_population=8, ga_generations=2, ga_workers=2)
with contextlib.redirect_stdout(io.StringIO()), threadpool_limits(limits=2):
    mask, result, history = run_binary_ga(train, train_y, val, val_y, 1.0, config)
    clf = SVC(C=1.0, gamma='scale').fit(train[:, mask], train_y)
    accuracy = accuracy_score(val_y, clf.predict(val[:, mask]))
np.testing.assert_allclose(result['validation_accuracy'], accuracy, rtol=0, atol=1e-12)
np.testing.assert_allclose(result['fitness'], .99*accuracy+.01*(1-mask.mean()), rtol=0, atol=1e-12)
assert mask.dtype == bool and mask.shape == (24,) and mask.any()
assert list(history.generation) == [0, 1, 2] and np.all(np.diff(history.fitness) >= 0)
checks.update(ga_score_matches_independent_svm_fit=True, ga_returns_nonempty_feature_mask=True,
              final_generation_is_evaluated=True, best_fitness_is_preserved=True)

with tempfile.TemporaryDirectory(prefix='blood_ga_alignment_') as temporary:
    directory=Path(temporary)
    inputs=tf.keras.Input((4,4,3))
    x=tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x=tf.keras.layers.Dense(1024,use_bias=False,name='deep_features')(x)
    x=tf.keras.layers.Dense(256,activation='relu',name='head_dense')(x)
    x=tf.keras.layers.Dropout(.3,name='head_dropout')(x)
    output=tf.keras.layers.Dense(8,activation='softmax',name='cell_class')(x)
    model=tf.keras.Model(inputs,output)
    model.jit_compile=False
    weights=np.zeros((3,1024),dtype=np.float32)
    weights[:,:3]=np.eye(3)
    model.get_layer('deep_features').set_weights([weights])
    model.save_weights(directory/'best_model.weights.h5')
    data={}
    for split,offset in [('train',0),('val',30),('test',60)]:
        pixels=np.broadcast_to((np.arange(12)+offset).astype('uint8')[:,None,None,None],(12,4,4,3)).copy()
        data[f'images_{split}']=pixels
        data[f'y_{split}']=np.arange(12)%8
        data[f'ids_{split}']=np.array([f'{split}_{i}' for i in range(12)])
    with contextlib.redirect_stdout(io.StringIO()):
        arrays=extract_features(model,data,directory,batch_size=4)
    for split in ['train','val','test']:
        np.testing.assert_allclose(arrays[f'X_{split}'][:,:3],data[f'images_{split}'].mean(axis=(1,2)),atol=1e-6)
        np.testing.assert_array_equal(arrays[f'y_{split}'],data[f'y_{split}'])
        np.testing.assert_array_equal(arrays[f'ids_{split}'],data[f'ids_{split}'])
    assert json.loads((directory/'feature_parity.json').read_text())['identical_classes']
    checks.update(actual_keras_feature_rows_match_input_images=True,
                  labels_and_ids_remain_paired=True, extracted_head_matches_image_model=True)

(ROOT/'pipeline_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2),flush=True)
