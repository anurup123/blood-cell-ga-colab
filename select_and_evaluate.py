"""Run extraction, validation-only GA selection, then the locked comparison."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import joblib

from pipeline import (DEFAULT_CONFIG, build_model, configure_runtime, evaluate_classifiers,
                      extract_features, file_digest, fit_scaler_and_choose_svm, load_data, lock_selection, run_binary_ga)

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run'
configure_runtime(DEFAULT_CONFIG['seed'], DEFAULT_CONFIG['cpu_threads'])
for attempt in range(720):
    if (RUN / 'training_complete.json').exists():
        break
    if attempt % 6 == 0:
        print('Waiting for best-validation training checkpoint...', flush=True)
    time.sleep(10)
else:
    raise RuntimeError('Training did not complete within two hours')
record=json.loads((RUN/'training_complete.json').read_text())
config=record['config']
if (RUN/'aligned_features.npz').exists() and (RUN/'feature_parity.json').exists():
    parity=json.loads((RUN/'feature_parity.json').read_text())
    assert parity['identical_classes'] and parity['checkpoint_sha256']==file_digest(RUN/'best_model.weights.h5')
    with np.load(RUN/'aligned_features.npz',allow_pickle=False) as saved:
        arrays={key:saved[key] for key in saved.files}
    print('Reused verified aligned features from the completed checkpoint',flush=True)
else:
    data, audit=load_data(ROOT/'data/bloodmnist_128.npz')
    model, _=build_model(config['image_size'], weights=None, seed=config['seed'])
    model.load_weights(RUN/'best_model.weights.h5')
    arrays=extract_features(model,data,RUN,config['batch_size'])
if (RUN/'svm_validation_search.csv').exists() and (RUN/'feature_scaler.joblib').exists():
    candidates=pd.read_csv(RUN/'svm_validation_search.csv').to_dict('records')
    assert [row['C'] for row in candidates]==config['svm_c_candidates']
    c=float(max(candidates,key=lambda row:(row['validation_accuracy'],-row['C']))['C'])
    scaler=joblib.load(RUN/'feature_scaler.joblib')
    scaled={s:np.asarray(scaler.transform(arrays[f'X_{s}']),dtype=np.float32) for s in ['train','val','test']}
    print('Reused validation-selected SVM C:',c,flush=True)
else:
    scaler,scaled,c=fit_scaler_and_choose_svm(arrays,RUN,config)
mask,ga,history=run_binary_ga(scaled['train'],arrays['y_train'],scaled['val'],arrays['y_val'],c,config)
lock_selection(mask,ga,history,RUN,config)
comparison,summary=evaluate_classifiers(arrays,scaled,mask,RUN,c,config)
