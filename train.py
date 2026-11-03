from pathlib import Path
import time
from pipeline import DEFAULT_CONFIG, configure_runtime, load_data, train_model

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run'
configure_runtime(DEFAULT_CONFIG['seed'], DEFAULT_CONFIG['cpu_threads'])
for attempt in range(180):
    if (ROOT / 'data/bloodmnist_128.npz').exists():
        break
    if attempt % 6 == 0:
        print('Waiting for the verified dataset download...', flush=True)
    time.sleep(10)
else:
    raise RuntimeError('Dataset download did not finish within 30 minutes')
data, audit = load_data(ROOT / 'data/bloodmnist_128.npz', RUN)
model = train_model(data, RUN)
