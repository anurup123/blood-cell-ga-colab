"""Blood-cell project: DenseNet features, standard binary GA, matched evaluation."""
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

CLASS_NAMES = ["basophil", "eosinophil", "erythroblast", "immature granulocyte",
               "lymphocyte", "monocyte", "neutrophil", "platelet"]
DEFAULT_CONFIG = {
    "seed": 42, "image_size": 128, "batch_size": 64, "warmup_epochs": 2,
    "finetune_epochs": 20, "early_stopping_patience": 5,
    "warmup_learning_rate": 0.001, "finetune_learning_rate": 0.0001,
    "weight_decay": 0.0001, "svm_c_candidates": [0.1, 1.0, 10.0],
    "ga_population": 18, "ga_generations": 20, "ga_accuracy_weight": 0.99,
    "ga_elites": 2, "ga_tournament": 3, "ga_crossover_probability": 0.8,
    "ga_workers": 2, "cpu_threads": 4,
    "deduplication": "remove_train_matches_to_val_or_test_and_val_matches_to_test",
}


def file_digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def configure_runtime(seed=42, threads=4):
    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    print("TensorFlow", tf.__version__, "GPU", tf.config.list_physical_devices("GPU"), flush=True)


def load_data(path, audit_dir=None):
    """Keep the official test set; remove exact overlaps from train/validation."""
    path = Path(path)
    if file_digest(path, "md5") != "adace1e0ed228fccda1f39692059dd4c":
        raise ValueError("Expected the official, unmodified BloodMNIST 128 dataset")
    arrays, hashes, hashes_by_row, rows = {}, {}, {}, []
    with np.load(path, allow_pickle=False) as archive:
        for split, expected in [("train", 11959), ("val", 1712), ("test", 3421)]:
            x = archive[f"{split}_images"]
            y = archive[f"{split}_labels"].reshape(-1).astype(np.int64)
            assert x.shape == (expected, 128, 128, 3) and x.dtype == np.uint8
            assert y.shape == (expected,) and set(np.unique(y)) == set(range(8))
            ids = np.array([f"{split}_{i:05d}" for i in range(expected)])
            image_hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in x]
            hashes_by_row[split] = image_hashes
            hashes[split] = set(image_hashes)
            arrays[f"images_{split}"], arrays[f"y_{split}"], arrays[f"ids_{split}"] = x, y, ids
            rows.extend({"image_id": image_id, "split": split, "label": int(label),
                         "class_name": CLASS_NAMES[int(label)], "image_sha256": digest}
                        for image_id, label, digest in zip(ids, y, image_hashes))
    intersections = {f"{a}_{b}": len(hashes[a] & hashes[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    removed = {}
    for split, forbidden in [("train", hashes["val"] | hashes["test"]), ("val", hashes["test"])]:
        keep = np.array([digest not in forbidden for digest in hashes_by_row[split]], dtype=bool)
        removed[split] = arrays[f"ids_{split}"][~keep].tolist()
        for prefix in ["images", "y", "ids"]:
            arrays[f"{prefix}_{split}"] = arrays[f"{prefix}_{split}"][keep]
        hashes[split] = {digest for digest, retain in zip(hashes_by_row[split], keep) if retain}
    after = {f"{a}_{b}": len(hashes[a] & hashes[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    assert not any(after.values()), f"Exact duplicate images still cross splits: {after}"
    audit = {"dataset_sha256": file_digest(path), "official_md5_verified": True,
             "original_exact_image_duplicates_across_splits": intersections,
             "remaining_exact_image_duplicates_across_splits": after,
             "removed_duplicate_image_ids": removed, "official_test_set_unchanged": True,
             "split_sizes": {s: len(arrays[f"y_{s}"]) for s in ["train", "val", "test"]},
             "class_names": CLASS_NAMES,
             "split_scope": "Official image-level splits with exact train/validation overlaps removed. Test set unchanged. Patient-independent generalization is not established."}
    manifest = pd.DataFrame(rows)
    removed_ids = set(removed["train"]) | set(removed["val"])
    manifest["included"] = ~manifest.image_id.isin(removed_ids)
    if audit_dir:
        directory = Path(audit_dir)
        directory.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(directory / "split_manifest.csv", index=False)
        (directory / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("Removed exact overlap rows:", {s: len(v) for s, v in removed.items()}, flush=True)
    included = manifest.loc[manifest.included]
    print(pd.crosstab(included.class_name, included.split).to_string(), flush=True)
    return arrays, audit


def image_dataset(images, labels=None, batch_size=64, shuffle=False, seed=42):
    data = tf.data.Dataset.from_tensor_slices(images if labels is None else (images, labels))
    if shuffle:
        data = data.shuffle(min(len(images), 4096), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 4
    data = data.with_options(options)
    return data.batch(batch_size).prefetch(2)


def build_model(image_size=128, weights="imagenet", seed=42):
    inputs = tf.keras.Input((image_size, image_size, 3), name="rgb_image")
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed, name="random_flip")(inputs)
    x = tf.keras.layers.RandomRotation(0.1, fill_mode="reflect", seed=seed + 1, name="random_rotation")(x)
    x = tf.keras.layers.Rescaling(1.0 / 255, name="rescale")(x)
    x = tf.keras.layers.Normalization(mean=[0.485, 0.456, 0.406],
                                     variance=[0.229**2, 0.224**2, 0.225**2], name="imagenet_normalization")(x)
    backbone = tf.keras.applications.DenseNet121(include_top=False, weights=weights,
                                                 input_shape=(image_size, image_size, 3))
    backbone.trainable = False
    # Frozen BatchNorm statistics are used during both fine-tuning and inference.
    x = backbone(x, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="deep_features")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="head_dense")(features)
    x = tf.keras.layers.Dropout(0.3, seed=seed + 2, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(8, activation="softmax", name="cell_class")(x)
    model = tf.keras.Model(inputs, outputs, name="blood_cell_densenet121")
    model.jit_compile = False
    return model, backbone


def train_model(data, run_dir, config=None):
    config = dict(DEFAULT_CONFIG if config is None else config)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    complete = run_dir / "training_complete.json"
    checkpoint = run_dir / "best_model.weights.h5"
    tf.keras.utils.set_random_seed(config["seed"])
    model, backbone = build_model(config["image_size"], weights=None if complete.exists() else "imagenet", seed=config["seed"])
    if complete.exists():
        record = json.loads(complete.read_text())
        assert record["config"] == config and record["checkpoint_sha256"] == file_digest(checkpoint)
        model.load_weights(checkpoint)
        print("Loaded completed checkpoint:", checkpoint, flush=True)
        return model
    if checkpoint.exists():
        raise RuntimeError("An incomplete run exists. Choose a new experiment directory for fresh training.")
    (run_dir / "protocol.json").write_text(json.dumps(config, indent=2) + "\n")
    train = image_dataset(data["images_train"], data["y_train"], config["batch_size"], True, config["seed"])
    validation = image_dataset(data["images_val"], data["y_val"], config["batch_size"])
    history_parts = []

    def compile_model(rate):
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=rate, weight_decay=config["weight_decay"]),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=False)

    def checkpoint_callback(threshold=None):
        return tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_accuracy", mode="max",
                    save_best_only=True, save_weights_only=True, initial_value_threshold=threshold)

    start = time.monotonic()
    compile_model(config["warmup_learning_rate"])
    print("PHASE 1: train classifier head", flush=True)
    warmup = model.fit(train, validation_data=validation, epochs=config["warmup_epochs"], verbose=2,
                       callbacks=[checkpoint_callback()])
    history_parts.append(pd.DataFrame(warmup.history).assign(phase="warmup"))
    warmup_best = max(warmup.history["val_accuracy"])
    model.load_weights(checkpoint)
    backbone.trainable = True
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(config["finetune_learning_rate"])
    print("PHASE 2: fine-tune DenseNet121", flush=True)
    fine = model.fit(train, validation_data=validation, epochs=config["finetune_epochs"], verbose=2, callbacks=[
        checkpoint_callback(warmup_best),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=config["early_stopping_patience"], min_delta=0.0005),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ])
    history_parts.append(pd.DataFrame(fine.history).assign(phase="finetune"))
    history = pd.concat(history_parts, ignore_index=True)
    history.insert(0, "epoch", np.arange(1, len(history) + 1))
    history.to_csv(run_dir / "training_history.csv", index=False)
    model.load_weights(checkpoint)
    model.save(run_dir / "best_model.keras")
    record = {"config": config, "checkpoint_sha256": file_digest(checkpoint),
              "best_validation_accuracy": float(history.val_accuracy.max()),
              "best_epoch": int(history.loc[history.val_accuracy.idxmax(), "epoch"]),
              "epochs_run": len(history), "training_seconds": time.monotonic() - start}
    complete.write_text(json.dumps(record, indent=2) + "\n")
    print("TRAINING COMPLETE", json.dumps(record), flush=True)
    return model


def extract_features(model, data, run_dir, batch_size=64):
    """The prediction iterator never shuffles; save explicit row IDs and labels."""
    run_dir = Path(run_dir)
    extractor = tf.keras.Model(model.input, model.get_layer("deep_features").output)
    extractor.jit_compile = False
    arrays = {}
    for split in ["train", "val", "test"]:
        features = extractor.predict(image_dataset(data[f"images_{split}"], batch_size=batch_size), verbose=1)
        assert features.shape == (len(data[f"y_{split}"]), 1024) and np.isfinite(features).all()
        arrays[f"X_{split}"] = features
        arrays[f"y_{split}"] = data[f"y_{split}"].copy()
        arrays[f"ids_{split}"] = data[f"ids_{split}"].copy()
    np.savez_compressed(run_dir / "aligned_features.npz", **arrays)
    inputs = tf.keras.Input((1024,))
    x = inputs
    for name in ["head_dense", "head_dropout", "cell_class"]:
        x = model.get_layer(name)(x)
    original_head = tf.keras.Model(inputs, x)
    original_head.jit_compile = False
    head_probabilities = original_head.predict(arrays["X_test"], batch_size=batch_size, verbose=0)
    image_probabilities = model.predict(image_dataset(data["images_test"], batch_size=batch_size), verbose=1)
    np.testing.assert_allclose(head_probabilities, image_probabilities, rtol=2e-4, atol=2e-5)
    np.testing.assert_array_equal(head_probabilities.argmax(1), image_probabilities.argmax(1))
    np.savez_compressed(run_dir / "densenet_test_predictions.npz", probabilities=image_probabilities,
                        labels=arrays["y_test"], image_ids=arrays["ids_test"])
    parity = {"identical_classes": True, "maximum_probability_difference": float(np.max(np.abs(head_probabilities - image_probabilities))),
              "feature_width": 1024, "checkpoint_sha256": file_digest(run_dir / "best_model.weights.h5"),
              "note": "This is an extraction-equivalence check; test accuracy is reported only after GA selection is locked."}
    (run_dir / "feature_parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    print("FEATURE ALIGNMENT AND HEAD PARITY PASSED", json.dumps(parity), flush=True)
    return arrays


def fit_scaler_and_choose_svm(arrays, run_dir, config=None):
    config = DEFAULT_CONFIG if config is None else config
    run_dir = Path(run_dir)
    scaler = StandardScaler().fit(arrays["X_train"])
    scaled = {s: np.asarray(scaler.transform(arrays[f"X_{s}"]), dtype=np.float32) for s in ["train", "val", "test"]}
    candidates = []
    with threadpool_limits(limits=config["cpu_threads"]):
        for c in config["svm_c_candidates"]:
            start = time.monotonic()
            model = SVC(C=c, kernel="rbf", gamma="scale", cache_size=512)
            model.fit(scaled["train"], arrays["y_train"])
            accuracy = accuracy_score(arrays["y_val"], model.predict(scaled["val"]))
            candidates.append({"C": c, "validation_accuracy": float(accuracy), "seconds": time.monotonic() - start})
            print("SVM VALIDATION", json.dumps(candidates[-1]), flush=True)
    chosen = max(candidates, key=lambda r: (r["validation_accuracy"], -r["C"]))
    pd.DataFrame(candidates).to_csv(run_dir / "svm_validation_search.csv", index=False)
    joblib.dump(scaler, run_dir / "feature_scaler.joblib")
    return scaler, scaled, float(chosen["C"])


def run_binary_ga(x_train, y_train, x_val, y_val, c=10.0, config=None):
    """Conventional tournament selection, uniform crossover, mutation and elitism.

    Only training and validation arrays are accepted. The all-feature candidate
    is included, and every generated population (including the last) is scored.
    """
    from concurrent.futures import ThreadPoolExecutor
    import os
    config = dict(DEFAULT_CONFIG if config is None else config)
    rng = np.random.default_rng(config["seed"])
    width = x_train.shape[1]
    assert x_val.shape[1] == width and len(x_train) == len(y_train) and len(x_val) == len(y_val)
    count, elites = config["ga_population"], config["ga_elites"]
    alpha = config["ga_accuracy_weight"]
    assert 0 < alpha <= 1 and 0 < elites < count
    population = [np.ones(width, dtype=bool)]
    for index in range(count - 1):
        mask = rng.random(width) < [0.25, 0.5, 0.75][index % 3]
        mask[rng.integers(width)] = True
        population.append(mask)
    cache, history = {}, []
    start = time.monotonic()

    def key(mask):
        return np.packbits(mask).tobytes()

    def score(mask):
        classifier = SVC(C=c, kernel="rbf", gamma="scale", cache_size=256)
        classifier.fit(x_train[:, mask], y_train)
        accuracy = float(accuracy_score(y_val, classifier.predict(x_val[:, mask])))
        return (float(alpha * accuracy + (1 - alpha) * (1 - mask.mean())), accuracy, int(mask.sum()))

    def ordering(mask):
        value = cache[key(mask)]
        return value[0], value[1], -value[2]

    best_mask = population[0].copy()
    workers = int(os.environ.get("BLOOD_GA_WORKERS", config["ga_workers"]))
    assert workers >= 1
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        for generation in range(config["ga_generations"] + 1):
            pending = {key(mask): mask for mask in population if key(mask) not in cache}
            for mask_key, result in zip(pending, pool.map(score, pending.values())):
                cache[mask_key] = result
            population.sort(key=ordering, reverse=True)
            if ordering(population[0]) > ordering(best_mask):
                best_mask = population[0].copy()
            best_score, best_accuracy, selected = cache[key(best_mask)]
            history.append({"generation": generation, "fitness": best_score, "validation_accuracy": best_accuracy,
                            "selected_features": selected, "unique_masks_evaluated": len(cache), "seconds": time.monotonic() - start})
            print(f"GA generation {generation:02d}/{config['ga_generations']} | validation {best_accuracy:.4%} | features {selected}/{width} | fitness {best_score:.6f}", flush=True)
            if generation == config["ga_generations"]:
                break

            def parent():
                candidates = rng.choice(len(population), size=config["ga_tournament"], replace=False)
                return max((population[i] for i in candidates), key=ordering)

            children = [mask.copy() for mask in population[:elites]]
            while len(children) < count:
                first, second = parent(), parent()
                child = first.copy()
                if rng.random() < config["ga_crossover_probability"]:
                    swap = rng.random(width) < 0.5
                    child[swap] = second[swap]
                child ^= rng.random(width) < 1.0 / width
                if not child.any():
                    child[rng.integers(width)] = True
                children.append(child)
            population = children
    baseline = cache[key(np.ones(width, dtype=bool))]
    result = {"algorithm": "standard_binary_genetic_algorithm", "fitness": cache[key(best_mask)][0],
              "validation_accuracy": cache[key(best_mask)][1], "selected_features": int(best_mask.sum()),
              "total_features": width, "all_feature_validation_accuracy": baseline[1],
              "accuracy_weight": alpha, "svm_C": c, "unique_masks_evaluated": len(cache),
              "generations": config["ga_generations"], "population_size": count,
              "elapsed_seconds": time.monotonic() - start, "seed": config["seed"], "evaluation_workers": workers}
    return best_mask, result, pd.DataFrame(history)


def lock_selection(mask, ga_result, history, run_dir, config):
    """Persist all choices before reporting held-out test accuracy."""
    from datetime import datetime, timezone
    run_dir = Path(run_dir)
    np.save(run_dir / "ga_mask.npy", np.asarray(mask, dtype=bool))
    np.save(run_dir / "selected_feature_indices.npy", np.flatnonzero(mask))
    history.to_csv(run_dir / "ga_history.csv", index=False)
    (run_dir / "ga_result.json").write_text(json.dumps(ga_result, indent=2) + "\n")
    record = {"locked_at_utc": datetime.now(timezone.utc).isoformat(),
              "mask_sha256": file_digest(run_dir / "ga_mask.npy"),
              "features_sha256": file_digest(run_dir / "aligned_features.npz"),
              "checkpoint_sha256": file_digest(run_dir / "best_model.weights.h5"),
              "svm_C": ga_result["svm_C"], "config": dict(config),
              "selection_data": "training and validation only"}
    (run_dir / "selection_locked.json").write_text(json.dumps(record, indent=2) + "\n")


def build_feature_head(width, seed=42):
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.Input((width,)), tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3, seed=seed), tf.keras.layers.Dense(8, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
    return model


def evaluate_classifiers(arrays, scaled, mask, run_dir, c, config=None, head_epochs=60):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    config = DEFAULT_CONFIG if config is None else config
    run_dir = Path(run_dir)
    lock = json.loads((run_dir / "selection_locked.json").read_text())
    assert lock["mask_sha256"] == file_digest(run_dir / "ga_mask.npy")
    assert lock["features_sha256"] == file_digest(run_dir / "aligned_features.npz")
    assert lock["svm_C"] == c
    np.testing.assert_array_equal(mask, np.load(run_dir / "ga_mask.npy"))
    rows = []

    def record(name, subset, split, prediction, feature_count):
        labels = arrays[f"y_{split}"]
        rows.append({"classifier": name, "features": subset, "n_features": feature_count, "split": split,
                     "samples": len(labels), "correct": int((prediction == labels).sum()),
                     "accuracy": float(accuracy_score(labels, prediction)),
                     "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
                     "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0))})
        np.savez_compressed(run_dir / f"predictions_{name}_{subset}_{split}.npz", predictions=prediction,
                            labels=labels, image_ids=arrays[f"ids_{split}"])
        if split == "test":
            pd.DataFrame(confusion_matrix(labels, prediction, labels=np.arange(8)), index=CLASS_NAMES,
                         columns=CLASS_NAMES).to_csv(run_dir / f"confusion_{name}_{subset}.csv")
            report = classification_report(labels, prediction, labels=np.arange(8), target_names=CLASS_NAMES,
                                           output_dict=True, zero_division=0)
            (run_dir / f"report_{name}_{subset}.json").write_text(json.dumps(report, indent=2) + "\n")

    with threadpool_limits(limits=config["cpu_threads"]):
        for subset, selection in [("all", np.ones_like(mask)), ("ga_selected", mask)]:
            x_train, x_val, x_test = [scaled[s][:, selection] for s in ["train", "val", "test"]]
            models = {
                "SVM": SVC(C=c, kernel="rbf", gamma="scale", cache_size=512),
                "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=config["cpu_threads"]),
                "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=config["seed"], n_jobs=config["cpu_threads"]),
            }
            for name, classifier in models.items():
                print("Fitting", name, subset, int(selection.sum()), "features", flush=True)
                classifier.fit(x_train, arrays["y_train"])
                if name == "SVM":
                    joblib.dump(classifier, run_dir / f"svm_{subset}.joblib", compress=3)
                for split, x in [("val", x_val), ("test", x_test)]:
                    record(name, subset, split, classifier.predict(x), int(selection.sum()))
            print("Fitting NeuralHead", subset, flush=True)
            head = build_feature_head(int(selection.sum()), config["seed"])
            head_path = run_dir / f"head_{subset}.weights.h5"
            train = image_dataset(x_train, arrays["y_train"], 64, True, config["seed"])
            validation = image_dataset(x_val, arrays["y_val"], 64)
            history = head.fit(train, validation_data=validation, epochs=head_epochs, verbose=2, callbacks=[
                tf.keras.callbacks.ModelCheckpoint(str(head_path), monitor="val_accuracy", mode="max", save_best_only=True, save_weights_only=True),
                tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=8, min_delta=0.0005),
            ])
            pd.DataFrame(history.history).to_csv(run_dir / f"head_history_{subset}.csv", index=False)
            head.load_weights(head_path)
            for split, x in [("val", x_val), ("test", x_test)]:
                prediction = head.predict(image_dataset(x, batch_size=64), verbose=0).argmax(axis=1)
                record("NeuralHead", subset, split, prediction, int(selection.sum()))
    with np.load(run_dir / "densenet_test_predictions.npz") as saved:
        image_prediction = saved["probabilities"].argmax(axis=1)
        np.testing.assert_array_equal(saved["labels"], arrays["y_test"])
        np.testing.assert_array_equal(saved["image_ids"], arrays["ids_test"])
    dense_accuracy = float(accuracy_score(arrays["y_test"], image_prediction))
    result = pd.DataFrame(rows)
    assert len(result) == 16
    result.to_csv(run_dir / "classifier_comparison.csv", index=False)
    summary = {"densenet_test_accuracy": dense_accuracy, "test_images": len(arrays["y_test"]),
               "features_before": len(mask), "features_after": int(mask.sum()),
               "feature_reduction_percent": float(100 * (1 - mask.mean())),
               "selected_svm_C": c, "ga_selection_used_test_labels": False}
    for subset in ["all", "ga_selected"]:
        row = result.query("classifier == 'SVM' and features == @subset and split == 'test'").iloc[0]
        summary[f"svm_{subset}_test_accuracy"] = float(row.accuracy)
        summary[f"svm_{subset}_correct"] = int(row.correct)
    summary["svm_change_percentage_points"] = 100 * (summary["svm_ga_selected_test_accuracy"] - summary["svm_all_test_accuracy"])
    (run_dir / "run_complete.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("FINAL TEST COMPARISON", flush=True)
    print(result.query("split == 'test'").pivot(index="classifier", columns="features", values="accuracy").to_string(), flush=True)
    print("RUN COMPLETE", json.dumps(summary), flush=True)
    return result, summary
