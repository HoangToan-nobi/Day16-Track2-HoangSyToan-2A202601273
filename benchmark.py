"""LightGBM benchmark for the Day 16 CPU lab."""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_DIR = Path.home() / "ml-benchmark"
OUTPUT_FILE = Path("benchmark_result.json")


def find_dataset() -> Path:
    candidates = sorted(DATA_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy file CSV trong {DATA_DIR}. "
            "Hãy tải dataset Kaggle trước."
        )
    return candidates[0]


def main() -> None:
    dataset_path = find_dataset()

    load_started = time.perf_counter()
    data = pd.read_csv(dataset_path)
    load_seconds = time.perf_counter() - load_started

    if "Class" not in data.columns:
        raise ValueError("Dataset không có cột nhãn 'Class'.")

    X = data.drop(columns=["Class"])
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=negative / positive,
        verbosity=-1,
    )

    train_started = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    train_seconds = time.perf_counter() - train_started

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    inference_input = X_test.iloc[: min(1000, len(X_test))]
    for _ in range(5):
        model.predict_proba(inference_input)

    single_row = X_test.iloc[[0]]
    latency_runs = 100
    latency_started = time.perf_counter()
    for _ in range(latency_runs):
        model.predict_proba(single_row)
    latency_ms = (time.perf_counter() - latency_started) / latency_runs * 1000

    throughput_started = time.perf_counter()
    throughput_runs = 20
    for _ in range(throughput_runs):
        model.predict_proba(inference_input)
    throughput_seconds = (time.perf_counter() - throughput_started) / throughput_runs
    rows_per_second = len(inference_input) / throughput_seconds

    result = {
        "dataset": str(dataset_path),
        "rows": int(len(data)),
        "features": int(X.shape[1]),
        "load_data_seconds": round(load_seconds, 4),
        "training_seconds": round(train_seconds, 4),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "auc_roc": round(float(roc_auc_score(y_test, probabilities)), 6),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 6),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 6),
        "inference_latency_1_row_ms": round(float(latency_ms), 6),
        "inference_throughput_1000_rows_per_second": round(float(rows_per_second), 3),
    }

    OUTPUT_FILE.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result, indent=2))
    print(f"\nĐã ghi kết quả vào: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
