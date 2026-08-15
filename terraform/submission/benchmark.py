import json
import statistics
import time
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


data_path = Path.home() / "ml-benchmark" / "creditcard.csv"

load_start = time.perf_counter()
df = pd.read_csv(data_path)
load_time = time.perf_counter() - load_start

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

positive_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = LGBMClassifier(
    objective="binary",
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    scale_pos_weight=positive_weight,
    random_state=42,
    n_jobs=2,
    verbosity=-1,
)

train_start = time.perf_counter()

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[early_stopping(20, verbose=False)],
)

training_time = time.perf_counter() - train_start

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

single_row = X_test.iloc[[0]]
latencies = []

for _ in range(50):
    start = time.perf_counter()
    model.predict_proba(single_row)
    latencies.append((time.perf_counter() - start) * 1000)

batch = X_test.iloc[:1000]
start = time.perf_counter()
model.predict(batch)
batch_time = time.perf_counter() - start

result = {
    "rows": int(len(df)),
    "load_data_seconds": round(load_time, 4),
    "training_seconds": round(training_time, 4),
    "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
    "auc_roc": round(roc_auc_score(y_test, probabilities), 6),
    "accuracy": round(accuracy_score(y_test, predictions), 6),
    "f1_score": round(f1_score(y_test, predictions, zero_division=0), 6),
    "precision": round(precision_score(y_test, predictions, zero_division=0), 6),
    "recall": round(recall_score(y_test, predictions, zero_division=0), 6),
    "inference_latency_ms": round(statistics.median(latencies), 4),
    "inference_throughput_rows_per_second": round(len(batch) / batch_time, 2),
}

Path("benchmark_result.json").write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

print(json.dumps(result, indent=2))
