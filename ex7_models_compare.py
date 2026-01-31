import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from codecarbon import OfflineEmissionsTracker


def run_with_tracking(name: str, model, X_train, y_train, X_test, y_test, output_dir: str = "reports") -> Tuple[str, float, float, float]:
    os.makedirs(output_dir, exist_ok=True)
    tracker = OfflineEmissionsTracker(project_name="ex7_models_compare", experiment_id=name, output_dir=output_dir, measure_power_secs=1)
    tracker.start()
    t0 = time.perf_counter()
    try:
        model.fit(X_train, y_train)
    finally:
        duration = time.perf_counter() - t0
        emissions = tracker.stop() or 0.0
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    return name, acc, duration, emissions


def plot_results(df: pd.DataFrame, output_dir: str = "reports") -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    # Bar plot for emissions
    plt.figure(figsize=(8, 4))
    plt.bar(df['model'], df['emissions_kg'])
    plt.ylabel('Emissions (kg CO2e)')
    plt.title('Emissions per model')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plot_em = os.path.join(output_dir, 'ex7_emissions_per_model.png')
    plt.savefig(plot_em, dpi=150)

    # Dual-axis plot: accuracy vs time
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.bar(df['model'], df['duration_s'], color='tab:orange', alpha=0.6, label='Time (s)')
    ax2.plot(df['model'], df['accuracy'], color='tab:blue', marker='o', label='Accuracy')
    ax1.set_ylabel('Time (s)')
    ax2.set_ylabel('Accuracy')
    ax1.set_title('Time vs Accuracy per model')
    ax1.set_xticklabels(df['model'], rotation=15)
    fig.tight_layout()
    plot_acc_time = os.path.join(output_dir, 'ex7_time_accuracy.png')
    fig.savefig(plot_acc_time, dpi=150)

    return plot_em, plot_acc_time


def main():
    output_dir = 'reports'
    os.makedirs(output_dir, exist_ok=True)

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    models: List[Tuple[str, object]] = [
        ("LogReg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)) ),
        ("SVM", make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale')) ),
        ("RandomForest", RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42) ),
        ("MLP", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)) ),
    ]

    rows = []
    for name, model in models:
        time.sleep(0.3)
        m, acc, dur, em = run_with_tracking(name, model, X_train, y_train, X_test, y_test, output_dir=output_dir)
        rows.append({
            'model': m,
            'accuracy': acc,
            'duration_s': dur,
            'emissions_kg': em,
        })
        print(f"{name:12s} -> acc: {acc:.3f}, time: {dur:.3f}s, emissions: {em:.8f} kg CO2e")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'ex7_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved results CSV: {csv_path}")

    plot_em, plot_acc_time = plot_results(df, output_dir=output_dir)
    print(f"Saved plots: {plot_em}, {plot_acc_time}")

    # Print a small sorted table by emissions
    print("\nSorted by emissions (kg CO2e):")
    print(df.sort_values('emissions_kg'))


if __name__ == '__main__':
    main()
