import os
import time
import math
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from codecarbon import OfflineEmissionsTracker


# --- Small CPU-bound workload ---
def workload(n: int = 7_000_000) -> int:
    s = 0.0
    for i in range(1, n + 1):
        s += math.sqrt(i)
    return int(s)


def run_tracked(label: str, n: int, output_dir: str = "reports") -> float:
    os.makedirs(output_dir, exist_ok=True)

    # Best-effort cleanup of stale CodeCarbon lock files that can persist after crashes
    def _clear_codecarbon_locks(paths: List[str]) -> None:
        patterns = [
            "emissions.csv.lock",
            "codecarbon.lock",
            ".codecarbon.lock",
        ]
        for p in paths:
            for fname in patterns:
                fpath = os.path.join(p, fname)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                except Exception:
                    pass

    # Clear in both target output dir and project root (historical default)
    _clear_codecarbon_locks([output_dir, "."])
    tracker = OfflineEmissionsTracker(
        project_name="ex6_reports",
        experiment_id=label,
        output_dir=output_dir,
        measure_power_secs=1,
    )
    tracker.start()
    try:
        workload(n)
    finally:
        emissions_kg = tracker.stop() or 0.0
    return emissions_kg


def analyze_and_plot(csv_path: str, output_dir: str = "reports") -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)

    # Show discovered columns (printed for context)
    print("Columns:", list(df.columns))

    # Prefer standard columns if present
    energy_col = None
    for c in [
        "energy_consumed",  # kWh total if provided
        "energy_kwh",       # sometimes named like this
    ]:
        if c in df.columns:
            energy_col = c
            break

    # Aggregate per experiment_id
    group_cols = [c for c in ["project_name", "experiment_id"] if c in df.columns]
    if not group_cols:
        # Fallback to a single group
        df["experiment_id"] = df.get("experiment_id", "run")
        group_cols = ["experiment_id"]

    agg_map = {"emissions": "sum"} if "emissions" in df.columns else {}
    if energy_col:
        agg_map[energy_col] = "sum"

    if not agg_map:
        # Try to reconstruct energy from CPU/GPU/RAM components if present
        component_cols = [
            "energy_consumed_cpu", "energy_consumed_gpu", "energy_consumed_ram",
            "cpu_energy", "gpu_energy", "ram_energy",
        ]
        components = [c for c in component_cols if c in df.columns]
        if components:
            df["energy_reconstructed"] = df[components].sum(axis=1)
            energy_col = "energy_reconstructed"
            agg_map = {energy_col: "sum"}

    agg = df.groupby(group_cols, dropna=False).agg(agg_map).reset_index()

    # Normalize column names for plotting
    emissions_col = "emissions"
    if emissions_col not in agg.columns:
        # Try alternative common names
        alt_emissions = ["emissions_kg", "total_emissions"]
        for c in alt_emissions:
            if c in agg.columns:
                emissions_col = c
                break

    # Compute an emission factor if possible (gCO2e/kWh)
    if energy_col and emissions_col in agg.columns:
        # Avoid division by zero
        agg["emission_factor_g_per_kwh"] = (agg[emissions_col] / agg[energy_col]).replace([float("inf"), -float("inf")], float("nan")) * 1000.0

    # Plot emissions per run
    plt.figure(figsize=(8, 4.5))
    x = agg["experiment_id"] if "experiment_id" in agg.columns else agg.index.astype(str)
    y = agg[emissions_col] if emissions_col in agg.columns else None

    if y is None:
        print("No emissions column found; will plot energy instead if available.")
        if energy_col is None:
            print("No energy column found either. Skipping plot.")
            return agg, ""
        y = agg[energy_col]
        y_label = f"{energy_col} (kWh)"
        title = "Energy per run"
    else:
        y_label = "Emissions (kg CO2e)"
        title = "Emissions per run"

    plt.bar(x, y)
    plt.title(title)
    plt.xlabel("experiment_id")
    plt.ylabel(y_label)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ex6_runs_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")

    return agg, out_path


def main():
    output_dir = "reports"

    # Run a few tracked workloads with different sizes
    runs: List[Tuple[str, int]] = [
        ("run_small", 3_000_000),
        ("run_medium", 7_000_000),
        ("run_large", 12_000_000),
    ]

    results = {}
    for label, n in runs:
        time.sleep(0.3)
        e = run_tracked(label, n, output_dir=output_dir)
        results[label] = e
        print(f"{label}: {e:.8f} kg CO2e")

    csv_path = os.path.join(output_dir, "emissions.csv")
    if not os.path.isfile(csv_path):
        print(f"ERROR: CSV not found at {csv_path}")
        return

    agg, plot_path = analyze_and_plot(csv_path, output_dir=output_dir)

    print("--- Summary (aggregated) ---")
    print(agg)
    print("If emission_factor_g_per_kwh is NaN, CodeCarbon may not have provided energy; try OfflineEmissionsTracker or a longer run.")


if __name__ == "__main__":
    main()
