import time
import math
from typing import Dict, List

from codecarbon import OfflineEmissionsTracker


def workload(n: int = 5_000_000) -> int:
    # Simple CPU-bound loop: sum of sqrt to create compute load
    s = 0.0
    for i in range(1, n + 1):
        s += math.sqrt(i)
    return int(s)


def run_for_country(country: str) -> float:
    tracker = OfflineEmissionsTracker(
        project_name="ex4_location_compare",
        experiment_id=country,
        measure_power_secs=1,
        country_iso_code=country,
    )
    tracker.start()
    try:
        workload()
    finally:
        emissions = tracker.stop() or 0.0
    return emissions


def main():
    countries = ["FRA", "USA", "CHN", "SWE"]
    results: Dict[str, float] = {}

    for c in countries:
        time.sleep(0.3)
        e = run_for_country(c)
        results[c] = e
        print(f"{c}: {e:.8f} kg CO2e")

    worst = max(results, key=results.get)
    best = min(results, key=results.get)
    ratio = (results[worst] / max(results[best], 1e-12)) if best in results and worst in results else float('inf')
    print("--- Summary ---")
    print("Emissions by country (kg CO2e):", results)
    print(f"Highest: {worst} = {results[worst]:.8f}; Lowest: {best} = {results[best]:.8f}; Ratio: ~{ratio:.1f}x")
    print("Note: Differences come from regional electricity carbon intensity used by CodeCarbon.")


if __name__ == "__main__":
    main()
