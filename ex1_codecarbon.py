import math
import time
from codecarbon import EmissionsTracker


def compute_factorials(n: int) -> int:
    """
    Compute factorials from 1..n and return the last value.
    Uses an iterative approach to add some CPU load.
    """
    result = 1
    for i in range(1, n + 1):
        result = math.factorial(i)
    return result


def main():
    N = 4000  # increase to add more workload if needed

    tracker = EmissionsTracker(project_name="ex1_factorials", measure_power_secs=1)
    tracker.start()
    t0 = time.perf_counter()

    try:
        res = compute_factorials(N)
    finally:
        duration_s = time.perf_counter() - t0
        emissions_kg = tracker.stop()  # returns total emissions in kg CO2e

    # Compute digit count of N! without converting the huge integer to a string
    # Using: digits(N!) = floor(ln(N!)/ln(10)) + 1 = floor(lgamma(N+1)/ln(10)) + 1
    digits = int(math.floor(math.lgamma(N + 1) / math.log(10))) + 1
    print(f"Computed factorials up to {N}!; last value has ~{digits} digits")
    print(f"Runtime: {duration_s:.3f} s")
    print(f"Emissions: {emissions_kg:.8f} kg CO2e")
    print("An emissions.csv (or output dir) should be created in the working directory.")


if __name__ == "__main__":
    main()
