import random
import time
from typing import List, Tuple

from codecarbon import EmissionsTracker


def insertion_sort(arr: List[int]) -> List[int]:
    a = arr[:]  # copy
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


def merge_sort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left: List[int], right: List[int]) -> List[int]:
    out: List[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    if i < len(left):
        out.extend(left[i:])
    if j < len(right):
        out.extend(right[j:])
    return out


def timed_with_emissions(fn, data: List[int], tracker: EmissionsTracker) -> Tuple[List[int], float, float]:
    # Measure only this function's emissions by starting/stopping the tracker around it.
    tracker.start()
    t0 = time.perf_counter()
    try:
        result = fn(data)
    finally:
        duration = time.perf_counter() - t0
        emissions_kg = tracker.stop()
    return result, duration, emissions_kg


def main():
    random.seed(42)
    N = 5000
    data = [random.randint(0, 10**6) for _ in range(N)]

    # Run trackers sequentially to avoid lock conflicts
    tracker_ins = EmissionsTracker(project_name="ex2_sort_compare", experiment_id="insertion_sort", measure_power_secs=1)
    sorted_ins, t_ins, e_ins = timed_with_emissions(insertion_sort, data, tracker_ins)

    # Give CodeCarbon a brief moment to release internal locks
    time.sleep(0.5)

    tracker_mer = EmissionsTracker(project_name="ex2_sort_compare", experiment_id="merge_sort", measure_power_secs=1)
    sorted_mer, t_mer, e_mer = timed_with_emissions(merge_sort, data, tracker_mer)

    assert sorted_ins == sorted_mer == sorted(data), "Sorting results mismatch!"

    print("Insertion sort:")
    print(f"  Time: {t_ins:.3f}s, Emissions: {(e_ins or 0.0):.8f} kg CO2e")
    print("Merge sort:")
    print(f"  Time: {t_mer:.3f}s, Emissions: {(e_mer or 0.0):.8f} kg CO2e")

    if (e_ins or 0.0) > (e_mer or 0.0):
        rel = (e_ins or 0.0) / max((e_mer or 0.0), 1e-12)
        print(f"Insertion sort emitted ~{rel:.1f}x more CO2e than merge sort.")
    elif (e_mer or 0.0) > (e_ins or 0.0):
        rel = (e_mer or 0.0) / max((e_ins or 0.0), 1e-12)
        print(f"Merge sort emitted ~{rel:.1f}x more CO2e than insertion sort.")
    else:
        print("Both algorithms emitted approximately the same CO2e.")


if __name__ == "__main__":
    main()
