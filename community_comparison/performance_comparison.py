import time, tracemalloc, gc, threading, os, psutil
from statistics import mean
import csv
from pathlib import Path


def _rss_sampler(stop_evt, pid, out_dict, interval=0.01):
    p = psutil.Process(pid)
    peak = p.memory_info().rss
    while not stop_evt.is_set():
        rss = p.memory_info().rss
        if rss > peak:
            peak = rss
        time.sleep(interval)
    out_dict["peak_rss"] = max(peak, out_dict.get("peak_rss", 0))

def measure_performance(func, *args, repeat=3, **kwargs):
    """
    Measure runtime and peak memory (Python + RSS) of a callable.
    Returns dict with best/avg timings and peaks across repeats.
    """
    pid = os.getpid()
    times, peaks_tracemalloc, peaks_rss = [], [], []

    for _ in range(repeat):
        gc.collect()

        # Start RSS sampler thread
        stop_evt = threading.Event()
        shared = {}
        t = threading.Thread(target=_rss_sampler, args=(stop_evt, pid, shared), daemon=True)
        t.start()

        # Start tracemalloc
        tracemalloc.start()
        start = time.perf_counter()

        result = func(*args, **kwargs)  # <-- run the target

        elapsed = time.perf_counter() - start
        current, peak_tm = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Stop sampler
        stop_evt.set()
        t.join()
        peak_rss = shared.get("peak_rss", psutil.Process(pid).memory_info().rss)

        times.append(elapsed)
        peaks_tracemalloc.append(peak_tm / (1024**2))     # MB
        peaks_rss.append(peak_rss / (1024**2))            # MB

        # help GC between repeats; avoid holding onto big objects
        del result
        gc.collect()

    return {
        "best_time_s": min(times),
        "avg_time_s": mean(times),
        "peak_tracemalloc_mb": max(peaks_tracemalloc),
        "peak_rss_mb": max(peaks_rss),
        "runs": repeat,
    }



def run_all_benchmarks(G, benchmarks, repeat=3):
    rows = []
    for name, fn in benchmarks:
        stats = measure_performance(fn, G, repeat=repeat)
        rows.append({
            "method": name,
            "runs": stats["runs"],
            "best_time_s": round(stats["best_time_s"], 4),
            "avg_time_s": round(stats["avg_time_s"], 4),
            "peak_tracemalloc_mb": round(stats["peak_tracemalloc_mb"], 2),
            "peak_rss_mb": round(stats["peak_rss_mb"], 2),
        })
    return rows


def write_results_csv(results, filepath="bench_results.csv", overwrite=True):
    """
    Write a list of dict rows (from run_all_benchmarks) to CSV.
    - results: list of dicts like {'method': ..., 'runs': ..., ...}
    - filepath: output path
    - overwrite: if False and file exists, it will append new rows (no header)
    """
    filepath = Path(filepath)
    fieldnames = ["method", "runs", "best_time_s", "avg_time_s",
                  "peak_tracemalloc_mb", "peak_rss_mb"]

    mode = "w" if (overwrite or not filepath.exists()) else "a"
    write_header = (mode == "w")

    with filepath.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in results:
            writer.writerow(row)
