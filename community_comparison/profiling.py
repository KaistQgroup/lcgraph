import time, csv, os, gc, tracemalloc
from contextlib import contextmanager
from typing import Dict, Any, Optional
import psutil

def _rss_mb() -> float:
    # Resident Set Size (MB) = total memory used by the process
    return psutil.Process().memory_info().rss / (1024**2)

@contextmanager
def time_mem_section(metrics: Dict[str, Any], prefix: str):
    """
    Measure wall time + RSS delta + Python heap peak for a code section.
    Results go into `metrics` with keys: {prefix}_sec, {prefix}_rss_mb_delta, {prefix}_py_peak_mb.
    """
    gc.collect()
    start_t = time.perf_counter()
    start_rss = _rss_mb()

    tracemalloc.start()
    try:
        yield
    finally:
        _, py_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        end_t = time.perf_counter()
        end_rss = _rss_mb()

        metrics[f"{prefix}_sec"] = round(end_t - start_t, 6)
        metrics[f"{prefix}_rss_mb_delta"] = round(end_rss - start_rss, 3)
        metrics[f"{prefix}_py_peak_mb"] = round(py_peak / (1024**2), 3)

def write_csv_row(path: str, row: Dict[str, Any], header: Optional[list] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header or list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
