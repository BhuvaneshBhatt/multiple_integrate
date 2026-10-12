"""Report cold certification versus reuse; timings are measurements, not test gates."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import sympy as sp
from semialg import chart_certificates as gc


def benchmark(repeats=7, hits=1000):
    x, y, t, u = sp.symbols("x y t u", real=True)
    chart = gc.Chart((t, u), (t + u * u, u), (x - y * y, y), ((u, 0, 1), (t, 0, 1)))
    cold = []
    hot = []
    for _ in range(repeats):
        gc.clear_chart_cache()
        start = time.perf_counter()
        certificate = gc.certify_chart(chart, x - y * y, (x, y))
        cold.append(time.perf_counter() - start)
        start = time.perf_counter()
        for _ in range(hits):
            assert gc.certify_chart(chart, x - y * y, (x, y)) is certificate
        hot.append((time.perf_counter() - start) / hits)
    return {
        "repeats": repeats,
        "hits_per_repeat": hits,
        "cold_median_seconds": statistics.median(cold),
        "reuse_median_seconds": statistics.median(hot),
        "median_ratio": statistics.median(cold) / statistics.median(hot),
        "cache_info": tuple(map(str, gc.chart_cache_info())),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--hits", type=int, default=1000)
    args = parser.parse_args()
    if args.repeats <= 0 or args.hits <= 0:
        parser.error("repeat and hit counts must be positive")
    print(json.dumps(benchmark(args.repeats, args.hits), indent=2))
