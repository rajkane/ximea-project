"""Small runner to execute the ONNX benchmark outside pytest.

Why:
- Some IDE / test runners may suppress pytest stdout.
- This script runs the exact same benchmark test function and always writes
  results to Models/onnx_benchmark.txt.

Usage:
  RUN_BENCHMARKS=1 BENCH_THREAD_SWEEP=1 BENCH_IMAGE_LIMIT=3 python -m src.model.run_onnx_benchmark

"""

from __future__ import annotations

import os
import sys


def main() -> int:
    # Ensure repository root is importable
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from src.tests.test_onnx_cpu_benchmark import test_pytorch_cpu_vs_onnxruntime_cpu_benchmark

    # The benchmark uses pytest.skip for opt-in; if user didn't set RUN_BENCHMARKS=1
    # we just print a friendly message.
    if os.getenv('RUN_BENCHMARKS', '').strip().lower() not in {'1', 'true', 'yes', 'y', 'on'}:
        print('Benchmark is disabled. Set RUN_BENCHMARKS=1 to run.')
        return 2

    try:
        test_pytorch_cpu_vs_onnxruntime_cpu_benchmark()
    except BaseException as e:
        # We don't want to crash a GUI/IDE session.
        print(f'Benchmark failed: {e}')
        return 1

    out_path = os.path.abspath(os.path.join(repo_root, 'Models', 'onnx_benchmark.txt'))
    print(f'Wrote results to: {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
