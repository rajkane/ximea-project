"""Inspect ONNX model metadata (quantization heuristic).

This does *not* modify the model.

Usage:
  python -m src.model.inspect_onnx_model Models/cap_model.onnx

Exit codes:
  0 = ok
  2 = file not found / missing args
"""

from __future__ import annotations

import os
import sys


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print('Usage: python -m src.model.inspect_onnx_model <path_to_model.onnx>')
        return 2

    path = argv[1]
    if not os.path.isfile(path):
        print(f'File not found: {path}')
        return 2

    # classes are not needed for the probe
    from src.model.onnx_inference import OnnxDetector

    det = OnnxDetector(path, classes=[], threads=1)
    print(f'onnx_path: {path}')
    print(f'is_quantized: {det.is_quantized}')
    print(f'initializer_dtypes: {det.model_dtype_summary()}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
