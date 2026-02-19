"""Headless camera probe.

Purpose:
- Diagnose camera startup when Ximea camera is not present.
- Exercises CameraModel.__config_camera() fallback path without starting the full GUI.

Usage:
  python -m src.model.debug_camera_probe

Env vars:
- OPENCV_CAMERA_INDEX
- OPENCV_CAMERA_MAX_INDEX
"""

from __future__ import annotations

import argparse
import os

from src.model.camera_model import CameraModel


def _read_annot_file(path: str) -> list[str]:
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()
    if not raw:
        return []
    # Supports either one-per-line or 'A, B'
    if '\n' in raw:
        return [line.strip() for line in raw.splitlines() if line.strip()]
    return [s.strip().strip('"').strip("'") for s in raw.split(',') if s.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--backend', default='onnx-cpu', help='auto|torch-cpu|torch-cuda|onnx-cpu')
    p.add_argument('--model', default=os.path.abspath('Models/caps.pth'))
    p.add_argument('--annot', default=os.path.abspath('Models/caps.pth.txt'))
    p.add_argument('--max-index', type=int, default=10)
    args = p.parse_args(argv)

    os.environ['INFERENCE_BACKEND_RESOLVED'] = str(args.backend)
    os.environ['OPENCV_CAMERA_INDEX'] = os.environ.get('OPENCV_CAMERA_INDEX', '0')
    os.environ['OPENCV_CAMERA_MAX_INDEX'] = str(args.max_index)

    cam = CameraModel()

    def on_status(msg: str):
        print('STATUS:', msg)

    def on_exc(msg: str):
        print('EXC:', msg)

    cam.status.connect(on_status)
    cam.exception.connect(on_exc)

    # Configure model
    if args.model and args.annot:
        cam.set_is_model(True)
        cam.set_model(args.model)
        cam.set_annot(_read_annot_file(args.annot))

    try:
        cfg = getattr(cam, '_CameraModel__config_camera')
        cfg()
        cam.detector()
        print('RESULT: backend=', cam._camera_backend)
        return 0
    except Exception as e:
        print('RESULT: exception=', repr(e))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
