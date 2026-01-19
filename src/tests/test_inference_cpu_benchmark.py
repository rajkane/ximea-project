import os
import sys
import time

# Ensure repository root is importable (so `import src...` works)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pytest
import torch


def _read_annotations(txt_path: str) -> list[str]:
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    # supports both newline-separated and comma-separated formats
    if not content:
        return []
    if '\n' in content:
        return [line.strip() for line in content.splitlines() if line.strip()]
    return [s.strip().strip('"').strip("'") for s in content.split(',') if s.strip()]


def _iter_images(folder: str, limit: int = 20):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    count = 0
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name.lower())[1] in exts:
            yield os.path.join(folder, name)
            count += 1
            if count >= limit:
                break


def test_cpu_inference_benchmark_cap_model():
    """CPU-only inference benchmark for deploy target without NVIDIA.

    This doesn't assert a specific speed (depends on hardware). It prints timing.
    Run explicitly when you want numbers:
        pytest -m benchmark -s
    """

    # Force CPU
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

    model_path = os.path.abspath(os.path.join(repo_root, 'Models', 'cap_model.pth'))
    ann_path = model_path + '.txt'
    dataset_valid = os.path.abspath(os.path.join(repo_root, 'Dataset', 'cap_dataset', 'valid'))

    if not os.path.isfile(model_path):
        pytest.skip(f"Model not found: {model_path}")
    if not os.path.isfile(ann_path):
        pytest.skip(f"Annotations not found: {ann_path}")
    if not os.path.isdir(dataset_valid):
        pytest.skip(f"Valid dataset folder not found: {dataset_valid}")

    annotations = _read_annotations(ann_path)
    if not annotations:
        pytest.skip(f"Empty annotations file: {ann_path}")

    try:
        from detecto import core
        from detecto.utils import read_image
    except Exception as e:
        pytest.skip(f"detecto not importable: {e}")

    # Load model (Deteco wraps FasterRCNN)
    model = core.Model.load(model_path, annotations)

    # Warmup + timing
    image_paths = list(_iter_images(dataset_valid, limit=10))
    if not image_paths:
        pytest.skip(f"No images found in {dataset_valid}")

    # Warmup (helps stabilize first-run overhead)
    for p in image_paths[:2]:
        img = read_image(p)
        _ = model.predict(img)

    # Timed runs
    t0 = time.perf_counter()
    for p in image_paths:
        img = read_image(p)
        _ = model.predict(img)
    t1 = time.perf_counter()

    total = t1 - t0
    per_img = total / len(image_paths)
    fps = (len(image_paths) / total) if total > 0 else 0.0

    # Print numbers for you (pytest -s)
    print(f"CPU inference benchmark (Deteco)\n"
          f"  model: {model_path}\n"
          f"  images: {len(image_paths)} from {dataset_valid}\n"
          f"  total: {total:.3f}s\n"
          f"  per_image: {per_img*1000:.1f}ms\n"
          f"  FPS: {fps:.2f}")

    # Basic sanity check: it shouldn't be astronomically slow or negative
    assert per_img > 0
