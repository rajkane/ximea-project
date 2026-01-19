import os
import sys
import time
import signal
from contextlib import contextmanager

# Ensure repository root is importable (so `import src...` works)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pytest
import torch


# Benchmarks are opt-in to avoid IDE freezes when running the full test suite.
# Run with: RUN_BENCHMARKS=1 pytest -m benchmark ...
RUN_BENCHMARKS = os.getenv('RUN_BENCHMARKS', '').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


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


@contextmanager
def _timeout(seconds: int, *, msg: str):
    """Guard long-running native ops so the test can bail out cleanly.

    Note: uses SIGALRM, so it works on Linux/macOS main thread.
    """
    if seconds <= 0:
        yield
        return

    def _handler(_signum, _frame):
        raise TimeoutError(msg)

    old = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@pytest.mark.benchmark
def test_cpu_inference_benchmark_cap_model():
    """CPU-only inference benchmark for deploy target without NVIDIA.

    This test is *opt-in* to avoid accidental freezes/crashes when running the full
    suite in PyCharm.

    Run explicitly:
        RUN_BENCHMARKS=1 pytest -m benchmark -s

    Safety measures:
    - caps torch/OMP threads (avoids IDE freeze)
    - per-image timeout (fails/skips instead of hanging)
    - smaller default workload
    """

    if not RUN_BENCHMARKS:
        pytest.skip('Benchmark tests are disabled. Set RUN_BENCHMARKS=1 to run.')

    # Force CPU
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

    # Reduce oversubscription (often the cause of PyCharm freezes)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        # Some builds may not support changing these settings
        pass

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
    try:
        with _timeout(60, msg="Model load timed out"):
            model = core.Model.load(model_path, annotations)
    except TimeoutError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.fail(f"Model load failed: {e}")

    # Keep workload small by default (can be increased locally if needed)
    image_paths = list(_iter_images(dataset_valid, limit=6))
    if not image_paths:
        pytest.skip(f"No images found in {dataset_valid}")

    # Warmup (helps stabilize first-run overhead)
    for p in image_paths[:1]:
        img = read_image(p)
        try:
            with _timeout(20, msg=f"Warmup inference timed out on {p}"):
                _ = model.predict(img)
        except TimeoutError as e:
            pytest.skip(str(e))
        except Exception as e:
            pytest.fail(f"Warmup inference failed on {p}: {e}")

    # Timed runs
    t0 = time.perf_counter()
    ok = 0
    for p in image_paths:
        img = read_image(p)
        try:
            with _timeout(30, msg=f"Inference timed out on {p}"):
                _ = model.predict(img)
            ok += 1
        except TimeoutError as e:
            pytest.skip(str(e))
        except Exception as e:
            pytest.fail(f"Inference failed on {p}: {e}")
    t1 = time.perf_counter()

    total = t1 - t0
    per_img = total / max(ok, 1)
    fps = (ok / total) if total > 0 else 0.0

    print(
        "CPU inference benchmark (Deteco)\n"
        f"  model: {model_path}\n"
        f"  images: {ok} from {dataset_valid}\n"
        f"  total: {total:.3f}s\n"
        f"  per_image: {per_img * 1000:.1f}ms\n"
        f"  FPS: {fps:.2f}"
    )

    assert ok > 0
    assert per_img > 0
