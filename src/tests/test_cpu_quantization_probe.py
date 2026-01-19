import os
import sys
import signal
from contextlib import contextmanager

import pytest
import torch

# Ensure repository root is importable (so `import src...` works)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


RUN_BENCHMARKS = os.getenv('RUN_BENCHMARKS', '').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


@contextmanager
def _timeout(seconds: int, *, msg: str):
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


def _read_annotations(txt_path: str) -> list[str]:
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        return []
    if '\n' in content:
        return [line.strip() for line in content.splitlines() if line.strip()]
    return [s.strip().strip('"').strip("'") for s in content.split(',') if s.strip()]


def _find_underlying_torch_model(det_obj) -> torch.nn.Module | None:
    """Best-effort extraction of the underlying torch.nn.Module from a Detecto Model.

    Detecto versions differ; common attribute names include: model, detector, _model.
    As a fallback, we scan __dict__ and pick the first torch.nn.Module.
    """

    for name in ('model', 'detector', '_model'):
        m = getattr(det_obj, name, None)
        if isinstance(m, torch.nn.Module):
            return m

    # Fallback: scan attributes
    for name, value in getattr(det_obj, '__dict__', {}).items():
        if isinstance(value, torch.nn.Module):
            return value

    return None


@pytest.mark.benchmark
def test_cpu_dynamic_quantization_probe_fasterrcnn():
    """Probe whether *dynamic quantization* works for the Detecto FasterRCNN model on CPU.

    Important:
    - This is NOT guaranteed to work for torchvision detection models.
    - The goal is to give a factual answer on *your* environment/model.

    The test is opt-in to avoid IDE freezes:
        RUN_BENCHMARKS=1 pytest -m benchmark -s -k quantization_probe

    Pass criteria:
    - If quantization works, we can run a forward/predict on a sample image.
    - If it doesn't, we skip with a clear reason (no crash).
    """

    if not RUN_BENCHMARKS:
        pytest.skip('Benchmark tests are disabled. Set RUN_BENCHMARKS=1 to run.')

    # Force CPU and cap threads
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    model_path = os.path.abspath(os.path.join(repo_root, 'Models', 'cap_model.pth'))
    ann_path = model_path + '.txt'
    dataset_valid = os.path.abspath(os.path.join(repo_root, 'Dataset', 'cap_dataset', 'valid'))

    if not os.path.isfile(model_path) or not os.path.isfile(ann_path) or not os.path.isdir(dataset_valid):
        pytest.skip('Required model/annotations/dataset not found for quantization probe.')

    annotations = _read_annotations(ann_path)
    if not annotations:
        pytest.skip('Empty annotations file; cannot load detecto model.')

    try:
        from detecto import core
        from detecto.utils import read_image
    except Exception as e:
        pytest.skip(f'detecto not importable: {e}')

    # Load detecto wrapper
    try:
        with _timeout(60, msg='Model load timed out'):
            det = core.Model.load(model_path, annotations)
    except TimeoutError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.fail(f'Model load failed: {e}')

    base = _find_underlying_torch_model(det)
    if base is None:
        pytest.skip('Could not locate underlying torch model inside detecto.Model on this version.')

    base.eval()

    # Pick 1 image
    img_path = None
    for name in sorted(os.listdir(dataset_valid)):
        if os.path.splitext(name.lower())[1] in {'.jpg', '.jpeg', '.png', '.bmp'}:
            img_path = os.path.join(dataset_valid, name)
            break
    if not img_path:
        pytest.skip('No valid images found to run the forward pass.')

    img = read_image(img_path)

    # Sanity: base predict works via Detecto API
    with _timeout(30, msg='Base inference timed out'):
        _ = det.predict(img)

    # Dynamic quantization attempt (least invasive)
    # Note: torchvision FasterRCNN contains many conv layers; dynamic quantization
    # usually targets Linear/RNN modules, so this likely won't help or may not be supported.
    try:
        qmodel = torch.ao.quantization.quantize_dynamic(
            base,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
    except Exception as e:
        pytest.skip(f'Dynamic quantization could not be applied: {e}')

    qmodel.eval()

    # Try running the quantized underlying model directly.
    # torchvision detection models accept List[Tensor] inputs.
    try:
        with torch.inference_mode():
            with _timeout(40, msg='Quantized forward timed out'):
                _ = qmodel([img])
    except TimeoutError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f'Quantized forward failed (quantization not usable here): {e}')

    # If we got here, quantized forward at least runs.
    assert True
