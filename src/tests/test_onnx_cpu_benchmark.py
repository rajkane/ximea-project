import os
import sys
import time
import signal
from contextlib import contextmanager

import pytest
import torch

# Ensure repository root is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


RUN_BENCHMARKS = os.getenv('RUN_BENCHMARKS', '').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
BENCH_IMAGE_LIMIT = int(os.getenv('BENCH_IMAGE_LIMIT', '6'))
REEXPORT_ONNX = os.getenv('REEXPORT_ONNX', '').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}

# Optional thread sweep. Example: BENCH_THREAD_SWEEP=1
# Keeps default safe behavior unless explicitly enabled.
THREAD_SWEEP = os.getenv('BENCH_THREAD_SWEEP', '').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
THREAD_CANDIDATES = [1, 2, 4]


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


def _iter_images(folder: str, limit: int = 6):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    count = 0
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name.lower())[1] in exts:
            yield os.path.join(folder, name)
            count += 1
            if count >= limit:
                break


def _to_chw_float32(np_img):
    """HWC uint8 -> CHW float32 in range [0,1]."""
    import numpy as np

    if not isinstance(np_img, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray image, got {type(np_img)}")

    t = torch.from_numpy(np_img).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
    return t


def _configure_cpu_threads(n: int) -> None:
    """Set process/thread env vars + torch threads.

    Note: ORT takes threads via SessionOptions, so we handle that elsewhere.
    """
    n = max(int(n), 1)
    os.environ['OMP_NUM_THREADS'] = str(n)
    os.environ['MKL_NUM_THREADS'] = str(n)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n)
    try:
        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _time_pytorch_predict(det, imgs, *, timeout_s: int = 60) -> tuple[float, float]:
    """Return (total_seconds, per_image_seconds)."""
    # Warmup
    with _timeout(30, msg='Warmup timed out'):
        _ = det.predict(imgs[0])

    t0 = time.perf_counter()
    for img in imgs:
        with _timeout(timeout_s, msg='PyTorch predict timed out'):
            _ = det.predict(img)
    t1 = time.perf_counter()

    total = t1 - t0
    return total, total / len(imgs)


@pytest.mark.benchmark
def test_pytorch_cpu_vs_onnxruntime_cpu_benchmark():
    """Benchmarks PyTorch CPU predict vs ONNX Runtime CPU (if export succeeds).

    Opt-in:
      RUN_BENCHMARKS=1

    Optional env knobs:
      BENCH_IMAGE_LIMIT=6         (default 6)
      REEXPORT_ONNX=1             (force re-export even if Models/cap_model.onnx exists)
      BENCH_THREAD_SWEEP=1        (measure 1/2/4 threads for both torch and ORT)

    Notes:
    - For torchvision detection models, ONNX export may fail depending on opset/version.
      If it fails we SKIP with the reason instead of failing.
    """

    if not RUN_BENCHMARKS:
        pytest.skip('Benchmark tests are disabled. Set RUN_BENCHMARKS=1 to run.')

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

    try:
        import onnxruntime as ort
    except Exception as e:
        pytest.skip(f'onnxruntime not importable: {e}')

    try:
        from detecto import core
        from detecto.utils import read_image
    except Exception as e:
        pytest.skip(f'detecto not importable: {e}')

    model_path = os.path.abspath(os.path.join(repo_root, 'Models', 'cap_model.pth'))
    ann_path = model_path + '.txt'
    dataset_valid = os.path.abspath(os.path.join(repo_root, 'Dataset', 'cap_dataset', 'valid'))

    if not os.path.isfile(model_path) or not os.path.isfile(ann_path) or not os.path.isdir(dataset_valid):
        pytest.skip('Required model/annotations/dataset not found for benchmark.')

    annotations = _read_annotations(ann_path)
    if not annotations:
        pytest.skip('Empty annotations file; cannot load model.')

    with _timeout(60, msg='Model load timed out'):
        det = core.Model.load(model_path, annotations)

    # Pick images
    image_paths = list(_iter_images(dataset_valid, limit=BENCH_IMAGE_LIMIT))
    if not image_paths:
        pytest.skip('No images found for benchmark.')

    imgs = [read_image(p) for p in image_paths]

    # --- Locate underlying torch model for ONNX export ---
    base = None
    for name in ('model', 'detector', '_model'):
        m = getattr(det, name, None)
        if isinstance(m, torch.nn.Module):
            base = m
            break
    if base is None:
        for _name, v in getattr(det, '__dict__', {}).items():
            if isinstance(v, torch.nn.Module):
                base = v
                break

    if base is None:
        pytest.skip('Could not locate underlying torch model for ONNX export.')

    base.eval()

    # ONNX export expects tensor input(s). For detection models, input is List[Tensor].
    sample = _to_chw_float32(imgs[0])

    onnx_path = os.path.abspath(os.path.join(repo_root, 'Models', 'cap_model.onnx'))

    need_export = REEXPORT_ONNX or (not os.path.isfile(onnx_path))
    if need_export:
        _configure_cpu_threads(1)
        try:
            with _timeout(180, msg='ONNX export timed out'):
                torch.onnx.export(
                    base,
                    args=([sample],),
                    f=onnx_path,
                    input_names=['images'],
                    output_names=['outputs'],
                    opset_version=12,
                    do_constant_folding=True,
                )
        except TimeoutError as e:
            pytest.skip(str(e))
        except Exception as e:
            pytest.skip(f'ONNX export failed: {e}')

    if not os.path.isfile(onnx_path):
        pytest.skip('ONNX export did not produce a file.')

    # Create ORT session (CPU) - created per-thread setting (sweep) for fair comparison.
    def _make_sess(threads: int):
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        return ort.InferenceSession(onnx_path, sess_options=so, providers=['CPUExecutionProvider'])

    def _infer_input_signature(sess):
        inp = sess.get_inputs()[0]
        expected_rank = None
        try:
            expected_rank = len(inp.shape)
        except Exception:
            expected_rank = None
        return inp.name, inp.shape, expected_rank

    def _to_ort_input(np_img, expected_rank):
        chw = _to_chw_float32(np_img)
        arr3 = chw.numpy()  # (C,H,W)
        if expected_rank == 3:
            return arr3
        if expected_rank == 4:
            return arr3[None, ...]
        return arr3[None, ...]

    def _time_ort(sess, inp_name, expected_rank, imgs_np, *, timeout_s: int = 60) -> tuple[float, float]:
        # warmup
        with _timeout(60, msg='ORT warmup timed out'):
            _ = sess.run(None, {inp_name: _to_ort_input(imgs_np[0], expected_rank)})

        t0 = time.perf_counter()
        for im in imgs_np:
            with _timeout(timeout_s, msg='ORT run timed out'):
                _ = sess.run(None, {inp_name: _to_ort_input(im, expected_rank)})
        t1 = time.perf_counter()
        total = t1 - t0
        return total, total / len(imgs_np)

    # Benchmark
    candidates = THREAD_CANDIDATES if THREAD_SWEEP else [1]
    rows: list[tuple[int, float, float, float, float]] = []

    last_inp_shape: object = 'unknown'
    last_expected_rank: object = 'unknown'
    pytorch_per_last = None
    ort_per_last = None

    for thr in candidates:
        _configure_cpu_threads(thr)
        pt_total, pt_per = _time_pytorch_predict(det, imgs)

        try:
            sess = _make_sess(thr)
        except Exception as e:
            pytest.skip(f'Failed to create ONNX Runtime session: {e}')

        inp_name, inp_shape, expected_rank = _infer_input_signature(sess)

        try:
            ort_total, ort_per = _time_ort(sess, inp_name, expected_rank, imgs)
        except Exception as e:
            pytest.skip(f'ORT run failed: {e} (input shape expected: {inp_shape})')

        rows.append((thr, pt_total, pt_per, ort_total, ort_per))

        # keep the single-run variables for summary
        last_inp_shape, last_expected_rank = inp_shape, expected_rank
        pytorch_per_last, ort_per_last = pt_per, ort_per

    # sanity
    assert rows

    # Build summary
    header = "thr | pytorch_ms | onnx_ms | speedup\n"
    table_lines = []
    for thr, _pt_total, pt_per, _ot_total, ot_per in rows:
        sp = (pt_per / ot_per) if ot_per > 0 else float('inf')
        table_lines.append(f"{thr:>3} | {pt_per*1000:>9.1f} | {ot_per*1000:>7.1f} | {sp:>7.3f}x")

    summary = (
        "CPU benchmark results\n"
        f"  timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  env: BENCH_IMAGE_LIMIT={BENCH_IMAGE_LIMIT}, BENCH_THREAD_SWEEP={int(THREAD_SWEEP)}, REEXPORT_ONNX={int(REEXPORT_ONNX)}\n"
        f"  images: {len(imgs)}\n"
        f"  onnx_input_shape: {last_inp_shape} (rank={last_expected_rank})\n"
        f"  onnx_file: {onnx_path}\n\n"
        + header
        + "\n".join(table_lines)
        + "\n"
    )

    print(summary)

    # Persist results for IDEs that suppress stdout.
    try:
        out_path = os.path.abspath(os.path.join(repo_root, 'Models', 'onnx_benchmark.txt'))
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(summary)
    except Exception:
        pass

    # sanity
    assert pytorch_per_last is not None and pytorch_per_last > 0
    assert ort_per_last is not None and ort_per_last > 0
