from src.external import sys, qtw
from src.controller.camera_controller import MainWindow


def _parse_backend_from_argv(argv: list[str]) -> str | None:
    """Parse --backend=<value> or --backend <value> from argv.

    Supported values:
    - auto
    - torch-cuda
    - torch-cpu
    - onnx-cpu
    """
    for i, a in enumerate(argv):
        if a.startswith("--backend="):
            return a.split("=", 1)[1].strip()
        if a == "--backend" and i + 1 < len(argv):
            return argv[i + 1].strip()
    return None


def _configure_inference_backend(argv: list[str]) -> None:
    """Choose backend once at process start.

    Priority:
      1) CLI: --backend
      2) ENV: INFERENCE_BACKEND
      3) AUTO: CUDA -> ONNX -> CPU

    The resolved backend is stored into env var `INFERENCE_BACKEND_RESOLVED`.

    NOTE: This does not implement ONNX inference yet; it only standardizes selection.
    """
    import os

    from src.model.backend import InferenceBackend, select_backend

    requested = _parse_backend_from_argv(argv) or os.getenv("INFERENCE_BACKEND")

    try:
        resolved = select_backend(requested)
    except Exception:
        # If user passes an invalid value, fall back to AUTO.
        resolved = select_backend(InferenceBackend.AUTO)

    os.environ["INFERENCE_BACKEND_RESOLVED"] = resolved.value

    # If user explicitly requests CPU, also hide CUDA devices so accidental CUDA ops don't fire.
    if resolved == InferenceBackend.TORCH_CPU:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def run():
    _configure_inference_backend(sys.argv)

    app = qtw.QApplication(sys.argv)
    app.setApplicationVersion("v1.0")
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()