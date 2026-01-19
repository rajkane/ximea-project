from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InferenceBackend(str, Enum):
    AUTO = "auto"
    TORCH_CUDA = "torch-cuda"
    TORCH_CPU = "torch-cpu"
    ONNX_CPU = "onnx-cpu"


@dataclass(frozen=True)
class BackendAvailability:
    torch_available: bool
    torch_cuda_available: bool
    onnxruntime_available: bool


def probe_backends() -> BackendAvailability:
    """Probe what's installed/available without importing heavy stuff unless needed."""

    torch_available = False
    torch_cuda_available = False
    try:
        import torch  # noqa: F401

        torch_available = True
        try:
            import torch as _torch

            torch_cuda_available = bool(_torch.cuda.is_available())
        except Exception:
            torch_cuda_available = False
    except Exception:
        torch_available = False
        torch_cuda_available = False

    try:
        from importlib import util as importlib_util

        onnxruntime_available = importlib_util.find_spec("onnxruntime") is not None
    except Exception:
        onnxruntime_available = False

    return BackendAvailability(
        torch_available=torch_available,
        torch_cuda_available=torch_cuda_available,
        onnxruntime_available=onnxruntime_available,
    )


def _normalize_requested(requested: InferenceBackend | str | None) -> InferenceBackend:
    if requested is None:
        return InferenceBackend.AUTO
    if isinstance(requested, InferenceBackend):
        return requested

    value = str(requested).strip().lower()
    return InferenceBackend(value)


def _select_auto(availability: BackendAvailability) -> InferenceBackend:
    """AUTO policy: CUDA -> ONNX CPU -> Torch CPU."""
    if availability.torch_available and availability.torch_cuda_available:
        return InferenceBackend.TORCH_CUDA
    if availability.onnxruntime_available:
        return InferenceBackend.ONNX_CPU
    return InferenceBackend.TORCH_CPU


def _cuda_allowed(availability: BackendAvailability) -> bool:
    return availability.torch_available and availability.torch_cuda_available


def _onnx_allowed(availability: BackendAvailability) -> bool:
    return availability.onnxruntime_available


def select_backend(
    requested: InferenceBackend | str | None,
    availability: BackendAvailability | None = None,
) -> InferenceBackend:
    """Select backend.

    Policy (your choice '1'):
    - AUTO prefers TORCH_CUDA, else ONNX_CPU, else TORCH_CPU.

    Notes:
    - We don't automatically select ONNX unless `onnxruntime` is installed.
    - This doesn't require that an ONNX model file exists; that's validated at load time.
    """

    availability = availability or probe_backends()

    requested_enum = _normalize_requested(requested)
    if requested_enum == InferenceBackend.AUTO:
        return _select_auto(availability)

    if requested_enum == InferenceBackend.TORCH_CUDA:
        return InferenceBackend.TORCH_CUDA if _cuda_allowed(availability) else _select_auto(availability)

    if requested_enum == InferenceBackend.ONNX_CPU:
        return InferenceBackend.ONNX_CPU if _onnx_allowed(availability) else _select_auto(availability)

    # TORCH_CPU (or any future explicit CPU variant)
    return InferenceBackend.TORCH_CPU
