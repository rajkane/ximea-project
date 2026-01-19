import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.model.backend import BackendAvailability, InferenceBackend, select_backend


def test_auto_prefers_cuda_when_available():
    avail = BackendAvailability(torch_available=True, torch_cuda_available=True, onnxruntime_available=True)
    assert select_backend(InferenceBackend.AUTO, avail) == InferenceBackend.TORCH_CUDA


def test_auto_falls_back_to_onnx_when_no_cuda_and_onnx_present():
    avail = BackendAvailability(torch_available=True, torch_cuda_available=False, onnxruntime_available=True)
    assert select_backend(InferenceBackend.AUTO, avail) == InferenceBackend.ONNX_CPU


def test_auto_falls_back_to_torch_cpu_when_no_cuda_and_no_onnx():
    avail = BackendAvailability(torch_available=True, torch_cuda_available=False, onnxruntime_available=False)
    assert select_backend(InferenceBackend.AUTO, avail) == InferenceBackend.TORCH_CPU


def test_request_cuda_falls_back_when_unavailable():
    avail = BackendAvailability(torch_available=True, torch_cuda_available=False, onnxruntime_available=False)
    assert select_backend(InferenceBackend.TORCH_CUDA, avail) == InferenceBackend.TORCH_CPU


def test_request_onnx_falls_back_when_unavailable():
    avail = BackendAvailability(torch_available=True, torch_cuda_available=True, onnxruntime_available=False)
    assert select_backend(InferenceBackend.ONNX_CPU, avail) == InferenceBackend.TORCH_CUDA
