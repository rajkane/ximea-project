import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.model.camera_model import CameraModel


def test_camera_model_prefers_int8_onnx_when_present(tmp_path, monkeypatch):
    # Arrange: fake model path
    base = tmp_path / 'cap_model.pth'
    base.write_text('x')

    onnx_fp32 = tmp_path / 'cap_model.onnx'
    onnx_fp32.write_text('x')

    onnx_int8 = tmp_path / 'cap_model.int8.onnx'
    onnx_int8.write_text('x')

    # Ensure ONNX backend selection
    monkeypatch.setenv('INFERENCE_BACKEND_RESOLVED', 'onnx-cpu')

    # Patch OnnxDetector to capture path without needing onnxruntime
    captured = {}

    class FakeOnnxDetector:
        def __init__(self, onnx_path, classes, threads=4):
            captured['path'] = onnx_path
            self.onnx_path = onnx_path
            self.classes = classes
            self.threads = threads

        def predict(self, frame):
            return [], None, None

    monkeypatch.setitem(sys.modules, 'src.model.onnx_inference', __import__('types').SimpleNamespace(OnnxDetector=FakeOnnxDetector))

    cam = CameraModel()
    cam.set_is_model(True)
    cam.set_model(str(base))
    cam.set_annot(['Cap OK'])

    # Act
    cam.detector()

    # Assert
    assert captured['path'].endswith('cap_model.int8.onnx')
