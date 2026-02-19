import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import types


def test_debug_probe_onnx_backend_with_opencv_fallback(monkeypatch, tmp_path):
    """Simulate: no Ximea, OpenCV camera opens, ONNX backend selected.

    We don't require onnxruntime or real camera hardware. The goal is to ensure the
    control flow reaches OpenCV backend selection and ONNX detector init is attempted.
    """
    from src.model import camera_model as cm

    # Fake Ximea failing
    monkeypatch.setattr(cm.CameraModel, '_try_setup_ximea', lambda self: False)

    # Fake OpenCV camera that opens at index 0
    class FakeCap:
        def __init__(self, idx):
            self.idx = idx
        def isOpened(self):
            return True
        def release(self):
            pass
        def read(self):
            return False, None
        def set(self, *args, **kwargs):
            return True

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=lambda idx: FakeCap(idx),
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
    )
    monkeypatch.setattr(cm, 'cv2', fake_cv2)

    # Create fake model files
    pth = tmp_path / 'caps.pth'
    pth.write_text('x')
    # Presence of int8 onnx drives selection inside CameraModel
    (tmp_path / 'caps.int8.onnx').write_text('x')
    (tmp_path / 'caps.onnx').write_text('x')

    # Patch OnnxDetector to avoid onnxruntime
    captured = {}

    class FakeOnnxDetector:
        def __init__(self, onnx_path, classes, threads=4):
            captured['onnx_path'] = onnx_path
            self.onnx_path = onnx_path
            self.classes = classes
        def predict(self, frame):
            return [], None, None

    monkeypatch.setitem(sys.modules, 'src.model.onnx_inference', types.SimpleNamespace(OnnxDetector=FakeOnnxDetector))

    monkeypatch.setenv('INFERENCE_BACKEND_RESOLVED', 'onnx-cpu')
    monkeypatch.setenv('OPENCV_CAMERA_INDEX', '0')
    monkeypatch.setenv('OPENCV_CAMERA_MAX_INDEX', '2')

    cam = cm.CameraModel()
    cam.set_is_model(True)
    cam.set_model(str(pth))
    cam.set_annot(['Cap OK'])

    # Configure camera backend
    cam._setup_opencv_fallback()
    assert cam._camera_backend == 'opencv'

    # Load detector
    cam.detector()
    assert captured['onnx_path'].endswith('caps.int8.onnx')
