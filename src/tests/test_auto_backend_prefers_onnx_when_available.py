import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import types


def test_auto_backend_prefers_onnx_when_files_exist(monkeypatch, tmp_path):
    from src.model import camera_model as cm

    # Create fake model paths; candidates will include .onnx
    pth = tmp_path / 'm.pth'
    pth.write_text('x')
    (tmp_path / 'm.onnx').write_text('x')

    class FakeOnnxDetector:
        def __init__(self, onnx_path, classes, threads=4):
            self.onnx_path = onnx_path
        def predict(self, frame):
            return [], None, None

    # Make torch path fail loudly if called
    def boom(*args, **kwargs):
        raise AssertionError('Torch loader should not be used when ONNX available in AUTO')

    fake_mod = types.ModuleType('src.model.onnx_inference')
    fake_mod.OnnxDetector = FakeOnnxDetector
    monkeypatch.setitem(sys.modules, 'src.model.onnx_inference', fake_mod)
    monkeypatch.setenv('INFERENCE_BACKEND_RESOLVED', 'auto')

    cam = cm.CameraModel()
    cam.set_is_model(True)
    cam.set_model(str(pth))
    cam.set_annot(['Cap OK'])

    monkeypatch.setattr(cm.CameraModel, '_load_torch_detector', boom)

    cam.detector()
    assert hasattr(cam.model, 'onnx_path')
    assert cam.model.onnx_path.endswith('.onnx')
