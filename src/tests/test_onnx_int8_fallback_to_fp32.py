import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import types


def test_onnx_int8_fallbacks_to_fp32_when_int8_not_supported(monkeypatch, tmp_path):
    from src.model import camera_model as cm

    pth = tmp_path / 'm.pth'
    pth.write_text('x')
    onnx_int8 = tmp_path / 'm.int8.onnx'
    onnx_fp32 = tmp_path / 'm.onnx'
    onnx_int8.write_text('x')
    onnx_fp32.write_text('x')

    attempts = []

    class FakeOnnxDetector:
        def __init__(self, onnx_path, classes, threads=4):
            attempts.append(onnx_path)
            if onnx_path.endswith('.int8.onnx'):
                raise RuntimeError('NOT_IMPLEMENTED: ConvInteger')
            self.onnx_path = onnx_path
        def predict(self, frame):
            return [], None, None

    monkeypatch.setitem(sys.modules, 'src.model.onnx_inference', types.SimpleNamespace(OnnxDetector=FakeOnnxDetector))
    monkeypatch.setenv('INFERENCE_BACKEND_RESOLVED', 'onnx-cpu')

    cam = cm.CameraModel()
    cam.set_is_model(True)
    cam.set_model(str(pth))
    cam.set_annot(['Cap OK'])

    ok = cam._try_load_onnx_detector('onnx-cpu')
    assert ok is True
    assert attempts[0].endswith('.int8.onnx')
    assert attempts[1].endswith('.onnx')
