import sys
import os

# Ensure repository root is importable (so `import src...` works)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import types
import numpy as np

from src.model.camera_model import CameraModel


class _DummyMutex:
    def lock(self):
        pass
    def unlock(self):
        pass


class _DummyCam:
    def __init__(self):
        self._exists = True
    def is_isexist(self):
        # first loop true, then false to exit
        if self._exists:
            self._exists = False
            return True
        return False
    def get_image(self, img):
        return None
    def set_exp_priority(self, v):
        return None
    def enable_aeag(self):
        return None
    def set_gain(self, g):
        return None
    def set_exposure(self, e):
        return None


class _DummyImg:
    def get_image_data_numpy(self, _):
        return np.zeros((100, 200, 3), dtype=np.uint8)


def test_detector_called_once_per_run(monkeypatch):
    cm = CameraModel()
    cm.set_is_model(True)
    cm.set_model('dummy.pth')
    cm.set_annot(['Cap OK'])
    cm.set_exposure_camera(0)

    # Make sure no real sleeping happens in the loop
    import time as _time
    monkeypatch.setattr(_time, 'sleep', lambda *_args, **_kwargs: None)

    # avoid hardware init
    def _fake_config(self):
        self.mutex = _DummyMutex()
        self.cam = _DummyCam()
        self.img = _DummyImg()
        self._CameraModel__sfps = types.SimpleNamespace(fps=lambda **_: '0')

    monkeypatch.setattr(CameraModel, '_CameraModel__config_camera', _fake_config)
    monkeypatch.setattr(CameraModel, '_CameraModel__close_cam', lambda self: None)
    monkeypatch.setattr(CameraModel, 'show_fps', lambda self, img: None)
    monkeypatch.setattr(CameraModel, 'detection', lambda self, img: None)
    monkeypatch.setattr(CameraModel, 'convert_frame', lambda self, img: None)

    calls = {'n': 0}
    def _fake_detector(self):
        calls['n'] += 1
        self._detector_ready = True

    monkeypatch.setattr(CameraModel, 'detector', _fake_detector)

    cm.run()
    assert calls['n'] == 1
