import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import types


def test_opencv_fallback_selects_first_available(monkeypatch):
    """When Ximea is unavailable, we should fall back to the first OpenCV camera that opens."""
    from src.model import camera_model as cm

    monkeypatch.setattr(cm.CameraModel, '_try_setup_ximea', lambda self: False)

    opened = {0: False, 1: True}

    class FakeCap:
        def __init__(self, idx):
            self.idx = idx
        def isOpened(self):
            return bool(opened.get(self.idx, False))
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
    monkeypatch.setenv('OPENCV_CAMERA_INDEX', '0')
    monkeypatch.setenv('OPENCV_CAMERA_MAX_INDEX', '2')

    cam = cm.CameraModel()
    cam._setup_opencv_fallback()

    assert cam._camera_backend == 'opencv'
    assert cam._cv_cap is not None
    assert cam._cv_cap.idx == 1
