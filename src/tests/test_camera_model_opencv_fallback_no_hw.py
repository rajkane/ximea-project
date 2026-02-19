import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import types

import pytest


def test_opencv_fallback_tries_indices_and_fails_cleanly(monkeypatch):
    """If Ximea is unavailable and no OpenCV camera can be opened, we should raise a clear error.

    This test does not require real camera hardware.
    """
    from src.model import camera_model as cm

    # Force Ximea setup to fail fast
    monkeypatch.setattr(cm.CameraModel, '_try_setup_ximea', lambda self: False)

    # Fake cv2.VideoCapture that always fails
    class FakeCap:
        def __init__(self, idx):
            self.idx = idx
        def isOpened(self):
            return False
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

    # Limit sweep
    monkeypatch.setenv('OPENCV_CAMERA_INDEX', '0')
    monkeypatch.setenv('OPENCV_CAMERA_MAX_INDEX', '2')

    cam = cm.CameraModel()

    with pytest.raises(RuntimeError) as e:
        cam._setup_opencv_fallback()

    assert 'Failed to open any OpenCV camera' in str(e.value)
