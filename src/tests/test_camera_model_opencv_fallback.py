import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np

from src.model.camera_model import CameraModel


class _FakeCap:
    def __init__(self, index, *, opened=True):
        self.index = index
        self._opened = opened
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        # return a dummy BGR frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return True, frame

    def set(self, *_args, **_kwargs):
        return True

    def release(self):
        self.released = True
        self._opened = False


def _patch_ximea_fail(monkeypatch):
    """Force Ximea open_device_by_SN to fail."""
    class _FakeXiCam:
        def open_device_by_SN(self, *_args, **_kwargs):
            raise RuntimeError('no ximea')

    from ximea import xiapi
    monkeypatch.setattr(xiapi, 'Camera', lambda: _FakeXiCam())


def test_opencv_fallback_used_when_ximea_fails(monkeypatch):
    _patch_ximea_fail(monkeypatch)

    # Prefer index 0 but make it fail, index 1 succeeds
    monkeypatch.setenv('OPENCV_CAMERA_INDEX', '0')
    monkeypatch.setenv('OPENCV_CAMERA_MAX_INDEX', '1')

    import cv2

    attempts: list[int] = []
    created: dict[int, _FakeCap] = {}

    def _fake_videocap(index: int):
        attempts.append(index)
        cap = _FakeCap(index, opened=(index == 1))
        created[index] = cap
        return cap

    monkeypatch.setattr(cv2, 'VideoCapture', _fake_videocap)

    cm = CameraModel()
    cfg = getattr(cm, '_CameraModel__config_camera')
    cfg()

    assert attempts == [0, 1]
    assert 0 in created and created[0].released is True

    assert cm._camera_backend == 'opencv'
    assert cm._cv_cap is not None
    assert cm._cv_cap.index == 1

    frame = cm._grab_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.shape[2] == 3


def test_opencv_fallback_scan_order_when_preferred_not_zero(monkeypatch):
    """Preferred index should be tried first, then the remaining 0..max indices."""
    _patch_ximea_fail(monkeypatch)

    # Prefer index 2; it fails; index 0 fails; index 1 succeeds.
    monkeypatch.setenv('OPENCV_CAMERA_INDEX', '2')
    monkeypatch.setenv('OPENCV_CAMERA_MAX_INDEX', '2')

    import cv2

    attempts: list[int] = []
    created: dict[int, _FakeCap] = {}

    def _fake_videocap(index: int):
        attempts.append(index)
        cap = _FakeCap(index, opened=(index == 1))
        created[index] = cap
        return cap

    monkeypatch.setattr(cv2, 'VideoCapture', _fake_videocap)

    cm = CameraModel()
    cfg = getattr(cm, '_CameraModel__config_camera')
    cfg()

    # Preferred is tried first, then 0..max excluding preferred
    assert attempts == [2, 0, 1]

    # All failed ones should be released
    assert created[2].released is True
    assert created[0].released is True

    assert cm._camera_backend == 'opencv'
    assert cm._cv_cap is not None
    assert cm._cv_cap.index == 1

    frame = cm._grab_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.shape[2] == 3


def test_opencv_fallback_raises_with_tried_indices_when_none_available(monkeypatch):
    _patch_ximea_fail(monkeypatch)

    # Try indices 0..2 (preferred 0)
    monkeypatch.setenv('OPENCV_CAMERA_INDEX', '0')
    monkeypatch.setenv('OPENCV_CAMERA_MAX_INDEX', '2')

    import cv2

    attempts: list[int] = []
    created: dict[int, _FakeCap] = {}

    def _fake_videocap(index: int):
        attempts.append(index)
        cap = _FakeCap(index, opened=False)
        created[index] = cap
        return cap

    monkeypatch.setattr(cv2, 'VideoCapture', _fake_videocap)

    cm = CameraModel()
    cfg = getattr(cm, '_CameraModel__config_camera')

    import pytest

    with pytest.raises(RuntimeError) as ei:
        cfg()

    # Ensure it attempted the full scan and message contains tried indices
    assert attempts == [0, 1, 2]
    msg = str(ei.value)
    assert 'Tried indices' in msg
    assert '0' in msg and '1' in msg and '2' in msg

    # All should have been released
    assert created[0].released is True
    assert created[1].released is True
    assert created[2].released is True

