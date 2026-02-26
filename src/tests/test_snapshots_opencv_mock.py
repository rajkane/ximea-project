import pytest
import types

import cv2

from src.model.snapshots_model import SnapshotsModel


class FakeCap:
    def __init__(self, idx=0):
        self._opened = True
        # store values set via set(prop, val)
        self._props = {}

    def isOpened(self):
        return True

    def set(self, prop, val):
        # emulate storing and allowing get to return same value
        self._props[prop] = val
        return True

    def get(self, prop):
        # return stored value if present, else None
        return self._props.get(prop, None)

    def read(self):
        # return a dummy valid frame
        import numpy as np
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        return True, frame

    def release(self):
        self._opened = False


@pytest.fixture(autouse=True)
def patch_videocapture(monkeypatch):
    monkeypatch.setattr(cv2, 'VideoCapture', lambda idx, apiPreference=None: FakeCap(idx))
    yield


def test_apply_exposure_and_gain_emits_status(monkeypatch):
    sm = SnapshotsModel()

    # Capture emitted status messages
    msgs = []
    sm.status.connect(lambda m: msgs.append(m))

    # configure
    sm.set_path('tmp')
    sm.set_number(1)
    sm.set_interval_camera(0)

    # simulate opening opencv camera
    sm._config_opencv()
    assert sm.cam_type == 'opencv'

    # Set exposure and gain via setters - should apply immediately
    sm.set_exposure_camera(100000)  # 100 ms in microseconds
    sm.set_gain_camera(5.5)

    # verify underlying fake capture has values set
    # CAP_PROP_EXPOSURE numeric constant exists in cv2
    exp_val = sm.cam_cv.get(cv2.CAP_PROP_EXPOSURE)
    gain_val = sm.cam_cv.get(cv2.CAP_PROP_GAIN)

    assert exp_val is not None
    assert gain_val is not None

    # status messages should include exposure/gain tries
    assert any('OpenCV exposure' in m for m in msgs)
    assert any('OpenCV gain' in m for m in msgs)

    # cleanup
    sm._close_opencv()

