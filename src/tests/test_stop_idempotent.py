import sys
import os

# Ensure repository root is importable (so `import src...` works)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.model.camera_model import CameraModel
from src.model.snapshots_model import SnapshotsModel


def test_camera_model_stop_idempotent(monkeypatch):
    cm = CameraModel()

    # Pretend it's not running to avoid Qt dependency
    monkeypatch.setattr(CameraModel, 'isRunning', lambda self: False)
    monkeypatch.setattr(CameraModel, 'quit', lambda self: None)
    monkeypatch.setattr(CameraModel, 'wait', lambda self: None)

    # stop should not raise and should be safe multiple times
    cm.stop()
    cm.stop()


def test_snapshots_model_stop_idempotent(monkeypatch):
    sm = SnapshotsModel()

    monkeypatch.setattr(SnapshotsModel, 'isRunning', lambda self: False)
    monkeypatch.setattr(SnapshotsModel, 'quit', lambda self: None)
    monkeypatch.setattr(SnapshotsModel, 'wait', lambda self: None)

    sm.stop()
    sm.stop()
