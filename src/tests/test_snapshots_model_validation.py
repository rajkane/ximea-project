import sys
import os

# Ensure repository root is importable (so `import src...` works)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.model.snapshots_model import SnapshotsModel


def test_snapshots_model_run_validates_required_fields(monkeypatch):
    sm = SnapshotsModel()

    # Prevent any real thread behavior
    monkeypatch.setattr(SnapshotsModel, 'isRunning', lambda self: False)

    captured = {'msg': None}

    class _Sig:
        def emit(self, msg):
            captured['msg'] = msg

    # Replace exception signal with a lightweight stub
    sm.exception = _Sig()

    sm.run()
    assert captured['msg'] is not None
    assert 'path is not set' in captured['msg']


def test_snapshots_model_creates_train_valid_dirs(monkeypatch, tmp_path):
    sm = SnapshotsModel()
    sm.set_path(str(tmp_path / 'ds'))
    sm.set_number(1)
    sm.set_gain_camera(1.0)
    sm.set_exposure_camera(100)

    # prevent thread waiting
    monkeypatch.setattr(SnapshotsModel, 'isRunning', lambda self: False)

    # Stub camera setup to avoid hardware and to terminate loop immediately
    class _DummyMutex:
        def lock(self):
            pass

        def unlock(self):
            pass

    class _DummyCam:
        def is_isexist(self):
            return False

    def _fake_config(self):
        self.mutex = _DummyMutex()
        self.cam = _DummyCam()

    monkeypatch.setattr(SnapshotsModel, '_SnapshotsModel__config_camera', _fake_config)
    monkeypatch.setattr(SnapshotsModel, '_SnapshotsModel__close_cam', lambda self: None)

    sm.run()

    assert (tmp_path / 'ds' / 'train').is_dir()
    assert (tmp_path / 'ds' / 'valid').is_dir()
