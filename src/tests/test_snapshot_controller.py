import sys
import os

# Ensure repository root is importable (so `import src...` works)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import gc
from src.controller.snapshot_controller import SnapshotDialog, SnapshotsModel


class _DummyEvent:
    def __init__(self):
        self.accepted = False

    def accept(self):
        self.accepted = True


class _DummySnapshotsModel(SnapshotsModel):
    def __init__(self):
        super().__init__()
        self.stopped = False

    def isRunning(self):
        return True

    def stop(self):
        self.stopped = True


def test_snapshot_dialog_close_event_stops_model(monkeypatch):
    # Avoid needing a QApplication by not running QDialog.__init__
    dlg = SnapshotDialog.__new__(SnapshotDialog)
    dlg.snapshot_model = _DummySnapshotsModel()

    monkeypatch.setattr(gc, 'collect', lambda: None)

    e = _DummyEvent()
    SnapshotDialog.closeEvent(dlg, e)

    assert e.accepted is True
    assert dlg.snapshot_model is None
