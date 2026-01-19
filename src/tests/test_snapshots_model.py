import sys
import os

# Ensure repository root is importable (so `import src...` works)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.model.snapshots_model import SnapshotsModel


def test_snapshots_model_gain_getter_returns_value_not_callable():
    sm = SnapshotsModel()
    sm.set_gain_camera(12.5)
    assert sm.get_gain_camera() == 12.5


def test_snapshots_model_interval_getter_returns_value_no_sleep():
    sm = SnapshotsModel()
    sm.set_interval_camera(2)
    assert sm.get_interval_camera() == 2
