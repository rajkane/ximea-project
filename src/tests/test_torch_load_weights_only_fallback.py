import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import types

import pytest


def test_torch_load_fallback_does_not_mask_invalid_pth(monkeypatch, tmp_path):
    """If a .pth is not a valid state_dict, fallback should raise a clear error."""
    from src.model import camera_model as cm

    # Create invalid pth file
    pth = tmp_path / 'bad.pth'
    pth.write_bytes(b'\x08')

    # Patch detecto.core.Model.load to raise weights_only error
    class FakeDetecoModel:
        def __init__(self, classes):
            self._device = 'cpu'
            self._model = types.SimpleNamespace(load_state_dict=lambda x: None)

        @staticmethod
        def load(_path, _classes):
            raise RuntimeError('Weights only load failed: weights_only')

    # Patch torch.load to return non-dict to simulate invalid file
    import torch

    def fake_torch_load(*args, **kwargs):
        return 123

    monkeypatch.setattr(torch, 'load', fake_torch_load)

    fake_core = types.ModuleType('detecto.core')
    fake_core.Model = FakeDetecoModel
    monkeypatch.setitem(sys.modules, 'detecto.core', fake_core)

    cam = cm.CameraModel()
    cam.set_is_model(True)
    cam.set_model(str(pth))
    cam.set_annot(['Cap OK'])

    with pytest.raises(RuntimeError):
        cam._load_torch_detector('torch-cpu')
