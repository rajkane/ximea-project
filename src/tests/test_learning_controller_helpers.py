import sys
import os

# Ensure repository root is importable (so `import src...` works)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.controller.learning_controller import LearningWindow


def test_normalize_annotation_text_strips_and_handles_none():
    assert LearningWindow.normalize_annotation_text('  Cap OK, Cap NOK  ') == 'Cap OK, Cap NOK'
    assert LearningWindow.normalize_annotation_text('') == ''
    assert LearningWindow.normalize_annotation_text(None) == ''  # type: ignore[arg-type]


def test_dataset_has_train_valid(tmp_path):
    ds = tmp_path / 'dataset'
    ds.mkdir()
    assert LearningWindow.dataset_has_train_valid(str(ds)) is False

    (ds / 'train').mkdir()
    (ds / 'valid').mkdir()
    assert LearningWindow.dataset_has_train_valid(str(ds)) is True
