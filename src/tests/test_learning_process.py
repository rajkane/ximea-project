import sys
import os
import torch
import pytest

# Ensure repository root is importable (so `import src...` works)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Monkeypatch torchvision's heavy model before importing WorkerRCNN
import torchvision

# Provide a lightweight dummy fasterrcnn to avoid heavy downloads during tests
def _dummy_fasterrcnn_resnet50_fpn(*args, **kwargs):
    class ClsScore:
        in_features = 1024
    class BoxPredictor:
        def __init__(self):
            self.cls_score = ClsScore()
    class RoiHeads:
        def __init__(self):
            self.box_predictor = BoxPredictor()
    class DummyModel:
        def __init__(self):
            self.roi_heads = RoiHeads()
        def eval(self):
            return self
        def to(self, device):
            return self
        def parameters(self):
            return []
        def train(self):
            pass
        def __call__(self, images, targets=None):
            if targets is not None:
                return {"loss_cls": torch.tensor(0.0)}
            return []
    return DummyModel()

# Patch the function used in WorkerRCNN.__init__
try:
    torchvision.models.detection.fasterrcnn_resnet50_fpn = _dummy_fasterrcnn_resnet50_fpn
except Exception:
    # If attribute doesn't exist, attach it
    import types
    if not hasattr(torchvision.models, 'detection'):
        torchvision.models.detection = types.SimpleNamespace()
    torchvision.models.detection.fasterrcnn_resnet50_fpn = _dummy_fasterrcnn_resnet50_fpn

from src.model.learning_process import WorkerRCNN


def test_get_augment_returns_compose():
    w = WorkerRCNN(dataset_name='.', batch_size=1, annotation=['Cap OK'], epochs=1, lr_step_size=1, learning_rate=0.001, model_name='test')
    # set resize and some augment params
    w.set_resize(128)
    w.set_random_horizontal_flip(0.5)
    aug = w.get_augment()
    from torchvision import transforms
    assert isinstance(aug, transforms.Compose)


def test_validate_dataset_labels_raises_on_unknown():
    w = WorkerRCNN(dataset_name='.', batch_size=1, annotation=['Cap OK'], epochs=1, lr_step_size=1, learning_rate=0.001, model_name='test')
    # simple iterable mimicking dataloader: yields (images, targets)
    dataloader = [ (None, [{'labels': ['Cap OK', 'Cap NOK']}]) ]
    with pytest.raises(ValueError):
        w.validate_dataset_labels(dataloader)


def test_convert_to_int_labels_behavior():
    w = WorkerRCNN(dataset_name='.', batch_size=1, annotation=['Cap OK'], epochs=1, lr_step_size=1, learning_rate=0.001, model_name='test')
    targets = [ {'labels': ['Cap OK'], 'boxes': torch.tensor([[0,0,10,10]])} ]
    w.convert_to_int_labels(targets)
    assert isinstance(targets[0]['labels'], torch.Tensor)
    assert targets[0]['labels'].dtype == torch.int64
    assert targets[0]['labels'].tolist() == [1]

    # unknown label should raise
    targets2 = [ {'labels': ['Cap NOK'], 'boxes': torch.tensor([[0,0,10,10]])} ]
    with pytest.raises(ValueError):
        w.convert_to_int_labels(targets2)


def test_annotation_string_is_parsed_to_list():
    w = WorkerRCNN(dataset_name='.', batch_size=1, annotation='"Cap OK", "Cap NOK"', epochs=1, lr_step_size=1, learning_rate=0.001, model_name='test')
    assert w.annotation == ['Cap OK', 'Cap NOK']


def test_dataset_getitem_resize_tuple_no_ambiguous_truth(monkeypatch):
    """Regression test: Resize((h,w)) used to make scale_factor an array and crash."""
    import pandas as pd
    from src.model import rcnn_model
    from torchvision import transforms

    # Fake a minimal csv dataframe with one image_id and one box
    df = pd.DataFrame([
        {
            'filename': 'dummy.jpg',
            'width': 200,
            'height': 100,
            'class': 'Cap OK',
            'xmin': 10,
            'ymin': 20,
            'xmax': 30,
            'ymax': 40,
            'image_id': 0,
        }
    ])

    monkeypatch.setattr(rcnn_model, 'xml_to_csv', lambda _: df)
    # read_image should return something PIL/torchvision can resize; use a torch tensor image
    monkeypatch.setattr(rcnn_model, 'read_image', lambda _: torch.zeros((3, 100, 200), dtype=torch.uint8))

    ds = rcnn_model.Dataset(label_data='not_a_file_but_folder', transform=transforms.Compose([
        transforms.Resize((128, 128)),
    ]))

    img, target = ds[0]
    assert 'boxes' in target and target['boxes'].shape[-1] == 4
