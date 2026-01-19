import os
import sys

# Ensure repository root is importable (so `import src...` works)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import pandas as pd
from torchvision import transforms

from src.model import rcnn_model


def test_dataset_apply_transforms_resize_scales_boxes(monkeypatch):
    # One image with known width/height and one box.
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
    monkeypatch.setattr(rcnn_model, 'read_image', lambda _: torch.zeros((3, 100, 200), dtype=torch.uint8))

    ds = rcnn_model.Dataset(label_data='unused', transform=transforms.Compose([
        transforms.Resize((100, 100)),  # min(size)=100 => original min=100 => scale_factor=1
    ]))

    _img, target = ds[0]
    assert target['boxes'].shape == (1, 4)
    assert target['boxes'].dtype == torch.float32
    # scale_factor should be 1 => box unchanged (but converted to long during scaling path only)
    assert target['boxes'][0].tolist() == [10.0, 20.0, 30.0, 40.0]


def test_dataset_horizontal_flip_flips_boxes(monkeypatch):
    df = pd.DataFrame([
        {
            'filename': 'dummy.jpg',
            'width': 100,
            'height': 100,
            'class': 'Cap OK',
            'xmin': 10,
            'ymin': 0,
            'xmax': 20,
            'ymax': 10,
            'image_id': 0,
        }
    ])

    monkeypatch.setattr(rcnn_model, 'xml_to_csv', lambda _: df)
    monkeypatch.setattr(rcnn_model, 'read_image', lambda _: torch.zeros((3, 100, 100), dtype=torch.uint8))

    # Force flip always by monkeypatching random.random
    monkeypatch.setattr(rcnn_model.random, 'random', lambda: 0.0)

    ds = rcnn_model.Dataset(label_data='unused', transform=transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
    ]))

    _img, target = ds[0]
    # width=100, original [10,20] => flipped should be [80,90]
    assert target['boxes'][0].tolist() == [80.0, 0.0, 90.0, 10.0]
