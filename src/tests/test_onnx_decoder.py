import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np

from src.model.onnx_inference import OnnxDetector


def test_decode_outputs_boxes_labels_scores():
    boxes = np.zeros((2, 4), dtype=np.float32)
    labels = np.array([1, 2], dtype=np.int64)
    scores = np.array([0.9, 0.7], dtype=np.float32)

    out_labels, out_boxes, out_scores = OnnxDetector._decode_outputs([boxes, labels, scores])

    assert out_boxes.shape == (2, 4)
    assert out_scores.shape == (2,)
    assert out_labels == ['1', '2']


def test_decode_outputs_boxes_scores_labels_order_swapped():
    boxes = np.zeros((1, 4), dtype=np.float32)
    scores = np.array([0.5], dtype=np.float32)
    labels = np.array([3], dtype=np.int64)

    out_labels, out_boxes, out_scores = OnnxDetector._decode_outputs([boxes, scores, labels])
    assert out_labels == ['3']
    assert out_boxes.shape == (1, 4)
    assert out_scores.shape == (1,)


def test_decode_outputs_unknown_schema_returns_empty():
    out_labels, out_boxes, out_scores = OnnxDetector._decode_outputs([np.zeros((5, 5), dtype=np.float32)])
    assert out_labels == []
    assert out_boxes is None or getattr(out_boxes, 'shape', None) != (5, 4)
