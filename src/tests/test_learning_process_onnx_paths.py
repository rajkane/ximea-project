import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.model.learning_process import WorkerRCNN


def test_onnx_paths_naming():
    w = WorkerRCNN(dataset_name='Dataset/cap_dataset', batch_size=1, annotation=['Cap OK'], epochs=1,
                   lr_step_size=1, learning_rate=0.01, model_name='cap_model')
    onnx_path, onnx_int8_path = w._onnx_paths()
    assert onnx_path.endswith(os.path.join('Models', 'cap_model.onnx'))
    assert onnx_int8_path.endswith(os.path.join('Models', 'cap_model.int8.onnx'))
