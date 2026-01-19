# small smoke test for convert_to_int_labels
import sys
import os
# Ensure project root is on sys.path so `import src...` works
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.learning_process import WorkerRCNN
import torch
import traceback

print('Starting test_convert_labels')
try:
    # Create a worker with one annotation 'Cap OK' (so mapping has '__background__' and 'Cap OK')
    worker = WorkerRCNN(dataset_name='.', batch_size=1, annotation=['Cap OK'], epochs=1, lr_step_size=1, learning_rate=0.001, model_name='test')

    # Create fake targets where labels contain an unknown label 'Cap NOK'
    targets = [
        {'labels': ['Cap OK', 'Cap NOK'], 'boxes': torch.tensor([[0,0,10,10],[5,5,15,15]])}
    ]

    worker.convert_to_int_labels(targets)
    print('Converted labels:', targets[0]['labels'])
except Exception as e:
    print('Exception during test:')
    traceback.print_exc()
