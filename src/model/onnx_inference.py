from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class OnnxDetectionResult:
    labels: list[str]
    boxes: Any
    scores: Any


class OnnxDetector:
    """CPU ONNX Runtime inference wrapper.

    This is a minimal wrapper designed to match the usage pattern in CameraModel
    (a .predict(frame) method returning (labels, boxes, scores)).

    Notes:
    - The current ONNX export pipeline produces an input of shape (C,H,W) (rank=3)
      in our benchmark. This wrapper supports rank-3 and rank-4 models.
    - Output format depends on the exported graph.

    Quantization:
    - ONNX itself is *not* quantization. This class includes a best-effort probe
      that flags `is_quantized=True` if the ONNX graph contains int8/uint8 weights
      in initializers. (This is a heuristic, but works for common post-training
      quantization flows.)
    """

    def __init__(self, onnx_path: str, classes: list[str], *, threads: int = 4):
        self.onnx_path = onnx_path
        self.classes = classes
        self.threads = max(int(threads), 1)

        # Quantization info (best-effort)
        self.is_quantized: bool = False
        self._dtype_counts: dict[str, int] = {}

        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError(f"onnxruntime is not available: {e}")

        if not os.path.isfile(self.onnx_path):
            raise FileNotFoundError(self.onnx_path)

        self._probe_quantization()

        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.inter_op_num_threads = 1

        self.sess = ort.InferenceSession(self.onnx_path, sess_options=so, providers=['CPUExecutionProvider'])
        self.inp = self.sess.get_inputs()[0]
        self.inp_name = self.inp.name
        try:
            self.expected_rank = len(self.inp.shape)
        except Exception:
            self.expected_rank = None

    def _probe_quantization(self) -> None:
        """Heuristic: check initializer tensor dtypes.

        - int8 / uint8 initializers usually mean the model was quantized.
        - float16/float32 means not quantized (at least not int8-quantized).
        """
        try:
            import onnx
            from onnx import TensorProto
        except Exception:
            # onnx package not available -> leave unknown/False
            return

        try:
            model = onnx.load(self.onnx_path)
        except Exception:
            return

        counts: dict[str, int] = {}
        for init in getattr(model.graph, 'initializer', []):
            # init.data_type is an int enum
            try:
                dtype_name = TensorProto.DataType.Name(int(init.data_type))
            except Exception:
                dtype_name = str(init.data_type)
            counts[dtype_name] = counts.get(dtype_name, 0) + 1

        self._dtype_counts = counts

        # Heuristic: if we see INT8 or UINT8 weights, treat as quantized.
        if counts.get('INT8', 0) > 0 or counts.get('UINT8', 0) > 0:
            self.is_quantized = True

    def model_dtype_summary(self) -> str:
        if not self._dtype_counts:
            return "unknown"
        items = ", ".join(f"{k}={v}" for k, v in sorted(self._dtype_counts.items()))
        return items

    def _to_input(self, frame):
        import numpy as np
        import torch

        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray frame, got {type(frame)}")

        # HWC uint8 -> CHW float32 [0,1]
        chw = torch.from_numpy(frame).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
        arr3 = chw.numpy()

        if self.expected_rank == 3:
            return arr3
        if self.expected_rank == 4:
            return arr3[None, ...]
        return arr3[None, ...]

    @staticmethod
    def _as_list(outputs):
        if outputs is None:
            return []
        if isinstance(outputs, (list, tuple)):
            return list(outputs)
        return [outputs]

    @staticmethod
    def _only_arrays(outputs_list):
        return [o for o in outputs_list if hasattr(o, 'shape')]

    @staticmethod
    def _find_boxes(arrays):
        for a in arrays:
            try:
                if len(a.shape) == 2 and int(a.shape[1]) == 4:
                    return a
            except Exception:
                continue
        return None

    @staticmethod
    def _find_labels(arrays, boxes):
        import numpy as np

        for a in arrays:
            if a is boxes:
                continue
            try:
                if len(a.shape) == 1 and np.issubdtype(a.dtype, np.integer):
                    return a
            except Exception:
                continue
        return None

    @staticmethod
    def _find_scores(arrays, boxes, labels_arr):
        import numpy as np

        for a in arrays:
            if a is boxes or a is labels_arr:
                continue
            try:
                if len(a.shape) == 1 and np.issubdtype(a.dtype, np.floating):
                    return a
            except Exception:
                continue
        return None

    @staticmethod
    def _fallback_1d(arrays, boxes, labels_arr):
        for a in arrays:
            if a is boxes or a is labels_arr:
                continue
            try:
                if len(a.shape) == 1:
                    return a
            except Exception:
                continue
        return None

    @staticmethod
    def _labels_to_strings(labels_arr):
        if labels_arr is None:
            return []
        try:
            return [str(int(x)) for x in labels_arr.tolist()]
        except Exception:
            return []

    @staticmethod
    def _decode_outputs(outputs):
        """Best-effort decode for common torchvision FasterRCNN ONNX exports.

        Returns (labels:list[str], boxes, scores).
        """
        outputs_list = OnnxDetector._as_list(outputs)
        if not outputs_list:
            return [], None, None

        arrays = OnnxDetector._only_arrays(outputs_list)
        boxes = OnnxDetector._find_boxes(arrays)
        labels_arr = OnnxDetector._find_labels(arrays, boxes)
        scores = OnnxDetector._find_scores(arrays, boxes, labels_arr)
        if boxes is not None and scores is None:
            scores = OnnxDetector._fallback_1d(arrays, boxes, labels_arr)

        labels = OnnxDetector._labels_to_strings(labels_arr)
        return labels, boxes, scores

    def predict(self, frame):
        # Run session
        outputs = self.sess.run(None, {self.inp_name: self._to_input(frame)})

        # Decode to Detecto-like output. For now labels are numeric strings unless
        # ONNX provides label ids we can map.
        return self._decode_outputs(outputs)
