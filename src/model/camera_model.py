import gc
import time

import cv2
from sfps import SFPS
from ximea import xiapi

from src.external import qtc, qtg
from src.model.constants import Const


class CameraModel(qtc.QThread):
    """Camera streaming thread.

    Primary backend: Ximea (xiapi)
    Fallback backend: OpenCV VideoCapture (USB / laptop webcam)
    """

    update = qtc.pyqtSignal(qtg.QImage)
    status = qtc.pyqtSignal(str)
    exception = qtc.pyqtSignal(str)
    check_conn = qtc.pyqtSignal(bool)

    def __init__(self, model: str | None = None, annot: list | None = None):
        super(CameraModel, self).__init__()

        # threading
        self.mutex: qtc.QMutex | None = None
        self.thread: bool = False

        # model / detection
        self.__is_model: bool | None = None
        self.__model: str | None = model
        self.__annot: list | None = annot
        self.model = None
        self.device: str | None = None
        self._detector_ready: bool = False

        # camera settings
        self.__auto: bool | None = None
        self.__gain: float | None = None
        self.__exposure: int | None = None
        self.__size = Const.SIZE

        # fps
        self.__sfps: SFPS | None = None

        # camera backends
        self.cam = None
        self.img = None
        self._camera_backend: str | None = None  # 'ximea' or 'opencv'
        self._cv_cap = None

        # Performance tuning (env-controlled; safe defaults)
        # - INFERENCE_EVERY_N=1 means run inference on every frame
        # - INFERENCE_RESIZE=0 disables inference resize (uses current frame size)
        self._inference_every_n: int = self._parse_int_env('INFERENCE_EVERY_N', 1)
        self._inference_every_n = max(self._inference_every_n, 1)
        self._inference_resize: int = self._parse_int_env('INFERENCE_RESIZE', 0)
        self._inference_resize = max(self._inference_resize, 0)

        self._frame_idx: int = 0

    # -----------------
    # Public setters (used by GUI + tests)
    # -----------------
    def set_is_model(self, is_model: bool):
        self.__is_model = is_model
        self._detector_ready = False

    def set_model(self, model: str):
        self.__model = model
        self._detector_ready = False

    def set_annot(self, annot: list):
        self.__annot = annot
        self._detector_ready = False

    def set_auto_gain_exposure(self, auto: bool):
        self.__auto = auto

    def set_gain_camera(self, gain: float):
        self.__gain = gain

    def set_exposure_camera(self, exposure: int):
        self.__exposure = exposure

    def set_scale_camera(self, scale: float):
        self.__size = qtc.QSize(int(Const.WIDTH * scale), int(Const.HEIGHT * scale))

    # -----------------
    # Ximea AEAG helpers
    # -----------------
    def eag_auto_maunal(self):
        """Apply auto-exposure/gain or manual settings for Ximea backend."""
        # In the original code, `__auto` False meant automatic AEAG.
        if self.__auto is False:
            try:
                self.cam.enable_aeag()
            except Exception:
                pass
            return

        try:
            if self.__gain is not None:
                self.cam.set_gain(self.__gain)
            if self.__exposure is not None:
                self.cam.set_exposure(self.__exposure)
        except Exception:
            pass

    # -----------------
    # Detector
    # -----------------
    @staticmethod
    def _is_weights_only_error(exc: Exception) -> bool:
        msg = str(exc)
        return ('weights_only' in msg) or ('Weights only load failed' in msg)

    def _onnx_candidates(self) -> list[str]:
        import os

        base = os.path.splitext(self.__model)[0]
        onnx_int8 = base + '.int8.onnx'
        onnx_fp32 = base + '.onnx'

        out: list[str] = []
        if os.path.isfile(onnx_int8):
            out.append(onnx_int8)
        if os.path.isfile(onnx_fp32):
            out.append(onnx_fp32)
        return out

    def _try_load_onnx_path(self, onnx_path: str, threads: int) -> bool:
        try:
            from src.model.onnx_inference import OnnxDetector

            self.device = 'cpu'
            self.model = OnnxDetector(onnx_path, self.__annot, threads=threads)
            self._detector_ready = True
            try:
                self.status.emit(f"ONNX loaded: {onnx_path}")
            except Exception:
                pass
            return True
        except Exception as e:
            try:
                self.exception.emit(f"ONNX load failed for {onnx_path}: {e}")
            except Exception:
                pass
            return False

    def _try_load_onnx_detector(self, resolved: str) -> bool:
        if resolved != 'onnx-cpu':
            return False

        candidates = self._onnx_candidates()
        if not candidates:
            import os
            base = os.path.splitext(self.__model)[0]
            try:
                self.exception.emit(f"ONNX file not found. Expected {base}.int8.onnx or {base}.onnx")
            except Exception:
                pass
            return False

        import os
        threads = int(os.getenv('ONNX_NUM_THREADS', '4'))
        for onnx_path in candidates:
            if self._try_load_onnx_path(onnx_path, threads):
                return True
        return False

    def _load_torch_detector(self, resolved: str):
        import torch
        from detecto import core

        if resolved == 'torch-cpu':
            self.device = 'cpu'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.device == 'cuda':
            torch.cuda.empty_cache()

        try:
            self.model = core.Model.load(self.__model, self.__annot)
            self._detector_ready = True
            return
        except Exception as e:
            if not self._is_weights_only_error(e):
                raise
            try:
                self.status.emit(f"Deteco Model.load failed ({e}); retrying torch.load(weights_only=False) ...")
            except Exception:
                pass

        m = core.Model(self.__annot)
        state = torch.load(self.__model, map_location=m._device, weights_only=False)
        if not isinstance(state, dict):
            raise RuntimeError(f"Expected state_dict dict in {self.__model}, got {type(state)}")
        m._model.load_state_dict(state)
        self.model = m
        self._detector_ready = True

    def detector(self):
        """Load detector model once (idempotent)."""
        if self._detector_ready:
            return
        if not (self.__is_model and self.__model is not None and self.__annot is not None):
            return

        import os
        resolved = os.getenv('INFERENCE_BACKEND_RESOLVED', '').strip().lower()

        # Treat AUTO as a hint: if ONNX files exist, prefer ONNX on CPU-only setups.
        if resolved in {'', 'auto'}:
            if self._onnx_candidates():
                resolved = 'onnx-cpu'
            else:
                resolved = 'torch-cpu'

        # If user requested ONNX, do not silently fall back to torch.
        if resolved == 'onnx-cpu':
            if self._try_load_onnx_detector('onnx-cpu'):
                return
            raise RuntimeError(
                'ONNX backend selected (or preferred by AUTO), but ONNX model could not be loaded. '
                'Check .onnx/.int8.onnx and onnxruntime installation.'
            )

        if self._try_load_onnx_detector(resolved):
            return

        self._load_torch_detector(resolved)

    @staticmethod
    def detector_objects(model, frame):
        """Draw detections on the frame (expects Detecto-like predict output)."""
        labels, boxes, scores = model.predict(frame)
        for i in range(boxes.shape[0]):
            box = boxes[i]
            if scores[i] > .80:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 150), 2)
                cv2.rectangle(frame, (int(box[0]), int(box[1])),
                              (int(box[0]) + 200, int(box[1]) + 35), (0, 0, 0), -1)
                cv2.putText(frame, f"{labels[i]}: {int(scores[i] * 100)}%",
                            (int(box[0]) + 15, int(box[1] + 25)),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 150), 1)

    # -----------------
    # Camera configuration
    # -----------------
    def _ensure_thread_primitives(self):
        if not isinstance(self.mutex, qtc.QMutex):
            self.mutex = qtc.QMutex()
        if not isinstance(self.__sfps, SFPS):
            self.__sfps = SFPS(nframes=5, interval=1)

    @staticmethod
    def _parse_int_env(name: str, default: int) -> int:
        import os

        val = os.getenv(name, str(default)).strip()
        try:
            return int(val)
        except Exception:
            return default

    def _try_setup_ximea(self) -> bool:
        try:
            if not isinstance(self.cam, xiapi.Camera):
                self.cam = xiapi.Camera()

            self.status.emit("Communication ...")
            self.cam.open_device_by_SN("UBFAS2438006")
            self.status.emit("Device Opened")
            self.cam.set_proc_num_threads(8)
            self.cam.enable_auto_bandwidth_calculation()
            self.cam.set_debug_level("XI_DL_FATAL")

            self.cam.set_exp_priority(.5)
            self.eag_auto_maunal()
            self.cam.enable_auto_wb()

            self.cam.set_buffer_policy("XI_BP_UNSAFE")
            self.cam.set_imgdataformat("XI_RGB24")
            self.cam.set_acq_buffer_size(1000000000)

            self.status.emit("Creating Instance of Image to Store Image Data and Metadata ...")
            if not isinstance(self.img, xiapi.Image):
                self.img = xiapi.Image()
            self.status.emit("Created Instance")
            self.status.emit('Starting Data Acquisition...')
            self.cam.start_acquisition()
            self.status.emit('Started Data Acquisition')

            self._camera_backend = 'ximea'
            return True
        except Exception:
            return False

    def _open_opencv_camera(self, index: int):
        cap = cv2.VideoCapture(index)
        try:
            opened = bool(cap.isOpened())
        except Exception:
            opened = False

        if not opened:
            try:
                cap.release()
            except Exception:
                pass
            return None

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        except Exception:
            pass

        return cap

    def _setup_opencv_fallback(self):
        # Default max index increased: some systems expose webcams at higher indices
        preferred = self._parse_int_env('OPENCV_CAMERA_INDEX', 0)
        max_index = self._parse_int_env('OPENCV_CAMERA_MAX_INDEX', 10)
        max_index = max(max_index, preferred)

        tried: list[int] = []
        indices = [preferred] + [i for i in range(0, max_index + 1) if i != preferred]

        for idx in indices:
            tried.append(idx)
            self.status.emit(f"Ximea not available, trying OpenCV camera index {idx}")
            cap = self._open_opencv_camera(idx)
            if cap is not None:
                self._cv_cap = cap
                self._camera_backend = 'opencv'
                self.status.emit(f"OpenCV camera opened at index {idx}")
                return

        hint = (
            "Failed to open any OpenCV camera. "
            f"Tried indices: {tried}. "
            "Hint: set OPENCV_CAMERA_INDEX and OPENCV_CAMERA_MAX_INDEX env vars. "
            "Example: OPENCV_CAMERA_INDEX=0 OPENCV_CAMERA_MAX_INDEX=10"
        )
        self.status.emit(hint)
        raise RuntimeError(hint)

    def __config_camera(self):
        """Configure camera (Ximea -> OpenCV fallback)."""
        self._ensure_thread_primitives()
        if self._try_setup_ximea():
            return
        self._camera_backend = None
        self._setup_opencv_fallback()

    # -----------------
    # Streaming pipeline
    # -----------------
    def show_fps(self, img):
        fps = self.__sfps.fps(format_spec='.1f')
        cv2.rectangle(img, Const.START_POINT_BG_FPS, Const.END_POINT_BG_FPS, Const.RECT_COLOR_BG, Const.THICKNEES_RECTANGLE)
        cv2.putText(
            img,
            text=f"FPS: {fps}",
            org=Const.ORG_TEXT_FPS,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=Const.FONT_SCALE_FPS,
            color=Const.RECT_COLOR_TEXT_FPS,
            thickness=Const.THICKNEES,
            lineType=cv2.LINE_AA
        )

    def _grab_frame(self):
        """Return an RGB numpy frame."""
        if self._camera_backend == 'opencv':
            ok, frame_bgr = self._cv_cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError('OpenCV camera read failed')
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        self.cam.get_image(self.img)
        self.cam.set_exp_priority(.5)
        self.eag_auto_maunal()
        return self.img.get_image_data_numpy(True)

    @staticmethod
    def _has_drawable_detections(labels, boxes, scores) -> bool:
        if not labels or boxes is None or scores is None:
            return False
        try:
            if getattr(boxes, 'shape', None) is None:
                return False
            if len(boxes.shape) != 2 or int(boxes.shape[1]) != 4:
                return False
        except Exception:
            return False
        return True

    def _maybe_downscale_for_inference(self, img):
        """Optionally resize image for inference and return (img_for_infer, scale_x, scale_y)."""
        if self._inference_resize <= 0:
            return img, 1.0, 1.0

        h, w = img.shape[0], img.shape[1]
        target = int(self._inference_resize)
        # Keep aspect ratio: scale so max(H,W) == target
        max_dim = max(h, w)
        if max_dim <= target:
            return img, 1.0, 1.0

        scale = target / float(max_dim)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(img, (new_w, new_h))
        return resized, (w / float(new_w)), (h / float(new_h))

    def _draw_detections_scaled(self, img, labels, boxes, scores, sx: float, sy: float):
        """Draw boxes on original img where boxes were predicted on a resized image."""
        if boxes is None or scores is None or not labels:
            return
        try:
            import numpy as np
            if isinstance(boxes, np.ndarray):
                boxes_scaled = boxes.copy()
                boxes_scaled[:, 0] *= sx
                boxes_scaled[:, 2] *= sx
                boxes_scaled[:, 1] *= sy
                boxes_scaled[:, 3] *= sy
                boxes = boxes_scaled
        except Exception:
            pass

        # mimic detector_objects but using provided arrays
        for i in range(getattr(boxes, 'shape', [0])[0]):
            box = boxes[i]
            if scores[i] > .80:
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 150), 2)
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[0]) + 200, int(box[1]) + 35), (0, 0, 0), -1)
                cv2.putText(img, f"{labels[i]}: {int(scores[i] * 100)}%", (int(box[0]) + 15, int(box[1] + 25)), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 150), 1)

    def detection(self, img):
        if not (self.__is_model and self.__model is not None and self.__annot is not None):
            return

        self._frame_idx += 1
        if (self._frame_idx % self._inference_every_n) != 0:
            return

        img_inf, sx, sy = self._maybe_downscale_for_inference(img)

        try:
            labels, boxes, scores = self.model.predict(img_inf)
        except Exception:
            return

        if not self._has_drawable_detections(labels, boxes, scores):
            return

        if sx != 1.0 or sy != 1.0:
            self._draw_detections_scaled(img, labels, boxes, scores, sx, sy)
            return

        self.detector_objects(model=self.model, frame=img)

    def convert_frame(self, img):
        convert = qtg.QImage(
            img.data,
            img.shape[1],
            img.shape[0],
            qtg.QImage.Format.Format_RGB888
        )
        pic = convert.scaled(self.__size, Const.KEEP_ASPECT_RATION_BY_EXPANDING, Const.FAST_TRANSFORMATION)
        self.update.emit(pic)

    def _sleep_seconds(self) -> float:
        try:
            if self.__exposure is None or not Const.WAIT_EXPOSURE:
                return 0.0
            return float(self.__exposure) / float(Const.WAIT_EXPOSURE)
        except Exception:
            return 0.0

    def _process_one_frame(self):
        self.check_conn.emit(True)

        img = self._grab_frame()
        img = cv2.resize(img, (800, 600))

        self.show_fps(img)
        self.detection(img=img)
        self.convert_frame(img=img)

        self.status.emit("Camera Streaming ...")

        sleep_s = self._sleep_seconds()
        if sleep_s > 0:
            time.sleep(sleep_s)

    # -----------------
    # Thread lifecycle
    # -----------------
    def _with_mutex(self, fn):
        self.mutex.lock()
        try:
            fn()
        finally:
            self.mutex.unlock()

    def _should_continue(self) -> bool:
        if self._camera_backend == 'opencv':
            return True
        try:
            return bool(self.cam.is_isexist())
        except Exception:
            return False

    def _close_opencv(self):
        if self._camera_backend != 'opencv' or self._cv_cap is None:
            return
        try:
            self._cv_cap.release()
        except Exception:
            pass
        self._cv_cap = None

    def _close_ximea(self):
        if self._camera_backend != 'ximea':
            return

        self.status.emit("Stopping Acquisition ...")
        try:
            self.cam.set_counter_selector("XI_CNT_SEL_TRANSPORT_SKIPPED_FRAMES")
            self.cam.get_counter_value()
            self.cam.set_counter_selector("XI_CNT_SEL_API_SKIPPED_FRAMES")
            self.cam.get_counter_value()
        except Exception:
            pass

        try:
            self.cam.stop_acquisition()
        except Exception:
            pass

        self.status.emit("Stopped Acquisition")
        self.status.emit("Closing Camera ...")

        try:
            self.cam.close_device()
        except Exception:
            pass

        self.status.emit("Camera Closed")

    def __close_cam(self):
        if self.mutex is not None:
            self.mutex.lock()
        try:
            self._close_opencv()
            self._close_ximea()
            self.check_conn.emit(False)
        finally:
            if self.mutex is not None:
                self.mutex.unlock()

    def run(self):
        try:
            self.thread = True
            self.__config_camera()
            self.detector()

            while self.thread and self._should_continue():
                self._with_mutex(self._process_one_frame)

            self.__close_cam()

        except Exception as e:
            try:
                self.exception.emit(str(e))
            except Exception:
                pass

        finally:
            self.stop()
            try:
                del self.__sfps
                del self.cam
                del self.img
                del self.model
            except Exception:
                pass
            gc.collect()
            self.sfps, self.cam, self.img, self.model = None, None, None, None

    def stop(self):
        """Stop thread safely (idempotent)."""
        self.thread = False
        try:
            self.check_conn.emit(False)
        except Exception:
            pass

        if self.isRunning():
            self.quit()
            # Avoid waiting on itself when called from within the worker thread
            try:
                if qtc.QThread.currentThread() is not self:
                    self.wait()
            except Exception:
                pass
