import gc
import time
from src.model.constants import Const
from src.external import qtc, qtg
from ximea import xiapi
from sfps import SFPS
import cv2

class CameraModel(qtc.QThread):
    """gain = qtc.pyqtSignal(int)
    exposure = qtc.pyqtSignal(int)"""
    update = qtc.pyqtSignal(qtg.QImage)
    status = qtc.pyqtSignal(str)
    exception = qtc.pyqtSignal(str)
    check_conn = qtc.pyqtSignal(bool)

    def __init__(self, model :str = None, annot :list = None):
        super(CameraModel, self).__init__()
        self.mutex = None
        self.thread = False
        self.__is_model = None
        self.__auto = None
        self.__gain = None
        self.__exposure = None
        self.__model = model
        self.__annot = annot
        self.__size = Const.SIZE
        self.__sfps = None
        self.model =  None
        self.cam = None
        self.img = None
        self.device = None
        self._detector_ready = False
        # camera backend: 'ximea' or 'opencv'
        self._camera_backend = None
        self._cv_cap = None

    # -----------------
    # Model / detector
    # -----------------
    def detector(self):
        """Load detector model once (idempotent).

        Tests rely on this method existing and being called exactly once per run.
        """
        if self._detector_ready:
            return
        if not (self.__is_model and self.__model is not None and self.__annot is not None):
            return

        import os
        import torch
        from detecto import core

        resolved = os.getenv("INFERENCE_BACKEND_RESOLVED", "").strip().lower()

        # If user selected ONNX, try it first (CPU only). If anything fails, fall back to Detecto.
        if resolved == "onnx-cpu":
            try:
                onnx_path = os.path.splitext(self.__model)[0] + ".onnx"
                threads = int(os.getenv("ONNX_NUM_THREADS", "4"))
                from src.model.onnx_inference import OnnxDetector

                self.device = "cpu"
                self.model = OnnxDetector(onnx_path, self.__annot, threads=threads)
                self._detector_ready = True
                return
            except Exception:
                pass

        if resolved == "torch-cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            torch.cuda.empty_cache()

        self.model = core.Model.load(self.__model, self.__annot)
        self._detector_ready = True

    # -----------------
    # Camera config helpers (reduce complexity)
    # -----------------
    def _ensure_thread_primitives(self):
        if not isinstance(self.mutex, qtc.QMutex):
            self.mutex = qtc.QMutex()
        if not isinstance(self.__sfps, SFPS):
            self.__sfps = SFPS(nframes=5, interval=1)

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

    @staticmethod
    def _parse_int_env(name: str, default: int) -> int:
        import os

        val = os.getenv(name, str(default)).strip()
        try:
            return int(val)
        except Exception:
            return default

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
        preferred = self._parse_int_env('OPENCV_CAMERA_INDEX', 0)
        max_index = self._parse_int_env('OPENCV_CAMERA_MAX_INDEX', 3)
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
                return

        self.status.emit(f"Failed to open any OpenCV camera. Tried indices: {tried}")
        raise RuntimeError(f"Failed to open any OpenCV camera. Tried indices: {tried}")

    def __config_camera(self):
        """Configure camera.

        Primary: Ximea (xiapi)
        Fallback: OpenCV VideoCapture (USB / laptop webcam)

        You can control OpenCV fallback via env vars:
        - OPENCV_CAMERA_INDEX: preferred index (default 0)
        - OPENCV_CAMERA_MAX_INDEX: max index to try when preferred fails (default 3)
        """
        self._ensure_thread_primitives()

        if self._try_setup_ximea():
            return

        self._camera_backend = None
        self._setup_opencv_fallback()

    # -----------------
    # Run loop helpers
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

    def _grab_frame(self):
        """Return an RGB numpy frame."""
        if self._camera_backend == 'opencv':
            ok, frame_bgr = self._cv_cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError('OpenCV camera read failed')
            # BGR -> RGB
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # default: ximea
        self.cam.get_image(self.img)
        self.cam.set_exp_priority(.5)
        self.eag_auto_maunal()
        return self.img.get_image_data_numpy(True)

    def detection(self, img):
        if self.__is_model and self.__model is not None and self.__annot is not None:
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
        """Compute safe sleep duration for the streaming loop."""
        try:
            if self.__exposure is None or not Const.WAIT_EXPOSURE:
                return 0.0
            return float(self.__exposure) / float(Const.WAIT_EXPOSURE)
        except Exception:
            return 0.0

    def _process_one_frame(self):
        """Grab one frame, apply processing, emit UI updates."""
        self.check_conn.emit(True)

        img = self._grab_frame()

        # unify size
        img = cv2.resize(img, (800, 600))

        self.show_fps(img)
        self.detection(img=img)
        self.convert_frame(img=img)

        self.status.emit("Camera Streaming ...")

        sleep_s = self._sleep_seconds()
        if sleep_s > 0:
            time.sleep(sleep_s)

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
            self.wait()

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

    def set_is_model(self, is_model: bool):
        self.__is_model = is_model
        # reset detector state when toggling
        self._detector_ready = False

    def set_auto_gain_exposure(self, auto: bool):
        self.__auto = auto

    def set_model(self, model: str):
        self.__model = model
        self._detector_ready = False

    def set_annot(self, annot: list):
        self.__annot = annot
        self._detector_ready = False

    def set_gain_camera(self, gain: float):
        self.__gain = gain

    def set_exposure_camera(self, exposure: int):
        self.__exposure = exposure

    def set_scale_camera(self, scale: float):
        self.__size = qtc.QSize(int(Const.WIDTH * scale), int(Const.HEIGHT * scale))

    def eag_auto_maunal(self):
        """Apply auto-exposure/gain or manual settings for Ximea backend."""
        # In the original code, `__auto` False meant automatic AEAG.
        if self.__auto is False:
            try:
                self.cam.enable_aeag()
            except Exception:
                pass
        else:
            try:
                if self.__gain is not None:
                    self.cam.set_gain(self.__gain)
                if self.__exposure is not None:
                    self.cam.set_exposure(self.__exposure)
            except Exception:
                pass
