import datetime
import os

from src.external import qtc, qtg
from src.model.constants import Const
try:
    from ximea import xiapi
    _HAS_XIMEA = True
except Exception:
    xiapi = None
    _HAS_XIMEA = False
import time
import cv2

class SnapshotsModel(qtc.QThread):
    update = qtc.pyqtSignal(qtg.QImage)
    status = qtc.pyqtSignal(str)
    progress = qtc.pyqtSignal(int)
    exception = qtc.pyqtSignal(str)
    check_conn = qtc.pyqtSignal(bool)

    def __init__(self):
        super(SnapshotsModel, self).__init__()
        self.mutex = None
        self.thread = False
        self.cam = None
        self.cam_cv = None  # OpenCV fallback handle
        self.cam_type = None  # 'ximea' or 'opencv'
        self.img = None
        self.size = Const.SIZE
        self.path = None
        self.gain = None
        self.exposure = None
        self.number = None
        self.interval = None

    def set_scale_camera(self, scale: float):
        self.size = qtc.QSize(int(Const.WIDTH * scale), int(Const.HEIGHT * scale))

    def set_path(self, path):
        self.path = path

    def get_path(self):
        return self.path

    def set_gain_camera(self, gain: float):
        self.gain = gain
        # If OpenCV backend is active, try to apply immediately (best-effort)
        try:
            if self.cam_type == 'opencv' and self.cam_cv is not None and self.cam_cv.isOpened():
                self._apply_opencv_gain_value()
        except Exception:
            pass

    def get_gain_camera(self):
         return self.gain

    def set_exposure_camera(self, exposure: int):
        self.exposure = exposure
        # If OpenCV backend is active, try to apply immediately (best-effort)
        try:
            if self.cam_type == 'opencv' and self.cam_cv is not None and self.cam_cv.isOpened():
                self._apply_opencv_exposure_value()
        except Exception:
            pass

    def get_exposure_camera(self):
         return self.exposure

    def set_number(self, number: int):
        self.number = number

    def set_interval_camera(self, interval: int):
        self.interval = interval

    def get_interval_camera(self):
        return self.interval

    # --- refactored helpers start ---
    def _config_ximea(self):
        """Configure ximea camera; may raise on failure."""
        if not isinstance(self.cam, xiapi.Camera):
            self.cam = xiapi.Camera()
        # Attempt to open configured device; original code used fixed serial
        try:
            self.cam.open_device_by_SN("UBFAS2438006")
        except Exception:
            self.cam.open_device()

        self._init_ximea_basic_params()

        # attempt to apply requested exposure/gain before starting acquisition
        try:
            self._apply_ximea_exposure_gain()
        except Exception:
            pass

        # respect requested gain/exposure if set, otherwise use Const defaults
        self._ensure_ximea_requested_values()

        if not isinstance(self.img, xiapi.Image):
            self.img = xiapi.Image()
        self.cam.start_acquisition()
        # try to re-apply/verify after acquisition started
        try:
            self._apply_ximea_exposure_gain()
        except Exception:
            pass
        self.cam_type = 'ximea'

    def _init_ximea_basic_params(self):
        """Initialize basic Ximea parameters (small helper to reduce complexity)."""
        self.cam.enable_auto_bandwidth_calculation()
        self.cam.set_debug_level("XI_DL_FATAL")
        self.cam.set_proc_num_threads(4)
        self.cam.enable_auto_wb()
        self.cam.set_buffer_policy("XI_BP_UNSAFE")
        self.cam.set_imgdataformat("XI_RGB24")
        # prefer manual AEAG for snapshot capture so set_exposure/set_gain apply
        try:
            if hasattr(self.cam, 'disable_aeag'):
                self.cam.disable_aeag()
        except Exception:
            pass

    def _ensure_ximea_requested_values(self):
        """Set requested gain/exposure to camera (keeps the code small in _config_ximea)."""
        self._ensure_ximea_gain_value()
        self._ensure_ximea_exposure_value()

    def _ensure_ximea_gain_value(self):
        # Single-attempt setter: try to set what was requested (or default) and ignore failures
        val = self.gain if self.gain is not None else Const.GAIN
        try:
            self.cam.set_gain(val)
        except Exception:
            try:
                # fallback to numeric coercion
                self.cam.set_gain(float(val))
            except Exception:
                pass

    def _ensure_ximea_exposure_value(self):
        # Single-attempt setter: prefer integer microseconds, fall back to default
        val = int(self.exposure) if self.exposure is not None else Const.EXPOSURE
        try:
            self.cam.set_exposure(val)
        except Exception:
            try:
                self.cam.set_exposure(float(val))
            except Exception:
                pass

    def _apply_ximea_exposure(self):
        """Apply exposure and verify read-back (delegates to helper)."""
        if self.exposure is None:
            return
        emin, emax = self._get_ximea_exposure_limits()
        exp = min(max(int(self.exposure), emin), emax)
        self._set_and_verify_exposure(exp)

    def _set_and_verify_exposure(self, exp: int):
        """Perform the exposure set/read/verify cycle by calling small helpers."""
        return self._perform_exposure_cycle(exp)

    def _perform_exposure_cycle(self, exp: int):
        """Perform the exposure set/read/verify cycle by calling small helpers."""
        self._try_set_exposure(exp)
        read_exp = self._ensure_exposure_readback(exp)
        self._emit_exposure_status(exp, read_exp)

    def _ensure_exposure_readback(self, exp: int):
        """Read exposure and try direct update if necessary; return read value or None."""
        read_exp = self._read_ximea_exposure()
        if self._exposure_needs_direct_update(exp, read_exp):
            self._try_ximea_direct_exposure(exp)
            read_exp = self._read_ximea_exposure()
        return read_exp

    def _emit_exposure_status(self, exp: int, read_exp: int | None):
        try:
            self.status.emit(f"Ximea exposure set/read: {exp}/{read_exp}")
        except Exception:
            pass

    def _try_set_exposure(self, exp: int):
        # Try setting exposure and wait briefly for the camera to apply it, with a few retries.
        for _ in range(5):
            try:
                self.cam.set_exposure(exp)
            except Exception:
                try:
                    self.cam.set_exposure(float(exp))
                except Exception:
                    pass
            # small delay to give camera time to apply long exposures
            time.sleep(0.05)
            try:
                read = int(self.cam.get_exposure())
                if abs(read - exp) <= max(1, int(0.01 * exp)):
                    return
            except Exception:
                # if read not available yet, continue retrying
                pass

    def _read_ximea_exposure(self):
        try:
            return int(self.cam.get_exposure())
        except Exception:
            return None

    def _try_ximea_direct_exposure(self, exp: int):
        # Try direct exposure update which some drivers require, with retries like _try_set_exposure
        if not hasattr(self.cam, 'set_exposure_direct'):
            return
        for _ in range(3):
            try:
                self.cam.set_exposure_direct(exp)
            except Exception:
                pass
            time.sleep(0.05)
            try:
                read = int(self.cam.get_exposure())
                if abs(read - exp) <= max(1, int(0.01 * exp)):
                    return
            except Exception:
                pass

    def _apply_ximea_gain(self):
        """Apply gain and verify read-back via small helpers."""
        if self.gain is None:
            return
        gmin, gmax = self._get_ximea_gain_limits()
        gain = float(self.gain)
        gain = min(max(gain, gmin), gmax)
        self._try_set_gain_with_retries(gain)

    def _get_ximea_gain_limits(self):
        try:
            gmin = float(self.cam.get_gain_minimum())
            gmax = float(self.cam.get_gain_maximum())
        except Exception:
            gmin, gmax = -100.0, 100.0
        return gmin, gmax

    def _try_set_gain_with_retries(self, gain: float):
        """Try to set gain multiple times and emit status when verified (delegates to helpers)."""
        for _ in range(4):
            self._attempt_set_gain(gain)
            time.sleep(0.02)
            read_gain = self._read_gain()
            if read_gain is not None and abs(read_gain - gain) <= max(0.1, 0.01 * abs(gain)):
                self._emit_gain_status(gain, read_gain)
                return
        # final attempt to read and emit
        read_gain = self._read_gain()
        self._emit_gain_status(gain, read_gain)

    def _attempt_set_gain(self, gain: float):
        try:
            self.cam.set_gain(gain)
        except Exception:
            try:
                self.cam.set_gain(float(gain))
            except Exception:
                pass

    def _read_gain(self):
        try:
            return float(self.cam.get_gain())
        except Exception:
            return None

    def _set_max_resolution_opencv(self):
        """Try to set a high resolution on OpenCV capture; use first supported candidate."""
        candidates = [
            (3840, 2160),
            (2560, 1440),
            (1920, 1080),
            (1280, 720),
            (1024, 768),
            (800, 600)
        ]
        for w, h in candidates:
            if self._try_set_opencv_resolution(w, h):
                return
        # fallback: read current
        try:
            rw = int(self.cam_cv.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            rh = int(self.cam_cv.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self.status.emit(f"OpenCV resolution: {rw}x{rh}")
        except Exception:
            pass

    def _try_set_opencv_resolution(self, w: int, h: int) -> bool:
        try:
            self.cam_cv.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cam_cv.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            rw = int(self.cam_cv.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            rh = int(self.cam_cv.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if rw >= w and rh >= h:
                self.status.emit(f"OpenCV resolution set to {rw}x{rh}")
                return True
        except Exception:
            return False
        return False

    def _apply_opencv_exposure_gain(self):
        """Best-effort attempts to set exposure and gain on the OpenCV capture and verify by reading back."""
        try:
            self._disable_opencv_auto_exposure()
            self._try_opencv_exposure_scales()
            self._set_opencv_gain()
        except Exception:
            # keep best-effort without raising
            pass

    def _disable_opencv_auto_exposure(self):
        """Try several ways to disable auto-exposure on various backends.

        Different OpenCV backends/drivers expect different values for
        CAP_PROP_AUTO_EXPOSURE (v4l2 uses 1/3 or 0.25/1.0 variants). We try a
        list of common values and leave the one that appears supported.
        """
        tried = []
        candidates = [0.25, 0.75, 1, 3, 0]
        for v in candidates:
            try:
                self.cam_cv.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(v))
                # read back what the driver reports
                try:
                    rv = self.cam_cv.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                except Exception:
                    rv = None
                tried.append((v, rv))
                # if readback is close to written value, assume success
                if rv is not None and abs((rv or 0) - v) < 1.0:
                    self.status.emit(f"OpenCV auto_exposure set to {v} (read {rv})")
                    return
            except Exception:
                continue
        # fallback: emit what we tried
        if tried:
            self.status.emit(f"OpenCV auto_exposure tried: {tried}")

    def _try_opencv_exposure_scales(self):
        # delegate to small helpers to keep complexity per function low
        if self.exposure is None:
            return
        candidates = self._generate_opencv_exposure_candidates(self.exposure)
        tried = []
        for val in candidates:
            ok, read = self._try_exposure_candidate(val)
            tried.append((val, read))
            if ok:
                self.status.emit(f"OpenCV exposure set (requested={val}, read={read})")
                return
        if tried:
            self.status.emit(f"OpenCV exposure tried: {tried}")

    def _generate_opencv_exposure_candidates(self, exposure_us: float):
        exp_us = float(exposure_us)
        exp_s = exp_us / 1_000_000.0
        exp_ms = exp_s * 1000.0
        candidates = [exp_ms, exp_s, exp_us, -exp_ms]
        if exp_s > 0:
            candidates.append(max(0.0, 1.0/exp_s))
        # deduplicate while preserving order
        out = []
        seen = set()
        for c in candidates:
            if c in seen or c is None:
                continue
            seen.add(c)
            out.append(c)
        return out

    def _try_exposure_candidate(self, val: float):
        try:
            self.cam_cv.set(cv2.CAP_PROP_EXPOSURE, float(val))
            read = self.cam_cv.get(cv2.CAP_PROP_EXPOSURE)
            if read is not None and abs((read or 0) - val) / max(1.0, abs(val)) < 0.5:
                return True, read
            return False, read
        except Exception:
            return False, None

    def _apply_opencv_exposure_value(self):
        """Apply current exposure value immediately to OpenCV camera (best-effort)."""
        if self.exposure is None or self.cam_cv is None:
            return
        candidates = self._generate_opencv_exposure_candidates(self.exposure)
        for val in candidates:
            ok, read = self._try_exposure_candidate(val)
            if ok:
                self.status.emit(f"OpenCV exposure set (requested={val}, read={read})")
                return
        tried = [(v, self.cam_cv.get(cv2.CAP_PROP_EXPOSURE) if self.cam_cv is not None else None) for v in candidates]
        self.status.emit(f"OpenCV exposure tried (live): {tried}")

    def _apply_opencv_gain_value(self):
        """Apply current gain value immediately to OpenCV camera (best-effort)."""
        if self.gain is None or self.cam_cv is None:
            return
        try:
            self.cam_cv.set(cv2.CAP_PROP_GAIN, float(self.gain))
            read_gain = self.cam_cv.get(cv2.CAP_PROP_GAIN)
            self.status.emit(f"OpenCV gain set/read: {self.gain}/{read_gain}")
        except Exception:
            self.status.emit(f"OpenCV gain set failed for: {self.gain}")

    def _config_opencv(self):
        """Configure OpenCV camera; may raise on failure."""
        self.cam_cv = cv2.VideoCapture(0, cv2.CAP_ANY)
        if not self.cam_cv.isOpened():
            raise RuntimeError("Could not open OpenCV camera at index 0")

        # Try to set max resolution supported
        try:
            self._set_max_resolution_opencv()
        except Exception:
            pass

        # Apply exposure/gain best-effort
        try:
            self._apply_opencv_exposure_gain()
        except Exception:
            pass

        self.cam_type = 'opencv'
        # signal that a camera is available
        self.check_conn.emit(True)

    def _close_ximea(self):
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
        try:
            self.cam.close_device()
        except Exception:
            pass
        self.check_conn.emit(False)

    def _close_opencv(self):
        try:
            self.cam_cv.release()
        except Exception:
            pass
        self.check_conn.emit(False)

    def _capture_ximea(self):
        # ensure gain/exposure are integers where appropriate
        if self.gain is not None:
            try:
                self.cam.set_gain(int(self.gain))
            except Exception:
                pass
        if self.exposure is not None:
            try:
                self.cam.set_exposure(int(self.exposure))
            except Exception:
                pass
        self.cam.get_image(self.img)
        img = self.img.get_image_data_numpy(True)
        # ximea returns RGB24 ordering already; return native resolution
        return img

    def _capture_opencv(self):
        if self.interval:
            time.sleep(self.interval)
        ret, frame = self.cam_cv.read()
        if not ret or frame is None:
            raise RuntimeError("OpenCV capture failed")
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return img

    def _camera_available(self):
        if self.cam_type == 'ximea' and isinstance(self.cam, xiapi.Camera):
            try:
                return bool(self.cam.is_isexist())
            except Exception:
                return True
        if self.cam_type == 'opencv' and self.cam_cv is not None:
            try:
                return bool(self.cam_cv.isOpened())
            except Exception:
                return False
        return False
    # --- refactored helpers end ---

    def __config_camera(self):
        """Try to configure Ximea camera, otherwise fall back to OpenCV VideoCapture(0).

        This keeps behaviour for Ximea users while allowing dataset capture on machines
        without the hardware.
        """
        # ensure mutex
        if not isinstance(self.mutex, qtc.QMutex):
            self.mutex = qtc.QMutex()

        # Try Ximea first (if available)
        if _HAS_XIMEA:
            try:
                self._config_ximea()
                return
            except Exception:
                # Failed to configure Ximea; fall back to OpenCV
                self.status.emit("Ximea camera not available, falling back to OpenCV")

        # OpenCV fallback
        self._config_opencv()

    def __close_cam(self):
        # delegate to backend-specific closers
        if self.cam_type == 'ximea' and isinstance(self.cam, xiapi.Camera):
            self._close_ximea()
            return
        if self.cam_type == 'opencv' and self.cam_cv is not None:
            self._close_opencv()
            return
        self.check_conn.emit(False)

    def _validate_config(self):
        if not self.path or not isinstance(self.path, str):
            raise ValueError("SnapshotsModel: path is not set")
        if not isinstance(self.number, int) or self.number <= 0:
            raise ValueError("SnapshotsModel: number of images is not set")
        if self.gain is None:
            raise ValueError("SnapshotsModel: gain is not set")
        if self.exposure is None:
            raise ValueError("SnapshotsModel: exposure is not set")

    def _ensure_output_dirs(self):
        os.makedirs(os.path.join(self.path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.path, 'valid'), exist_ok=True)

    def _write_image(self, img, index: int):
        check_count = int(self.number * .8)
        subdir = 'train' if index < check_count else 'valid'
        # make filename safe and deterministic-ish
        ts = datetime.datetime.now().isoformat(timespec='seconds').replace(':', '-')
        out_path = os.path.join(f"{self.path}/{subdir}/{ts}_{index}.jpeg")
        # Save at native resolution; convert RGB->BGR for cv2.imwrite
        try:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception:
            bgr = img
        cv2.imwrite(out_path, bgr)

    def _capture_one(self, index: int):
        # Common capture routine with backend branching
        self.check_conn.emit(True)
        try:
            if self.cam_type == 'ximea':
                img = self._capture_ximea()
            elif self.cam_type == 'opencv':
                img = self._capture_opencv()
            else:
                self.status.emit("No camera available for capture")
                return
        except Exception as e:
            self.status.emit(f"Capture failed: {e}")
            return

        # Prepare preview scaled image
        try:
            # if image is numpy array, create QImage from raw data
            convert = qtg.QImage(
                img.data,
                img.shape[1],
                img.shape[0],
                qtg.QImage.Format.Format_RGB888
            )
            pic = convert.scaled(self.size, Const.KEEP_ASPECT_RATION_BY_EXPANDING, Const.FAST_TRANSFORMATION)
            self.update.emit(pic)
        except Exception:
            pass

        # Persist image at native resolution
        self._write_image(img, index)
        self.status.emit(f"Image: {index}")
        self.progress.emit(index)
        if index == self.number:
            self.status.emit("Done")

    def run(self):
        try:
            self.thread = True
            self._validate_config()
            self._ensure_output_dirs()

            self.__config_camera()
            self.mutex.lock()
            try:
                for i in range(1, self.number + 1):
                    if self.thread and self._camera_available():
                        self._capture_one(i)
                    else:
                        self.status.emit(f"Canceled, Last Image {i}")
                        break
            finally:
                self.mutex.unlock()

            self.__close_cam()
            self.stop()

        except Exception as e:
            # unify exception handling so GUI can display it
            self.stop()
            self.exception.emit(str(e))

    def stop(self):
        # allow stop() to be called safely multiple times
        self.thread = False
        try:
            self.check_conn.emit(False)
        except Exception:
            pass
        if self.isRunning():
            self.quit()
            self.wait()
