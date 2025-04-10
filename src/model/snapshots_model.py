import datetime
import os

from src.external import qtc, qtg
from src.model.constants import Const
from ximea import xiapi
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

    def get_gain_camera(self):
        return self.gain()

    def set_exposure_camera(self, exposure: int):
        self.exposure = exposure

    def get_exposure_camera(self):
        return self.exposure

    def set_number(self, number: int):
        self.number = number

    def set_interval_camera(self, interval: int):
        self.interval = interval

    def get_interval_camera(self):
        time.sleep(self.interval)

    def __config_camera(self):
        if not isinstance(self.mutex, qtc.QMutex):
            self.mutex = qtc.QMutex()
        if not isinstance(self.cam, xiapi.Camera):
            self.cam = xiapi.Camera()
        self.cam.open_device_by_SN("UBFAS2438006")
        self.cam.set_debug_level("XI_DL_FATAL")
        self.cam.get_proc_num_threads_maximum()
        self.cam.enable_auto_wb()
        self.cam.set_gain(Const.GAIN)
        self.cam.set_exposure(Const.EXPOSURE)
        # self.cam.set_buffer_policy("XI_BP_UNSAFE")
        self.cam.set_imgdataformat("XI_RGB32")
        # self.cam.set_transport_data_target("XI_TRANSPORT_DATA_TARGET_UNIFIED")
        self.cam.set_acq_buffer_size(512000000)
        self.cam.enable_auto_bandwidth_calculation()
        if not isinstance(self.img, xiapi.Image):
            self.img = xiapi.Image()
        self.cam.start_acquisition()

    def __close_cam(self):
        self.cam.set_counter_selector("XI_CNT_SEL_TRANSPORT_SKIPPED_FRAMES")
        self.cam.get_counter_value()
        self.cam.set_counter_selector("XI_CNT_SEL_API_SKIPPED_FRAMES")
        self.cam.get_counter_value()
        self.cam.stop_acquisition()
        self.cam.close_device()
        self.check_conn.emit(False)

    def run(self):
        try:
            self.thread = True
            self.__config_camera()
            for i in range(1, self.number+1):
                self.mutex.lock()
                if i > self.number:
                    break
                if self.cam.is_isexist():
                    self.check_conn.emit(True)
                    self.cam.set_gain(self.gain)
                    self.cam.set_exposure(self.exposure)
                    self.get_interval_camera()
                    self.cam.get_image(self.img)
                    img = self.img.get_image_data_numpy(False)

                    convert = qtg.QImage(
                        img.data,
                        img.shape[1],
                        img.shape[0],
                        qtg.QImage.Format.Format_RGB32
                    )

                    pic = convert.scaled(self.size, Const.KEEP_ASPECT_RATION_BY_EXPANDING, Const.FAST_TRANSFORMATION)
                    self.update.emit(pic)
                    cv2.imwrite(os.path.join(f"{self.path}/{datetime.datetime.now()}.jpeg"), img)
                    self.status.emit(f"Image: {i}")
                    self.progress.emit(i)
                self.mutex.unlock()

            self.status.emit("Done")
            self.__close_cam()
            self.stop()

        except xiapi.Xi_error as e:
            self.stop()
            self.exception.emit(str(e))

    def stop(self):
        if self.isRunning():
            self.check_conn.emit(False)
            self.thread = False
            self.quit()
            self.wait()
