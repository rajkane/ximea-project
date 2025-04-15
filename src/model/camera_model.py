import gc
import time
from src.model.constants import Const
from src.external import qtc, qtg
from ximea import xiapi
from sfps import SFPS
import cv2

class CameraModel(qtc.QThread):
    update = qtc.pyqtSignal(qtg.QImage, name="camera-signal")
    status = qtc.pyqtSignal(str, name="status-signal")
    exception = qtc.pyqtSignal(str, name="exception-signal")
    check_conn = qtc.pyqtSignal(bool, name="check-conn-signal")

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

    def set_is_model(self, is_model: bool):
        self.__is_model = is_model

    def set_auto_gain_exposure(self, auto: bool):
        self.__auto = auto

    def set_model(self, model: str):
        self.__model = model

    def set_annot(self, annot: list):
        self.__annot = annot

    def set_gain_camera(self, gain: float):
        self.__gain = gain

    def set_exposure_camera(self, exposure: int):
        self.__exposure = exposure

    def set_scale_camera(self, scale: float):
        self.__size = qtc.QSize(int(Const.WIDTH * scale), int(Const.HEIGHT * scale))

    @staticmethod
    def detector_objects(model, frame):
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


    def eag_auto_maunal(self):
        if self.__auto is False:
            self.cam.enable_aeag()
        else:
            self.cam.set_gain(self.__gain)
            self.cam.set_exposure(self.__exposure)

    def __config_camera(self):
        if not isinstance(self.mutex, qtc.QMutex):
            self.mutex = qtc.QMutex()

        if not isinstance(self.__sfps, SFPS):
            self.__sfps = SFPS(nframes=5, interval=1)

        if not isinstance(self.cam, xiapi.Camera):
            self.cam = xiapi.Camera()

        self.status.emit("Communication ...")
        self.cam.open_device_by_SN("UBFAS2438006")
        self.status.emit("Device Opened")
        self.cam.set_proc_num_threads(8)
        self.cam.enable_auto_bandwidth_calculation()
        self.cam.set_debug_level("XI_DL_FATAL")
        # self.cam.set_acq_buffer_size(1500000000)

        self.cam.enable_auto_wb()
        self.cam.set_buffer_policy("XI_BP_UNSAFE")
        self.cam.set_imgdataformat("XI_RGB24")
        # self.cam.set_transport_data_target("XI_TRANSPORT_DATA_TARGET_UNIFIED")

        self.eag_auto_maunal()

        self.status.emit("Creating Instance of Image to Store Image Data and Metadata ...")

        if not isinstance(self.img, xiapi.Image):
            self.img = xiapi.Image()
        self.status.emit("Created Instance")
        self.status.emit('Starting Data Acquisition...')
        self.cam.start_acquisition()
        self.status.emit('Started Data Acquisition')

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

    def __close_cam(self):
        self.mutex.lock()
        self.status.emit("Stopping Acquisition ...")
        self.cam.set_counter_selector("XI_CNT_SEL_TRANSPORT_SKIPPED_FRAMES")
        self.cam.get_counter_value()
        self.cam.set_counter_selector("XI_CNT_SEL_API_SKIPPED_FRAMES")
        self.cam.get_counter_value()
        self.cam.stop_acquisition()
        self.status.emit("Stopped Acquisition")
        self.status.emit("Closing Camera ...")
        self.cam.close_device()
        self.status.emit("Camera Closed")
        self.check_conn.emit(False)
        self.mutex.unlock()

    def detector(self):
        if self.__is_model and self.__model is not None and self.__annot is not None:
            import torch
            from detecto import core
            # Ensure GPU is available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda":
                torch.cuda.empty_cache()
            self.model = core.Model.load(self.__model, self.__annot)

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

    def run(self):
        try:
            self.thread = True
            self.__config_camera()

            while self.thread:
                self.mutex.lock()
                if self.cam.is_isexist():
                    self.check_conn.emit(True)

                    self.detector()

                    self.cam.get_image(self.img)
                    img = self.img.get_image_data_numpy(True)
                    img = cv2.resize(img, (800, 600))

                    self.show_fps(img)

                    self.detection(img=img)

                    self.eag_auto_maunal()

                    self.convert_frame(img=img)

                    self.status.emit("Camera Streaming ...")
                    time.sleep(self.__exposure / Const.WAIT_EXPOSURE)
                else:
                    break
                self.mutex.unlock()

            self.__close_cam()

        except xiapi.Xi_error as e:
            self.exception.emit(str(e))

        finally:
            self.stop()
            del self.__sfps
            del self.cam
            del self.img
            del self.model
            gc.collect()
            self.sfps, self.cam, self.img, self.model = None, None, None, None

    def stop(self):
        if self.isRunning():
            self.check_conn.emit(False)
            self.thread = False
            self.quit()
            self.wait()
