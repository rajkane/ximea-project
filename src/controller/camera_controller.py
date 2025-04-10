import gc
import os
from src.model.constants import Const
from fractions import Fraction
from src.external import qtw, qtc, qtg
from src.view.mainwindow import Ui_MainWindow
from src.model.camera_model import CameraModel
from src.controller.snapshot_controller import SnapshotDialog
from src.model.constants import Exposure


class MainWindow(qtw.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.cam_process = None
        self.snapshots_window = None
        pixmap = qtg.QPixmap(os.path.join("./assets/video-camera-alt.png"))
        self.lbl_camera_stream.setPixmap(pixmap)
        self.scale = 1
        self.__config_buttons()
        self.__config_dial_gain()
        self.statusBar().showMessage("Ready")

    def __config_camera_window(self):
        if isinstance(self.cam_process, CameraModel):
            if self.cam_process.isRunning():
                if  qtc.Qt.WindowState.WindowMaximized == self.windowState():
                    self.scale = 1.5
                else:
                    self.scale = 1
                self.cam_process.set_scale_camera(self.scale)

    def __config_buttons(self):
        self.btn_start.setEnabled(True)
        self.btn_start.clicked.connect(self.__start_camera)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.__stop_camera)
        self.btn_fill_dataset.clicked.connect(self.__open_snapshots_window)

    def __config_dial_gain(self):
        self.dial_gain.setEnabled(False)
        self.lbl_gain_value.setText(f"{self.dial_gain.value()}dB")
        self.dial_gain.valueChanged.connect(self.__change_gain_camera)

        self.dial_exposure.setEnabled(False)
        self.lbl_exposure_value.setText("1/4000s")
        self.dial_exposure.valueChanged.connect(self.__change_exposure_camera)

    def __start_camera(self):
        if not isinstance(self.cam_process, CameraModel):
            self.cam_process = CameraModel()
        self.cam_process.update.connect(self.__update_image)
        self.cam_process.status.connect(self.__get_status)
        self.cam_process.check_conn.connect(self.__check_conn_camera)
        self.cam_process.exception.connect(self.__get_exception)
        self.cam_process.start()
        if self.cam_process.isRunning():
            self.__change_gain_camera(self.dial_gain.value())
            self.__change_exposure_camera(self.dial_exposure.value())

    def __stop_camera(self):
        if isinstance(self.cam_process, CameraModel):
            if self.cam_process.isRunning():
                self.cam_process.stop()

                del self.cam_process
                gc.collect()
                self.cam_process = None

    def __open_snapshots_window(self):
        if not isinstance(self.snapshots_window, SnapshotDialog):
            self.snapshots_window = SnapshotDialog()
        self.snapshots_window.setModal(True)
        self.snapshots_window.show()
        self.__stop_camera()

    def __change_gain_camera(self, gain: int):
        self.lbl_gain_value.setText(f"{gain}dB")
        self.cam_process.set_gain_camera(float(gain))

    def __change_exposure_camera(self, exposure: int):
        value = Exposure.data.get(exposure)
        if value < 1:
            frac = Fraction(value).limit_denominator(4000)
            ratio = f"{frac.numerator}/{frac.denominator}"
            self.lbl_exposure_value.setText(f"{ratio}s")
        else:
            self.lbl_exposure_value.setText(f"{value}s")
        self.cam_process.set_exposure_camera(int(value * Const.WAIT_EXPOSURE))

    @qtc.pyqtSlot(bool)
    def __check_conn_camera(self, check):
        if check:
            self.btn_stop.setEnabled(True)
            self.btn_start.setEnabled(False)
            self.dial_gain.setEnabled(True)
            self.dial_exposure.setEnabled(True)
        else:
            self.btn_stop.setEnabled(False)
            self.btn_start.setEnabled(True)
            self.dial_gain.setEnabled(False)
            self.dial_exposure.setEnabled(False)

    @qtc.pyqtSlot(qtg.QImage)
    def __update_image(self, img):
        self.lbl_camera_stream.setPixmap(qtg.QPixmap.fromImage(img))

    @qtc.pyqtSlot(str)
    def __get_status(self, status):
        self.statusBar().showMessage(status)

    @qtc.pyqtSlot(str)
    def __get_exception(self, status):
        self.statusBar().setStyleSheet(
            """
                background-color: darkred;
                color: white;
            """
        )
        self.statusBar().showMessage(status)

    def wheelEvent(self, e):
        if isinstance(self.cam_process, CameraModel):
            if self.cam_process.isRunning():
                if e.angleDelta().y() > 0 and self.scale <= 1.5:
                    self.scale += Const.STEP
                elif e.angleDelta().y() < 0 and self.scale >= 1:
                    self.scale -= Const.STEP

                if 1 <= self.scale <= 1.5 and qtc.Qt.WindowState.WindowMaximized == self.windowState():
                    self.cam_process.set_scale_camera(self.scale)

    def mouseDoubleClickEvent(self, e):
        self.showNormal()
        self.__config_camera_window()

    def resizeEvent(self, e):
        self.__config_camera_window()

    def keyPressEvent(self, e):
        if e.key() == qtc.Qt.Key.Key_Escape:
            self.showNormal()
            self.__config_camera_window()

    def closeEvent(self, e):
        msg = qtw.QMessageBox.question(
            self,
            "QUIT",
            "Are you sure to quit?",
        )
        if msg == qtw.QMessageBox.StandardButton.Yes:
            if isinstance(self.snapshots_window, SnapshotDialog):
                self.snapshots_window.close()
            if isinstance(self.cam_process, CameraModel):
                if self.cam_process.isRunning():
                    self.cam_process.stop()
                    e.ignore()

            e.accept()
        else:
            e.ignore()