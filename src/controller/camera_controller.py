import gc
import os
from src.model.constants import Const
from fractions import Fraction
from src.external import qtw, qtc, qtg
from src.view.mainwindow import Ui_MainWindow
from src.model.camera_model import CameraModel
from src.controller.snapshot_controller import SnapshotDialog
from src.controller.learning_controller import LearningWindow
from src.model.constants import Exposure


class MainWindow(qtw.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.path_model = None
        self.annotation = None
        self.cam_process = None
        self.snapshots_window = None
        self.learning_window = None
        pixmap = qtg.QPixmap(os.path.join("./assets/video-camera-alt.png"))
        self.lbl_camera_stream.setPixmap(pixmap)
        self.scale = 1
        self.__config_buttons()
        self.__config_other_objects()
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
        self.btn_learning.clicked.connect(self.__open_learning_window)
        self.tbtn_model.setEnabled(False)
        self.tbtn_model.clicked.connect(self.__open_model)
        self.tbtn_annotation.setEnabled(False)
        self.tbtn_annotation.clicked.connect(self.__open_annotation)
        self.chb_model.checkStateChanged.connect(self.__enable_model)
        self.chb_manual.checkStateChanged.connect(self.__auto_gain_exposure)

    def __config_other_objects(self):
        self.dial_gain.setEnabled(False)
        self.lbl_gain_value.setText(f"{self.dial_gain.value()}dB")
        self.dial_gain.valueChanged.connect(self.__change_gain_camera)

        self.dial_exposure.setEnabled(False)
        self.lbl_exposure_value.setText("1/4000s")
        self.dial_exposure.valueChanged.connect(self.__change_exposure_camera)

    def __auto_gain_exposure(self):
        if self.chb_manual.isChecked():
            self.dial_gain.setEnabled(True)
            self.dial_exposure.setEnabled(True)
            if isinstance(self.cam_process, CameraModel):
                if self.cam_process.isRunning():
                    self.cam_process.set_auto_gain_exposure(True)
        else:
            self.dial_gain.setEnabled(False)
            self.dial_exposure.setEnabled(False)
            if isinstance(self.cam_process, CameraModel):
                if self.cam_process.isRunning():
                    self.cam_process.set_auto_gain_exposure(False)

    def __enable_model(self):
        if self.chb_model.isChecked():
            self.tbtn_model.setEnabled(True)
            self.tbtn_annotation.setEnabled(True)
            if isinstance(self.cam_process, CameraModel):
                if self.cam_process.isRunning():
                    self.cam_process.set_is_model(True)
        else:
            self.tbtn_model.setEnabled(False)
            self.tbtn_annotation.setEnabled(False)
            if isinstance(self.cam_process, CameraModel):
                if self.cam_process.isRunning():
                    self.cam_process.set_is_model(False)

    def __open_model(self):
        path_model, _ = qtw.QFileDialog.getOpenFileName(
            self,
            "Open Model",
            "",
            filter="Model (*.pth)",
            initialFilter="Model (*.pth)")
        if path_model:
            self.path_model = path_model
        self.le_model.setText(os.path.basename(path_model))

    def __open_annotation(self):
        path_annotation, _ = qtw.QFileDialog.getOpenFileName(
            self,
            "Open Annotation",
            "",
            filter="Annotation (*.txt)",
            initialFilter="Annotation (*.txt)")
        if path_annotation:
            with open(path_annotation, 'r') as f:
                annotation = [name.rstrip() for name in f]
            f.close()
            self.annotation = annotation
            self.le_annotation.setText(os.path.basename(path_annotation))

    def __start_camera(self):
        #if not isinstance(self.cam_process, CameraModel):
        self.cam_process = CameraModel()
        if self.chb_model.isChecked():
            if self.path_model is not None and self.annotation is not None:
                self.cam_process.set_is_model(True)
                self.cam_process.set_model(model=self.path_model)
                self.cam_process.set_annot(annot=self.annotation)
        else:
            self.cam_process.set_is_model(False)
        self.cam_process.set_auto_gain_exposure(False)
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
        self.chb_manual.setChecked(False)
        self.dial_gain.setEnabled(False)
        self.dial_exposure.setEnabled(False)

    def __open_snapshots_window(self):
        self.__stop_camera()
        if not isinstance(self.snapshots_window, SnapshotDialog):
            self.snapshots_window = SnapshotDialog()
        self.snapshots_window.setWindowModality(qtc.Qt.WindowModality.ApplicationModal)
        self.snapshots_window.show()

    def __open_learning_window(self):
        if not isinstance(self.learning_window, LearningWindow):
            self.learning_window = LearningWindow()
        self.learning_window.setWindowModality(qtc.Qt.WindowModality.ApplicationModal)
        self.learning_window.show()

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
            # self.dial_gain.setEnabled(True)
            # self.dial_exposure.setEnabled(True)
            self.tbtn_model.setEnabled(False)
            self.tbtn_annotation.setEnabled(False)
            self.le_model.setEnabled(False)
            self.le_annotation.setEnabled(False)
            self.btn_fill_dataset.setEnabled(False)
            self.btn_learning.setEnabled(False)
        else:
            self.btn_stop.setEnabled(False)
            self.btn_start.setEnabled(True)
            # self.dial_gain.setEnabled(False)
            # self.dial_exposure.setEnabled(False)
            if self.chb_model.isChecked():
                self.tbtn_model.setEnabled(True)
                self.tbtn_annotation.setEnabled(True)
            self.le_model.setEnabled(True)
            self.le_annotation.setEnabled(True)
            self.btn_fill_dataset.setEnabled(True)
            self.btn_learning.setEnabled(True)

    @qtc.pyqtSlot(qtg.QImage)
    def __update_image(self, img):
        self.lbl_camera_stream.setPixmap(qtg.QPixmap.fromImage(img))

    @qtc.pyqtSlot(str)
    def __get_status(self, status):
        self.statusBar().setStyleSheet(
            """
                background-color: green;
                color: white;
            """
        )
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
            if isinstance(self.learning_window, LearningWindow):
                self.learning_window.close()
            if isinstance(self.cam_process, CameraModel):
                if self.cam_process.isRunning():
                    self.cam_process.stop()

            e.accept()
        else:
            e.ignore()