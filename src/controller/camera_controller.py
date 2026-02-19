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
    def _camera_is_running(self) -> bool:
        return isinstance(self.cam_process, CameraModel) and self.cam_process.isRunning()

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
        self.__init_backend_selector()
        self.statusBar().showMessage("Ready")
        self._last_status_detail: str | None = None
        self._last_error_detail: str | None = None
        # enable statusbar double-click to show details
        try:
            self.statusbar.messageChanged.connect(self._status_message_changed)
        except Exception:
            pass

    def _status_message_changed(self, _msg: str):
        # placeholder hook so statusbar exists; kept for future
        return

    def _set_status_with_details(self, message: str, *, is_error: bool):
        max_len = 120
        message = str(message)
        preview = message if len(message) <= max_len else (message[:max_len - 1] + '…')

        if is_error:
            self._last_error_detail = message
        else:
            self._last_status_detail = message

        # Always print full details to console for copy/paste.
        try:
            import sys
            print(message, file=sys.stderr if is_error else sys.stdout)
        except Exception:
            pass

        self.statusBar().showMessage(preview)

    def __init_backend_selector(self):
        """Populate the backend selector.

        This controls env vars INFERENCE_BACKEND / INFERENCE_BACKEND_RESOLVED used by CameraModel.detector().
        """
        cb = getattr(self, 'cb_backend', None)
        if cb is None:
            return

        from src.model.backend import InferenceBackend

        cb.clear()
        cb.addItem("Auto", InferenceBackend.AUTO.value)
        cb.addItem("PyTorch CPU", InferenceBackend.TORCH_CPU.value)
        cb.addItem("ONNX CPU", InferenceBackend.ONNX_CPU.value)
        cb.addItem("PyTorch CUDA", InferenceBackend.TORCH_CUDA.value)

    def __apply_selected_backend(self):
        import os
        cb = getattr(self, 'cb_backend', None)
        if cb is None:
            return

        from src.model.backend import select_backend

        requested = cb.currentData()
        os.environ["INFERENCE_BACKEND"] = str(requested)
        resolved = select_backend(requested)
        os.environ["INFERENCE_BACKEND_RESOLVED"] = resolved.value

        if resolved.value == "torch-cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

        self.statusBar().showMessage(f"Backend: {resolved.value}")

    def __config_camera_window(self):
        if self._camera_is_running():
            if qtc.Qt.WindowState.WindowMaximized == self.windowState():
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
        manual = self.chb_manual.isChecked()
        self.dial_gain.setEnabled(manual)
        self.dial_exposure.setEnabled(manual)
        if self._camera_is_running():
            self.cam_process.set_auto_gain_exposure(manual)

    def __enable_model(self):
        enabled = self.chb_model.isChecked()
        self.tbtn_model.setEnabled(enabled)
        self.tbtn_annotation.setEnabled(enabled)
        if self._camera_is_running():
            self.cam_process.set_is_model(enabled)

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
        self.__apply_selected_backend()
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
        if self._camera_is_running():
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
        if self._camera_is_running():
            self.cam_process.set_gain_camera(float(gain))

    def __change_exposure_camera(self, exposure: int):
        value = Exposure.data.get(exposure)
        if value < 1:
            frac = Fraction(value).limit_denominator(4000)
            ratio = f"{frac.numerator}/{frac.denominator}"
            self.lbl_exposure_value.setText(f"{ratio}s")
        else:
            self.lbl_exposure_value.setText(f"{value}s")
        if self._camera_is_running():
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
        self._set_status_with_details(status, is_error=False)

    @qtc.pyqtSlot(str)
    def __get_exception(self, status):
        self.statusBar().setStyleSheet(
            """
                background-color: darkred;
                color: white;
            """
        )
        self._set_status_with_details(status, is_error=True)

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
        # If user double-clicks and we have details, show them.
        detail = self._last_error_detail or self._last_status_detail
        if detail:
            qtw.QMessageBox.information(self, 'Details', detail)
            return
        # ...existing behavior...
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