from src.external import qtw, qtc, qtg
from src.model.constants import Const
from src.model.snapshots_model import SnapshotsModel
from src.view.snapshots_dialog import Ui_dialog_snapshots
from pathlib import Path
import os

class SnapshotDialog(qtw.QDialog, Ui_dialog_snapshots):
    def __init__(self):
        super(SnapshotDialog, self).__init__()
        self.setupUi(self)
        self.setModal(True)
        self.ds_name = None
        self.snapshot_model = None
        self.path = None
        pixmap = qtg.QPixmap(os.path.join("./assets/video-camera-alt.png"))
        pixmap = pixmap.scaled(qtc.QSize(200, 200))
        self.lbl_snapshot.setPixmap(pixmap)
        self.gain = 30
        self.exposure = 33330
        self.interval = 1
        self.__init_conf()

    def __init_conf(self):
        self.le_name_dataset.textChanged.connect(self.__change_name_dataset)
        self.tbtn_directory.clicked.connect(self.__path_dialog)

        self.hs_gain.setValue(self.gain)
        self.hs_gain.valueChanged.connect(self.__change_gain)
        self.lbl_gain_value.setText(str(self.gain))

        self.rb_1_4000.setChecked(True)
        self.__get_exposure_1()
        self.__get_exposure_2()

        self.sb_interval.setValue(self.interval)
        self.sb_interval.valueChanged.connect(self.__change_interval)

        self.btn_action = qtw.QDialogButtonBox(qtw.QDialogButtonBox.StandardButton.Ok | qtw.QDialogButtonBox.StandardButton.Cancel)
        self.btn_action.accepted.connect(self.accept)
        self.btn_action.rejected.connect(self.reject)

    def __change_name_dataset(self, ds_name):
        self.ds_name = ds_name

    def __change_gain(self, gain):
        self.lbl_gain_value.setText(f"{gain}dB")
        self.gain = gain

    def __get_exposure_1(self):
        n = 1000000
        if self.rb_1_4000.isChecked():
            self.exposure = 1 / 4000 * n
        elif self.rb_1_2000.isChecked():
            self.exposure = 1 / 4000 * n
        elif self.rb_1_1000.isChecked():
            self.exposure = 1 / 1000 * n
        elif self.rb_1_500.isChecked():
            self.exposure = 1 / 500 * n
        elif self.rb_1_250.isChecked():
            self.exposure = 1 / 250 * n
        elif self.rb_1_125.isChecked():
            self.exposure = 1 / 125 * n
        elif self.rb_1_60.isChecked():
            self.exposure = 1 / 60 * n
        elif self.rb_1_30.isChecked():
            self.exposure = 1 / 30 * n

    def __get_exposure_2(self):
        n = 1000000
        if self.rb_1_15.isChecked():
            self.exposure = 1 / 15 * n
        elif self.rb_1_8.isChecked():
            self.exposure = 1 / 8 * n
        elif self.rb_1_4.isChecked():
            self.exposure = 1 / 4 * n
        elif self.rb_1_2.isChecked():
            self.exposure = 1 / 2 * n
        elif self.rb_1.isChecked():
            self.exposure = 1 * n
        elif self.rb_2.isChecked():
            self.exposure = 2 * n
        elif self.rb_4.isChecked():
            self.exposure = 4 * n
        elif self.rb_8.isChecked():
            self.exposure = 8 * n
        elif self.rb_15.isChecked():
            self.exposure = 15 * n

    def __change_interval(self, interval):
        self.interval = interval

    def __path_dialog(self):
        self.path = str(qtw.QFileDialog.getExistingDirectory(self, f"{Path.home().absolute()}"))
        self.le_directory.setText(self.path)

    def accept(self):
        self.progressBar.setValue(0)
        if self.ds_name != "" and self.path != "":
            dataset = os.path.join(f"{self.path}/{self.ds_name}")
            if not os.path.exists(dataset):
                os.makedirs(dataset)

            if not isinstance(self.snapshot_model, SnapshotsModel):
                self.snapshot_model = SnapshotsModel()

            self.snapshot_model.set_number(self.sb_number_of_images.value())
            self.progressBar.setMaximum(self.sb_number_of_images.value())
            self.snapshot_model.set_path(dataset)
            self.snapshot_model.set_scale_camera(.5)
            self.snapshot_model.set_gain_camera(self.gain)
            self.__get_exposure_1()
            self.__get_exposure_2()
            self.snapshot_model.set_exposure_camera(self.exposure)
            self.snapshot_model.set_interval_camera(self.interval)
            self.snapshot_model.update.connect(self.__update_image)
            self.snapshot_model.status.connect(self.__get_status)
            self.snapshot_model.progress.connect(self.__progress)
            self.snapshot_model.start()

    @qtc.pyqtSlot(qtg.QImage)
    def __update_image(self, img):
        self.lbl_snapshot.setPixmap(qtg.QPixmap.fromImage(img))

    @qtc.pyqtSlot(str)
    def __get_status(self, status):
        self.lbl_status.setText(status)

    @qtc.pyqtSlot(int)
    def __progress(self, val):
        self.progressBar.setValue(val)

