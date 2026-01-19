from src.external import qtw, qtc
from src.view.learningwindow import Ui_LearningWindow
from src.model.learning_process import WorkerRCNN
import pyqtgraph as pg
from pathlib import Path
import os
from typing import Optional


class LearningWindow(qtw.QMainWindow, Ui_LearningWindow):
    @staticmethod
    def normalize_annotation_text(text: Optional[str]) -> str:
        """Return raw annotation text normalized for downstream parsing.

        We keep this as a string and let WorkerRCNN normalize it into a list.
        """
        return (text or '').strip()

    @staticmethod
    def dataset_has_train_valid(dataset_path: str) -> bool:
        if not dataset_path:
            return False
        train_dir = os.path.join(dataset_path, 'train')
        valid_dir = os.path.join(dataset_path, 'valid')
        return os.path.isdir(train_dir) and os.path.isdir(valid_dir)

    def __init__(self):
        super(LearningWindow, self).__init__()
        self.setupUi(self)
        self.learning_worker = None
        self.__init_graph()
        self.__init_buttons()

    def __init_buttons(self):
        self.btn_start.clicked.connect(self.__start_learning)
        self.btn_stop.clicked.connect(self.__stop_learning)
        self.btn_clear.clicked.connect(self.__clear_graph)
        self.tbtn_dataset.clicked.connect(self.__path_directory)

    def __init_graph(self):
        # init graph deep learning
        self.graphicsView.setBackground("black")
        self.graphicsView.setXRange(0, self.sb_epoch.value(), padding=1)
        self.graphicsView.enableAutoRange()
        self.graphicsView.showGrid(x=True, y=True)
        self.graphicsView.setTitle("Record of Validation Losses", color="lightgreen", size="16pt")
        self.graphicsView.setLabel('left', "<span style=\"color:lightgreen;font-size:12pt\">Validation Loss</span>")
        self.graphicsView.setLabel('bottom', "<span style=\"color:lightgreen;font-size:12pt\">Epoch</span>")
        self.pen = pg.mkPen(cosmetic=True, width=3, color="g")

    def __path_directory(self):
        self.path = str(qtw.QFileDialog.getExistingDirectory(self, f"{Path.home().absolute()}"))
        self.le_dataset.setText(self.path)

    def upload_deep_learning_worker(self):
        annotation = self.normalize_annotation_text(self.le_annotation.text())
        name_dataset = (self.le_dataset.text() or '').strip()
        model_name = (self.le_model_name.text() or '').strip()

        self.learning_worker = WorkerRCNN(
            dataset_name=name_dataset,
            batch_size=self.sb_batch_size.value(),
            annotation=annotation,
            epochs=self.sb_epoch.value(),
            lr_step_size=self.sb_lr_step_size.value(),
            learning_rate=self.dsb_lr_rate.value(),
            model_name=model_name
        )

    def __start_learning(self):
        dataset = (self.le_dataset.text() or '').strip()
        annotation = self.normalize_annotation_text(self.le_annotation.text())
        model_name = (self.le_model_name.text() or '').strip()

        if dataset and annotation and model_name and self.dataset_has_train_valid(dataset):
            self.upload_deep_learning_worker()

            self.pte_report.clear()

            # set augmentation
            self.learning_worker.set_resize(self.sb_resize.value())
            self.learning_worker.set_random_horizontal_flip(self.dsb_hor_flip.value())
            self.learning_worker.set_random_vertical_flip(self.dsb_vert_flip.value())
            self.learning_worker.set_random_autocontrast(self.dsb_auto_contr.value())
            self.learning_worker.set_random_equalize(self.dsb_equalize.value())
            self.learning_worker.set_random_rotation(self.sb_rotation.value())

            self.learning_worker.enabled_learning_process.connect(self.__action_running_process)
            self.learning_worker.learn.connect(self.__update_deep_learning)
            self.learning_worker.learn_graph.connect(self.__update_deep_learning_graph)
            self.learning_worker.status.connect(self.__message_status)
            self.learning_worker.exception.connect(self.__message_exception)
            self.learning_worker.start()

        else:
            if not dataset:
                self.__message_exception("Dataset path is empty")
            elif not self.dataset_has_train_valid(dataset):
                self.__message_exception("Dataset must contain train/ and valid/ folders")
            elif not annotation:
                self.__message_exception("Annotation is empty")
            elif not model_name:
                self.__message_exception("Model name is empty")
            else:
                self.__message_exception("Check input data!")

    def __stop_learning(self):
        if isinstance(self.learning_worker, WorkerRCNN):
            if self.learning_worker.isRunning():
                self.learning_worker.enabled_learning_process.connect(self.__action_running_process)
                self.learning_worker.stop()

    def __clear_graph(self):
        return self.graphicsView.clear()

    @qtc.pyqtSlot(bool)
    def __action_running_process(self, val):
        if val:
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_clear.setEnabled(False)
            self.tbtn_dataset.setEnabled(False)
            self.sb_batch_size.setEnabled(False)
            self.le_annotation.setEnabled(False)
            self.sb_epoch.setEnabled(False)
            self.sb_lr_step_size.setEnabled(False)
            self.dsb_lr_rate.setEnabled(False)
            self.sb_resize.setEnabled(False)
            self.dsb_hor_flip.setEnabled(False)
            self.dsb_vert_flip.setEnabled(False)
            self.dsb_auto_contr.setEnabled(False)
            self.dsb_equalize.setEnabled(False)
            self.sb_rotation.setEnabled(False)
            self.le_model_name.setEnabled(False)
            self.graphicsView.setEnabled(False)
        else:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_clear.setEnabled(True)
            self.tbtn_dataset.setEnabled(True)
            self.tbtn_dataset.setEnabled(True)
            self.sb_batch_size.setEnabled(True)
            self.le_annotation.setEnabled(True)
            self.sb_epoch.setEnabled(True)
            self.sb_lr_step_size.setEnabled(True)
            self.dsb_lr_rate.setEnabled(True)
            self.sb_resize.setEnabled(True)
            self.dsb_hor_flip.setEnabled(True)
            self.dsb_vert_flip.setEnabled(True)
            self.dsb_auto_contr.setEnabled(True)
            self.dsb_equalize.setEnabled(True)
            self.sb_rotation.setEnabled(True)
            self.le_model_name.setEnabled(True)
            self.graphicsView.setEnabled(True)

    @qtc.pyqtSlot(str)
    def __update_deep_learning(self, val):
        self.pte_report.verticalScrollBar().setValue(self.pte_report.verticalScrollBar().maximum())
        return self.pte_report.appendPlainText(str(val))

    @qtc.pyqtSlot(list)
    def __update_deep_learning_graph(self, val):
        x, y = [], []
        for (e, l) in val:
            x.append(e), y.append(l)
        self.graphicsView.plot(x, y, name="Loss", pen=self.pen, symbol="+", symbolSize=20, symbolBrush="r")

    @qtc.pyqtSlot(str, name="exception")
    def __message_exception(self, message):
        self.statusBar().setStyleSheet("color: red; background-color: darkred")
        self.statusBar().showMessage(message)

    @qtc.pyqtSlot(str, name="bar-status")
    def __message_status(self, message):
        self.statusBar().setStyleSheet("color: lightgreen; background-color: darkgreen")
        self.statusBar().showMessage(message)