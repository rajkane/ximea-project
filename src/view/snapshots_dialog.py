# Form implementation generated from reading ui file 'src/view/snapshots_dialog.ui'
#
# Created by: PyQt6 UI code generator 6.8.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_dialog_snapshots(object):
    def setupUi(self, dialog_snapshots):
        dialog_snapshots.setObjectName("dialog_snapshots")
        dialog_snapshots.resize(705, 574)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(dialog_snapshots.sizePolicy().hasHeightForWidth())
        dialog_snapshots.setSizePolicy(sizePolicy)
        dialog_snapshots.setMaximumSize(QtCore.QSize(800, 600))
        dialog_snapshots.setStyleSheet("background-color: rgb(36, 31, 49);\n"
"color: rgb(255, 255, 255);")
        self.gridLayout_2 = QtWidgets.QGridLayout(dialog_snapshots)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.rb_4 = QtWidgets.QRadioButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_4.sizePolicy().hasHeightForWidth())
        self.rb_4.setSizePolicy(sizePolicy)
        self.rb_4.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.rb_4.setObjectName("rb_4")
        self.gridLayout.addWidget(self.rb_4, 12, 3, 1, 1)
        self.lbl_status = QtWidgets.QLabel(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_status.sizePolicy().hasHeightForWidth())
        self.lbl_status.setSizePolicy(sizePolicy)
        self.lbl_status.setObjectName("lbl_status")
        self.gridLayout.addWidget(self.lbl_status, 18, 1, 1, 2)
        self.rb_2 = QtWidgets.QRadioButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_2.sizePolicy().hasHeightForWidth())
        self.rb_2.setSizePolicy(sizePolicy)
        self.rb_2.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.rb_2.setObjectName("rb_2")
        self.gridLayout.addWidget(self.rb_2, 11, 3, 1, 1)
        self.rb_1_1000 = QtWidgets.QRadioButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_1_1000.sizePolicy().hasHeightForWidth())
        self.rb_1_1000.setSizePolicy(sizePolicy)
        self.rb_1_1000.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.rb_1_1000.setObjectName("rb_1_1000")
        self.gridLayout.addWidget(self.rb_1_1000, 10, 0, 1, 1)
        self.rb_15 = QtWidgets.QRadioButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_15.sizePolicy().hasHeightForWidth())
        self.rb_15.setSizePolicy(sizePolicy)
        self.rb_15.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.rb_15.setObjectName("rb_15")
        self.gridLayout.addWidget(self.rb_15, 14, 3, 1, 1)
        self.rb_1_60 = QtWidgets.QRadioButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_1_60.sizePolicy().hasHeightForWidth())
        self.rb_1_60.setSizePolicy(sizePolicy)
        self.rb_1_60.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.rb_1_60.setObjectName("rb_1_60")
        self.gridLayout.addWidget(self.rb_1_60, 14, 0, 1, 1)
        self.lbl_snapshot = QtWidgets.QLabel(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_snapshot.sizePolicy().hasHeightForWidth())
        self.lbl_snapshot.setSizePolicy(sizePolicy)
        self.lbl_snapshot.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.lbl_snapshot.setStyleSheet("background-color: rgb(61, 56, 70);")
        self.lbl_snapshot.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_snapshot.setObjectName("lbl_snapshot")
        self.gridLayout.addWidget(self.lbl_snapshot, 8, 1, 10, 2)
        self.rb_1_125 = QtWidgets.QRadioButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_1_125.sizePolicy().hasHeightForWidth())
        self.rb_1_125.setSizePolicy(sizePolicy)
        self.rb_1_125.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.rb_1_125.setObjectName("rb_1_125")
        self.gridLayout.addWidget(self.rb_1_125, 13, 0, 1, 1)
        self.rb_1_2000 = QtWidgets.QRadioButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_1_2000.sizePolicy().hasHeightForWidth())
        self.rb_1_2000.setSizePolicy(sizePolicy)
        self.rb_1_2000.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.rb_1_2000.setObjectName("rb_1_2000")
        self.gridLayout.addWidget(self.rb_1_2000, 9, 0, 1, 1)
        self.rb_1_2 = QtWidgets.QRadioButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_1_2.sizePolicy().hasHeightForWidth())
        self.rb_1_2.setSizePolicy(sizePolicy)
        self.rb_1_2.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.rb_1_2.setObjectName("rb_1_2")
        self.gridLayout.addWidget(self.rb_1_2, 9, 3, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(parent=dialog_snapshots)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.progressBar, 19, 0, 1, 4)
        self.rb_1_4000 = QtWidgets.QRadioButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_1_4000.sizePolicy().hasHeightForWidth())
        self.rb_1_4000.setSizePolicy(sizePolicy)
        self.rb_1_4000.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.rb_1_4000.setObjectName("rb_1_4000")
        self.gridLayout.addWidget(self.rb_1_4000, 8, 0, 1, 1)
        self.le_name_dataset = QtWidgets.QLineEdit(parent=dialog_snapshots)
        self.le_name_dataset.setObjectName("le_name_dataset")
        self.gridLayout.addWidget(self.le_name_dataset, 0, 0, 1, 3)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.gridLayout.addItem(spacerItem, 18, 0, 1, 1)
        self.rb_1_500 = QtWidgets.QRadioButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_1_500.sizePolicy().hasHeightForWidth())
        self.rb_1_500.setSizePolicy(sizePolicy)
        self.rb_1_500.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.rb_1_500.setObjectName("rb_1_500")
        self.gridLayout.addWidget(self.rb_1_500, 11, 0, 1, 1)
        self.lbl_exposure = QtWidgets.QLabel(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_exposure.sizePolicy().hasHeightForWidth())
        self.lbl_exposure.setSizePolicy(sizePolicy)
        self.lbl_exposure.setObjectName("lbl_exposure")
        self.gridLayout.addWidget(self.lbl_exposure, 6, 0, 1, 1)
        self.hs_gain = QtWidgets.QSlider(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hs_gain.sizePolicy().hasHeightForWidth())
        self.hs_gain.setSizePolicy(sizePolicy)
        self.hs_gain.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.hs_gain.setMinimum(1)
        self.hs_gain.setMaximum(72)
        self.hs_gain.setPageStep(1)
        self.hs_gain.setProperty("value", 30)
        self.hs_gain.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.hs_gain.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.hs_gain.setTickInterval(1)
        self.hs_gain.setObjectName("hs_gain")
        self.gridLayout.addWidget(self.hs_gain, 4, 1, 1, 2)
        self.tbtn_directory = QtWidgets.QToolButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tbtn_directory.sizePolicy().hasHeightForWidth())
        self.tbtn_directory.setSizePolicy(sizePolicy)
        self.tbtn_directory.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.tbtn_directory.setObjectName("tbtn_directory")
        self.gridLayout.addWidget(self.tbtn_directory, 1, 3, 1, 1)
        self.rb_8 = QtWidgets.QRadioButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_8.sizePolicy().hasHeightForWidth())
        self.rb_8.setSizePolicy(sizePolicy)
        self.rb_8.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.rb_8.setObjectName("rb_8")
        self.gridLayout.addWidget(self.rb_8, 13, 3, 1, 1)
        self.lbl_gain_value = QtWidgets.QLabel(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_gain_value.sizePolicy().hasHeightForWidth())
        self.lbl_gain_value.setSizePolicy(sizePolicy)
        self.lbl_gain_value.setObjectName("lbl_gain_value")
        self.gridLayout.addWidget(self.lbl_gain_value, 4, 3, 1, 1)
        self.rb_1_30 = QtWidgets.QRadioButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_1_30.sizePolicy().hasHeightForWidth())
        self.rb_1_30.setSizePolicy(sizePolicy)
        self.rb_1_30.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.rb_1_30.setObjectName("rb_1_30")
        self.gridLayout.addWidget(self.rb_1_30, 15, 0, 1, 1)
        self.rb_1_15 = QtWidgets.QRadioButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_1_15.sizePolicy().hasHeightForWidth())
        self.rb_1_15.setSizePolicy(sizePolicy)
        self.rb_1_15.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.rb_1_15.setObjectName("rb_1_15")
        self.gridLayout.addWidget(self.rb_1_15, 16, 0, 1, 1)
        self.lbl_gain = QtWidgets.QLabel(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_gain.sizePolicy().hasHeightForWidth())
        self.lbl_gain.setSizePolicy(sizePolicy)
        self.lbl_gain.setObjectName("lbl_gain")
        self.gridLayout.addWidget(self.lbl_gain, 4, 0, 1, 1)
        self.rb_1 = QtWidgets.QRadioButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_1.sizePolicy().hasHeightForWidth())
        self.rb_1.setSizePolicy(sizePolicy)
        self.rb_1.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.rb_1.setObjectName("rb_1")
        self.gridLayout.addWidget(self.rb_1, 10, 3, 1, 1)
        self.le_directory = QtWidgets.QLineEdit(parent=dialog_snapshots)
        self.le_directory.setReadOnly(True)
        self.le_directory.setObjectName("le_directory")
        self.gridLayout.addWidget(self.le_directory, 1, 0, 1, 3)
        self.rb_1_4 = QtWidgets.QRadioButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_1_4.sizePolicy().hasHeightForWidth())
        self.rb_1_4.setSizePolicy(sizePolicy)
        self.rb_1_4.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.rb_1_4.setObjectName("rb_1_4")
        self.gridLayout.addWidget(self.rb_1_4, 8, 3, 1, 1)
        self.rb_1_8 = QtWidgets.QRadioButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_1_8.sizePolicy().hasHeightForWidth())
        self.rb_1_8.setSizePolicy(sizePolicy)
        self.rb_1_8.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.rb_1_8.setObjectName("rb_1_8")
        self.gridLayout.addWidget(self.rb_1_8, 17, 0, 1, 1)
        self.lbl_num_img = QtWidgets.QLabel(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_num_img.sizePolicy().hasHeightForWidth())
        self.lbl_num_img.setSizePolicy(sizePolicy)
        self.lbl_num_img.setObjectName("lbl_num_img")
        self.gridLayout.addWidget(self.lbl_num_img, 2, 0, 1, 1)
        self.rb_1_250 = QtWidgets.QRadioButton(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_1_250.sizePolicy().hasHeightForWidth())
        self.rb_1_250.setSizePolicy(sizePolicy)
        self.rb_1_250.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.rb_1_250.setObjectName("rb_1_250")
        self.gridLayout.addWidget(self.rb_1_250, 12, 0, 1, 1)
        self.lbl_interval = QtWidgets.QLabel(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_interval.sizePolicy().hasHeightForWidth())
        self.lbl_interval.setSizePolicy(sizePolicy)
        self.lbl_interval.setObjectName("lbl_interval")
        self.gridLayout.addWidget(self.lbl_interval, 3, 0, 1, 1)
        self.sb_number_of_images = QtWidgets.QSpinBox(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sb_number_of_images.sizePolicy().hasHeightForWidth())
        self.sb_number_of_images.setSizePolicy(sizePolicy)
        self.sb_number_of_images.setMaximumSize(QtCore.QSize(80, 16777215))
        self.sb_number_of_images.setSpecialValueText("")
        self.sb_number_of_images.setAccelerated(True)
        self.sb_number_of_images.setMinimum(1)
        self.sb_number_of_images.setMaximum(500)
        self.sb_number_of_images.setObjectName("sb_number_of_images")
        self.gridLayout.addWidget(self.sb_number_of_images, 2, 1, 1, 1)
        self.sb_interval = QtWidgets.QSpinBox(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sb_interval.sizePolicy().hasHeightForWidth())
        self.sb_interval.setSizePolicy(sizePolicy)
        self.sb_interval.setMaximumSize(QtCore.QSize(80, 16777215))
        self.sb_interval.setMinimum(1)
        self.sb_interval.setMaximum(300)
        self.sb_interval.setObjectName("sb_interval")
        self.gridLayout.addWidget(self.sb_interval, 3, 1, 1, 2)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(parent=dialog_snapshots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.buttonBox.sizePolicy().hasHeightForWidth())
        self.buttonBox.setSizePolicy(sizePolicy)
        self.buttonBox.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout_2.addWidget(self.buttonBox, 0, 1, 1, 1)

        self.retranslateUi(dialog_snapshots)
        self.buttonBox.accepted.connect(dialog_snapshots.accept) # type: ignore
        self.buttonBox.rejected.connect(dialog_snapshots.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(dialog_snapshots)
        dialog_snapshots.setTabOrder(self.le_name_dataset, self.tbtn_directory)
        dialog_snapshots.setTabOrder(self.tbtn_directory, self.sb_number_of_images)
        dialog_snapshots.setTabOrder(self.sb_number_of_images, self.hs_gain)
        dialog_snapshots.setTabOrder(self.hs_gain, self.le_directory)

    def retranslateUi(self, dialog_snapshots):
        _translate = QtCore.QCoreApplication.translate
        dialog_snapshots.setWindowTitle(_translate("dialog_snapshots", "Fill Dataset"))
        self.rb_4.setText(_translate("dialog_snapshots", "4s"))
        self.lbl_status.setText(_translate("dialog_snapshots", "Status"))
        self.rb_2.setText(_translate("dialog_snapshots", "2s"))
        self.rb_1_1000.setText(_translate("dialog_snapshots", "1/1000s"))
        self.rb_15.setText(_translate("dialog_snapshots", "15s"))
        self.rb_1_60.setText(_translate("dialog_snapshots", "1/60s"))
        self.lbl_snapshot.setText(_translate("dialog_snapshots", "Snapshot"))
        self.rb_1_125.setText(_translate("dialog_snapshots", "1/125s"))
        self.rb_1_2000.setText(_translate("dialog_snapshots", "1/2000s"))
        self.rb_1_2.setText(_translate("dialog_snapshots", "1/2s"))
        self.rb_1_4000.setText(_translate("dialog_snapshots", "1/4000s"))
        self.le_name_dataset.setPlaceholderText(_translate("dialog_snapshots", "Enter Name Dataset"))
        self.rb_1_500.setText(_translate("dialog_snapshots", "1/500s"))
        self.lbl_exposure.setText(_translate("dialog_snapshots", "Exposure"))
        self.tbtn_directory.setText(_translate("dialog_snapshots", "..."))
        self.rb_8.setText(_translate("dialog_snapshots", "8s"))
        self.lbl_gain_value.setText(_translate("dialog_snapshots", "dB"))
        self.rb_1_30.setText(_translate("dialog_snapshots", "1/30s"))
        self.rb_1_15.setText(_translate("dialog_snapshots", "1/15s"))
        self.lbl_gain.setText(_translate("dialog_snapshots", "Gain"))
        self.rb_1.setText(_translate("dialog_snapshots", "1s"))
        self.le_directory.setPlaceholderText(_translate("dialog_snapshots", "Dataset Path"))
        self.rb_1_4.setText(_translate("dialog_snapshots", "1/4s"))
        self.rb_1_8.setText(_translate("dialog_snapshots", "1/8s"))
        self.lbl_num_img.setText(_translate("dialog_snapshots", "Img Num"))
        self.rb_1_250.setText(_translate("dialog_snapshots", "1/250s"))
        self.lbl_interval.setText(_translate("dialog_snapshots", "Interval"))
