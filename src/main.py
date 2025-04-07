from src.external import sys, qtw
from src.controller.camera_controller import MainWindow


def run():
    app = qtw.QApplication(sys.argv)
    app.setApplicationVersion("v1.0")
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()