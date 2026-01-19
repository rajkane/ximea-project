try:
    from PyQt6 import QtCore as qtc
    from PyQt6 import QtGui as qtg
    from PyQt6 import QtWidgets as qtw
    import sys
    import os
except Exception as ie:  # fallback stub to allow headless testing
    import sys
    import os
    import threading

    class _DummySignal:
        def __init__(self, *args, **kwargs):
            pass
        def connect(self, *args, **kwargs):
            pass
        def emit(self, *args, **kwargs):
            pass

    class _DummyQThread:
        def __init__(self, *args, **kwargs):
            # mimic minimal QThread interface used in the project
            pass
        def start(self):
            pass
        def quit(self):
            pass

    # Simple object to mirror the subset of QtCore used in the project
    class _qtc_stub:
        QThread = _DummyQThread
        pyqtSignal = lambda *args, **kwargs: _DummySignal()
        pyqtSlot = lambda *args, **kwargs: (lambda f: f)

    qtc = _qtc_stub()
    qtg = None
    qtw = None
