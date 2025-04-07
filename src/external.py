try:
    from PyQt6 import QtCore as qtc
    from PyQt6 import QtGui as qtg
    from PyQt6 import QtWidgets as qtw
    import sys
    import os
except ImportWarning as iw:
    print(iw)
except ImportError as ie:
    print(ie)