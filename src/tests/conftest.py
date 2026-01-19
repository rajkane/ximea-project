import os

# Make PyQt6/Qt usable in headless test environments (no DISPLAY)
# This must be set before importing PyQt6.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
