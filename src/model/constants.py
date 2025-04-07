from src.external import qtc

class Exposure:
    data = {
        1: 1/4000,
        2: 1/2000,
        3: 1/1000,
        4: 1/500,
        5: 1/250,
        6: 1/125,
        7: 1/60,
        8: 1/30,
        9: 1/15,
        10: 1/8,
        11: 1/4,
        12: 1/2,
        13: 1,
        14: 2,
        15: 4,
        16: 8,
        17: 15
    }

class Const:
    SCALE = 1
    WIDTH = 800
    HEIGHT = 600
    SIZE = qtc.QSize(int(WIDTH * SCALE), int(HEIGHT * SCALE))
    KEEP_ASPECT_RATION = qtc.Qt.AspectRatioMode.KeepAspectRatio
    KEEP_ASPECT_RATION_BY_EXPANDING = qtc.Qt.AspectRatioMode.KeepAspectRatioByExpanding
    FAST_TRANSFORMATION = qtc.Qt.TransformationMode.FastTransformation
    SMOOTH_TRANSFORMATION = qtc.Qt.TransformationMode.SmoothTransformation
    FONT_SCALE_FPS = 2
    RECT_COLOR_BG = (0, 0, 0)
    START_POINT_BG_FPS = (20, 40)
    END_POINT_BG_FPS = (370, 120)
    RECT_COLOR_TEXT_FPS = (255, 255, 255)
    ORG_TEXT_FPS = (40, 100)
    THICKNEES = 3
    THICKNEES_RECTANGLE = -1
    GAIN = 36
    EXPOSURE = 250
    WAIT_EXPOSURE = 1000000
    STEP = .05