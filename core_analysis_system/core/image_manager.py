import cv2
import numpy as np


class ImageManager:
    SUPPORTED_FORMATS = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')

    def __init__(self):
        self._original = None
        self._processed = None
        self._filename = ""
        self._scale = 1.0
        self._scale_unit = "mm"
        self._scale_note = ""

    def load_image(self, path):
        # 使用 numpy.fromfile + cv2.imdecode 支持中文路径
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            raise ValueError(f"无法加载图像: {path}")
        self._original = img.copy()
        self._processed = img.copy()
        self._filename = path
        return img

    @property
    def original(self):
        return self._original

    @property
    def processed(self):
        return self._processed

    @processed.setter
    def processed(self, img):
        self._processed = img

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val):
        self._scale = val

    @property
    def scale_unit(self):
        return self._scale_unit

    @scale_unit.setter
    def scale_unit(self, val):
        self._scale_unit = val

    @property
    def filename(self):
        return self._filename

    def reset(self):
        self._processed = self._original.copy() if self._original is not None else None

    def get_size_mm(self):
        if self._original is None or self._scale <= 0:
            return (0, 0)
        h, w = self._original.shape[:2]
        return (w / self._scale, h / self._scale)

    def pixels_to_mm(self, pixels):
        if self._scale <= 0:
            return 0
        return pixels / self._scale

    def mm_to_pixels(self, mm):
        return mm * self._scale
