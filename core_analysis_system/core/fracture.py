import cv2
import numpy as np


class FractureDetector:

    def __init__(self, low_threshold=0.1, high_threshold=0.3, min_length=20):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.min_length = min_length

    def detect(self, img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        edges = cv2.Canny(gray, self.low_threshold * 255, self.high_threshold * 255)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        lines = self._detect_lines(closed)
        return lines, closed

    def _detect_lines(self, binary):
        lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=20, minLineLength=self.min_length, maxLineGap=10)
        if lines is None:
            return []
        return [(x1, y1, x2, y2) for line in lines for x1, y1, x2, y2 in line]

    def measure(self, binary, lines, scale=1.0):
        total_area = float(np.sum(binary > 0))
        total_length = 0.0
        for x1, y1, x2, y2 in lines:
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total_length += length
        fracture_count = len(lines)
        width = total_area / total_length if total_length > 0 else 0
        return {
            "fracture_count": fracture_count,
            "total_length": total_length / scale,
            "total_area": total_area / (scale ** 2),
            "avg_width": width / scale,
        }
