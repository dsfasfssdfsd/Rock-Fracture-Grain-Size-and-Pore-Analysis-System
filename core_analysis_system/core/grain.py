import cv2
import numpy as np


class GrainDetector:

    def detect(self, img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, markers = cv2.connectedComponents(dist)
        markers = markers + 1
        color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
        markers = cv2.watershed(color_img, markers)
        contours = []
        for marker_id in range(2, int(markers.max()) + 1):
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[markers == marker_id] = 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                contours.append(cnts[0])
        return contours

    def measure(self, contours, scale=1.0):
        results = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 10:
                continue
            perimeter = cv2.arcLength(cnt, True)
            equiv_diameter = 2 * np.sqrt(area / np.pi)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            rect = cv2.minAreaRect(cnt)
            l_min, l_max = min(rect[1]), max(rect[1])
            sphericity = l_min / l_max if l_max > 0 else 0
            results.append({
                "id": i + 1,
                "area": area / (scale ** 2),
                "perimeter": perimeter / scale,
                "equiv_diameter": equiv_diameter / scale,
                "circularity": circularity,
                "sphericity": sphericity,
            })
        return results
