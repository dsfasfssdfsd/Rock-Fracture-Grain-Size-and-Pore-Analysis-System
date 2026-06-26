import cv2
import numpy as np


class Preprocessor:

    @staticmethod
    def auto_levels(img, low_percent=1, high_percent=99):
        if len(img.shape) == 2:
            channels = [img]
        else:
            channels = cv2.split(img)
        result = []
        for ch in channels:
            low = np.percentile(ch, low_percent)
            high = np.percentile(ch, high_percent)
            if high - low < 1:
                result.append(ch)
                continue
            stretched = np.clip((ch.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
            result.append(stretched)
        if len(img.shape) == 2:
            return result[0]
        return cv2.merge(result)

    @staticmethod
    def gaussian_filter(img, sigma=1.5):
        ksize = int(2 * round(3 * sigma) + 1)
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    @staticmethod
    def clahe(img, clip_limit=2.0, grid_size=8):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        return clahe.apply(gray)

    @staticmethod
    def highpass_filter(img, sigma=3):
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    @classmethod
    def preprocess_for_fracture(cls, img, sigma=1.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        gray = cls.auto_levels(gray)
        blurred = cls.gaussian_filter(gray, sigma)
        edges = cv2.Canny(blurred, 0.1 * 255, 0.3 * 255)
        return edges

    @classmethod
    def preprocess_for_pore(cls, img):
        return cls.auto_levels(img)

    @classmethod
    def preprocess_for_grain(cls, img, sigma=1.0):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        enhanced = cls.highpass_filter(gray, sigma)
        return cls.clahe(enhanced)
