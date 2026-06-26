import cv2
import numpy as np


class PoreDetector:

    def __init__(self, similarity_threshold=30):
        self.similarity_threshold = similarity_threshold

    def detect_from_seed(self, img, seed_point, threshold=None):
        if threshold is not None:
            self.similarity_threshold = threshold
        h, w = img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        diff = (int(self.similarity_threshold),) * 3
        if len(img.shape) == 2:
            diff = int(self.similarity_threshold)
        cv2.floodFill(img, mask, seed_point, (255, 255, 255), diff, diff,
                      cv2.FLOODFILL_FIXED_RANGE)
        return mask[1:-1, 1:-1]

    def detect_auto(self, img, pore_size_scale=1.0):
        """孔洞自动检测 —— Hough圆检测 + 内部暗度验证。

        思路：
        1. 高斯模糊去噪
        2. Hough圆变换检测圆孔
        3. 验证圆内部是否足够暗

        参数:
            pore_size_scale: 孔洞大小调节系数
                - <1.0: 识别更小的孔洞（降低最小半径）
                - =1.0: 默认大小
                - >1.0: 识别更大的孔洞（提高最小半径，过滤小孔）
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        h, w = gray.shape
        total = h * w

        # 1. 高斯模糊去噪（Hough变换对噪声敏感）
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

        # 2. 估算孔洞半径范围
        # scale越大，最小半径越大（只保留大孔洞）
        min_r = int(max(5, min(h, w) * 0.02) * pore_size_scale)
        max_r = int(min(h, w) * 0.25)
        min_dist = max(min_r * 2, 20)  # 圆心之间的最小距离

        # 3. Hough圆变换检测圆孔
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.5,           # 累加器分辨率
            minDist=min_dist, # 圆心最小间距
            param1=80,        # Canny高阈值
            param2=45,        # 累加器阈值（提高，减少误检）
            minRadius=min_r,
            maxRadius=max_r
        )

        if circles is None:
            return []

        circles = np.uint16(np.around(circles[0, :]))

        # 4. 验证：圆内部必须比周围暗
        filtered = []
        global_mean = gray.mean()

        for (x, y, r) in circles:
            # 创建圆内部mask
            mask_inner = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask_inner, (int(x), int(y)), int(r), 255, -1)
            inner_mean = cv2.mean(gray, mask=mask_inner)[0]

            # 创建环带mask（圆外一圈）
            mask_outer = np.zeros((h, w), dtype=np.uint8)
            ring_w = max(5, int(r * 0.5))
            cv2.circle(mask_outer, (int(x), int(y)), int(r + ring_w), 255, -1)
            ring_mask = cv2.subtract(mask_outer, mask_inner)

            if np.sum(ring_mask) > 0:
                outer_mean = cv2.mean(gray, mask=ring_mask)[0]
            else:
                outer_mean = global_mean

            # 内部必须比周围暗（提高到至少暗15灰度值，减少误检）
            if outer_mean - inner_mean < 15:
                continue

            # 面积检查
            area = np.pi * r * r
            if area > total * 0.10:
                continue

            # 生成轮廓（用圆的轮廓代替）
            cnt = []
            for angle in np.linspace(0, 2 * np.pi, 60, endpoint=False):
                px = int(x + r * np.cos(angle))
                py = int(y + r * np.sin(angle))
                cnt.append([[px, py]])
            cnt = np.array(cnt, dtype=np.int32)
            filtered.append(cnt)

        return filtered

    def measure(self, contours, scale=1.0):
        results = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 10:
                continue
            perimeter = cv2.arcLength(cnt, True)
            equiv_diameter = 2 * np.sqrt(area / np.pi)
            results.append({
                "id": i + 1,
                "area": area / (scale ** 2),
                "perimeter": perimeter / scale,
                "equiv_diameter": equiv_diameter / scale,
            })
        return results
