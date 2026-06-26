"""
基于FastSAM-x.pt的精确粒度和孔洞识别算法
使用提示点(Prompt Points)引导FastSAM进行精确分割
"""
import os
import sys
import cv2
import numpy as np

# 在导入torch之前添加DLL路径（解决GUI中c10.dll加载失败问题）
def _setup_torch_dll_path():
    try:
        import importlib.util
        spec = importlib.util.find_spec('torch')
        if spec and spec.origin:
            torch_lib = os.path.join(os.path.dirname(spec.origin), 'lib')
            if os.path.exists(torch_lib):
                os.add_dll_directory(torch_lib)
                return True
    except Exception:
        pass
    return False

_torch_dll_ready = _setup_torch_dll_path()


# 全局模型实例（延迟加载）
_fastsam_model = None
_fastsam_available = None


def _get_fastsam_model():
    """延迟加载FastSAM模型"""
    global _fastsam_model, _fastsam_available
    if _fastsam_available is False:
        return None
    if _fastsam_model is None:
        try:
            from ultralytics import FastSAM
            model_path = r'C:\Users\ZhuanZ1\Desktop\code1\code1\FastSAM-x.pt'
            _fastsam_model = FastSAM(model_path)
            _fastsam_available = True
        except Exception as e:
            print(f"FastSAM加载失败: {e}")
            _fastsam_available = False
            return None
    return _fastsam_model


def _is_fastsam_available():
    """FastSAM是否可用"""
    if _fastsam_available is None:
        _get_fastsam_model()
    return _fastsam_available


class FastSAMGrainDetector:
    """基于传统CV方法的粒度检测器（Watershed分割）"""

    def __init__(self,
                 min_area=30, max_area_ratio=0.3,
                 min_circularity=0.05, min_solidity=0.3,
                 conf=0.25, iou=0.8, imgsz=640,
                 min_dist=10):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_circularity = min_circularity
        self.min_solidity = min_solidity
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.min_dist = min_dist

    def _generate_prompt_points(self, img):
        """生成提示点：使用Watershed分割找颗粒中心"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        h, w = gray.shape

        # Otsu阈值获取颗粒区域
        _, binary = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 形态学去噪
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # 距离变换
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 找局部极大值作为种子点（降低阈值增加分割数量）
        threshold_val = int(dist_norm.max() * 0.15)  # 进一步降低阈值
        _, seeds = cv2.threshold(dist_norm, threshold_val, 255, cv2.THRESH_BINARY)

        # 连通域标记
        _, markers = cv2.connectedComponents(seeds)
        markers = markers + 1  # 避免背景为0
        markers[binary == 0] = 1  # 背景区域标记为1

        # Watershed分割
        color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(img.shape) != 3 else img.copy()
        markers = cv2.watershed(color_img, markers)

        # 找每个分割区域的中心点
        points = []
        for marker_id in range(2, int(markers.max()) + 1):
            region_mask = np.zeros((h, w), dtype=np.uint8)
            region_mask[markers == marker_id] = 255

            # 计算区域中心
            M = cv2.moments(region_mask)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m00'] / M['m00'])

                # 检查中心点是否在颗粒区域内
                if binary[cy, cx] > 0:
                    # 计算区域面积
                    area = np.sum(region_mask > 0)
                    if area >= self.min_area:
                        points.append([cx, cy])

        # 过滤距离太近的点
        filtered_points = []
        for p in points:
            too_close = False
            for fp in filtered_points:
                dist_sq = (p[0] - fp[0])**2 + (p[1] - fp[1])**2
                if dist_sq < self.min_dist**2:
                    too_close = True
                    break
            if not too_close:
                filtered_points.append(p)

        return filtered_points, binary

    def _detect_with_fastsam(self, img, points):
        """使用FastSAM提示点分割"""
        model = _get_fastsam_model()
        if model is None or len(points) == 0:
            return [], np.zeros(img.shape[:2], dtype=np.uint8)

        h, w = img.shape[:2]
        total_area = h * w

        try:
            # FastSAM提示点格式：[[x1,y1], [x2,y2], ...]
            # labels: 1表示前景点，0表示背景点
            points_array = np.array(points)
            labels = np.ones(len(points))  # 全部是前景点

            # 调用FastSAM
            results = model(img, points=points_array, labels=labels,
                            imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                            retina_masks=True, device='cpu')

            if results[0].masks is None:
                return [], np.zeros((h, w), dtype=np.uint8)

            masks = results[0].masks.data.cpu().numpy()

            contours_list = []
            combined_mask = np.zeros((h, w), dtype=np.uint8)

            for mask in masks:
                # 缩放到原图尺寸
                mask_resized = cv2.resize(mask.astype(np.uint8), (w, h),
                                          interpolation=cv2.INTER_NEAREST)

                area = mask_resized.sum()
                if area < self.min_area or area / total_area > self.max_area_ratio:
                    continue

                # 找轮廓
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                cnt = max(contours, key=cv2.contourArea)
                cnt_area = cv2.contourArea(cnt)

                # 圆形度过滤
                perimeter = cv2.arcLength(cnt, True)
                if perimeter < 1:
                    continue
                circularity = 4 * np.pi * cnt_area / (perimeter * perimeter)
                if circularity < self.min_circularity:
                    continue

                # 实体度过滤
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0 and cnt_area / hull_area < self.min_solidity:
                    continue

                contours_list.append(cnt)
                cv2.drawContours(combined_mask, [cnt], -1, 255, -1)

            return contours_list, combined_mask

        except Exception as e:
            print(f"FastSAM推理错误: {e}")
            return [], np.zeros((h, w), dtype=np.uint8)

    def _detect_fallback(self, img):
        """精确粒度检测 v2：多策略二值化 + Watershed + 边缘精化"""
        h, w = img.shape[:2]
        total_area = h * w

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 多策略二值化
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        th_adapt = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 5
        )

        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_channel)
            _, th_lab = cv2.threshold(l_enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            th_lab = th_otsu.copy()

        # 融合：多数票
        th_fused = np.zeros_like(gray)
        vote = (th_otsu > 0).astype(int) + (th_adapt > 0).astype(int) + (th_lab > 0).astype(int)
        th_fused[vote >= 2] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        th_fused = cv2.morphologyEx(th_fused, cv2.MORPH_OPEN, kernel, iterations=1)
        th_fused = cv2.morphologyEx(th_fused, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Watershed分割
        dist = cv2.distanceTransform(th_fused, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

        _, sure_fg1 = cv2.threshold(dist_norm, 0.55, 255, cv2.THRESH_BINARY)
        _, sure_fg2 = cv2.threshold(dist_norm, 0.35, 255, cv2.THRESH_BINARY)
        sure_fg = cv2.bitwise_or(sure_fg1.astype(np.uint8), sure_fg2.astype(np.uint8))

        sure_bg = cv2.dilate(th_fused, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)

        n_markers, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(img.shape) != 3 else img.copy()
        markers = cv2.watershed(color_img, markers)

        out_mask = np.zeros((h, w), dtype=np.uint8)
        for label in range(2, n_markers + 1):
            out_mask[markers == label] = 255
        out_mask[markers == -1] = 0

        # 边缘精化
        edges = cv2.Canny(gray, 30, 100)
        edges_dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

        mask_refined = cv2.bitwise_and(out_mask, out_mask, mask=cv2.bitwise_not(edges_dilated))
        mask_core = cv2.erode(out_mask, kernel, iterations=2)
        mask_final = cv2.bitwise_or(mask_core, mask_refined)

        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 连通域过滤
        contours_list = []
        combined_mask = np.zeros_like(mask_final)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_final, connectivity=8)

        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area or area / total_area > self.max_area_ratio:
                continue

            region = (labels == i).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            cnt = cnts[0]
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0 and area / hull_area < self.min_solidity:
                continue

            contours_list.append(cnt)
            cv2.drawContours(combined_mask, [cnt], -1, 255, -1)

        return contours_list, combined_mask

    def detect(self, img):
        """检测颗粒 - 使用传统CV Watershed方法"""
        # FastSAM提示点模式不稳定，使用传统CV方法
        return self._detect_fallback(img)


class FastSAMPoreDetector:
    """基于多尺度暗度投票的精确孔洞检测器
    支持FastSAM子进程调用（解决GUI中PyTorch DLL加载失败问题）
    """

    def __init__(self,
                 min_area=50, max_area_ratio=0.2,
                 min_circularity=0.15, max_circularity=1.0,
                 min_solidity=0.4,
                 conf=0.15, iou=0.7, imgsz=640,
                 min_dist=10,
                 dark_pct=25, min_votes=1,
                 scales=(21, 51, 101),
                 close_kernel=7, close_iter=2,
                 open_kernel=5, open_iter=1,
                 ws_thresh=0.35):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_circularity = min_circularity
        self.max_circularity = max_circularity
        self.min_solidity = min_solidity
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.min_dist = min_dist
        self.dark_pct = dark_pct
        self.min_votes = min_votes
        self.scales = scales
        self.close_kernel = close_kernel
        self.close_iter = close_iter
        self.open_kernel = open_kernel
        self.open_iter = open_iter
        self.ws_thresh = ws_thresh
        self._model = None
        self._fastsam_available = None
        self._use_subprocess = False

    def _load_model(self):
        """延迟加载FastSAM模型（优先常驻子进程，其次直接加载，最后一次性子进程）"""
        if self._model is None and self._fastsam_available is not False:
            # 1. 优先尝试常驻子进程
            try:
                from core.fastsam_daemon import get_fastsam_daemon
                daemon = get_fastsam_daemon()
                if daemon._ensure_started():
                    self._model = 'daemon'
                    self._fastsam_available = True
                    self._use_subprocess = True
                    return self._model
            except Exception:
                pass

            # 2. 尝试直接加载
            try:
                from ultralytics import FastSAM
                model_path = r'C:\Users\ZhuanZ1\Desktop\code1\code1\FastSAM-x.pt'
                self._model = FastSAM(model_path)
                self._fastsam_available = True
                self._use_subprocess = False
                return self._model
            except Exception:
                pass

            # 3. 回退到一次性子进程
            try:
                from core.fastsam_subprocess import run_fastsam_subprocess
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                test_result = run_fastsam_subprocess(
                    test_img, conf=0.5, iou=0.7, imgsz=128)
                if test_result is not None:
                    self._fastsam_available = True
                    self._use_subprocess = True
                    self._model = 'subprocess'
                    return self._model
            except Exception:
                pass

            self._fastsam_available = False
            self._model = None
        return self._model

    def detect(self, img):
        """检测孔洞
        优先FastSAM，失败则用多尺度暗度投票
        自适应图像尺寸
        """
        h, w = img.shape[:2]
        base_area = 250000
        cur_area = h * w
        scale_factor = (cur_area / base_area) ** 0.5

        orig = {
            'min_area': self.min_area,
            'close_kernel': self.close_kernel,
            'open_kernel': self.open_kernel,
            'scales': self.scales,
        }

        try:
            if scale_factor != 1.0:
                self.min_area = max(10, int(self.min_area * scale_factor * scale_factor))
                self.close_kernel = max(3, int(self.close_kernel * scale_factor))
                if self.close_kernel % 2 == 0:
                    self.close_kernel += 1
                self.open_kernel = max(3, int(self.open_kernel * scale_factor))
                if self.open_kernel % 2 == 0:
                    self.open_kernel += 1
                self.scales = tuple(max(3, int(s * scale_factor)) for s in orig['scales'])

            # 尝试FastSAM
            model = self._load_model()
            if model is not None:
                return self._detect_fastsam_main(img, model)
            # 回退到暗度投票
            return self._detect_fallback(img)
        finally:
            self.min_area = orig['min_area']
            self.close_kernel = orig['close_kernel']
            self.open_kernel = orig['open_kernel']
            self.scales = orig['scales']

    def _filter_contours(self, contours, h, w):
        """形状过滤：面积、圆形度、实体度"""
        total_area = h * w
        keep = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area / total_area > self.max_area_ratio:
                continue
            perim = cv2.arcLength(cnt, True)
            if perim < 1:
                continue
            circ = 4 * np.pi * area / (perim * perim)
            if circ < self.min_circularity or circ > self.max_circularity:
                continue
            hull = cv2.convexHull(cnt)
            hull_a = cv2.contourArea(hull)
            solid = area / hull_a if hull_a > 0 else 0
            if hull_a > 0 and solid < self.min_solidity:
                continue
            keep.append(cnt)
        return keep

    def _detect_fastsam_main(self, img, model):
        """FastSAM主检测 + 形状过滤"""
        h, w = img.shape[:2]

        if isinstance(model, str) and model == 'daemon':
            # 常驻子进程方式
            from core.fastsam_daemon import run_fastsam_daemon
            fastsam_contours = run_fastsam_daemon(
                img, conf=self.conf, iou=self.iou, imgsz=self.imgsz)
            if fastsam_contours is None:
                return self._detect_fallback(img)
        elif isinstance(model, str) and model == 'subprocess':
            # 一次性子进程方式
            from core.fastsam_subprocess import run_fastsam_subprocess
            fastsam_contours = run_fastsam_subprocess(
                img, conf=self.conf, iou=self.iou, imgsz=self.imgsz)
            if fastsam_contours is None:
                return self._detect_fallback(img)
        else:
            # 直接调用方式
            results = model(img, imgsz=self.imgsz, conf=self.conf,
                            iou=self.iou, device='cpu', retina_masks=True)

            fastsam_contours = []
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                for m in masks:
                    m_resized = cv2.resize(
                        m.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST)
                    cnts, _ = cv2.findContours(
                        m_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        fastsam_contours.append(max(cnts, key=cv2.contourArea))

        # 形状过滤
        filtered = self._filter_contours(fastsam_contours, h, w)

        # 生成combined_mask
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for cnt in filtered:
            cv2.drawContours(combined_mask, [cnt], -1, 255, -1)

        return filtered, combined_mask

    def _get_dark_mask(self, gray):
        """多尺度暗度投票：找比周围暗的区域（孔洞）"""
        h, w = gray.shape
        gray_f = gray.astype(np.float32)

        vote = np.zeros((h, w), dtype=np.uint8)
        for k in self.scales:
            ksize = k if k % 2 == 1 else k + 1
            local_mean = cv2.blur(gray_f, (ksize, ksize))
            diff = local_mean - gray_f
            thresh = np.percentile(diff.ravel(), 100 - self.dark_pct)
            vote += (diff >= thresh).astype(np.uint8)

        result = np.zeros((h, w), dtype=np.uint8)
        result[vote >= self.min_votes] = 255
        return result

    def _detect_fallback(self, img):
        """多尺度暗度投票 + 形态学 + Watershed + 形状过滤"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        h, w = gray.shape
        total_area = h * w

        dark_mask = self._get_dark_mask(gray)

        kernel_c = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.close_kernel, self.close_kernel))
        mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel_c,
                                iterations=self.close_iter)

        kernel_o = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.open_kernel, self.open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_o,
                                iterations=self.open_iter)

        contours_fill, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_fill:
            if cv2.contourArea(cnt) > max(50, self.min_area // 3):
                cv2.drawContours(mask, [cnt], -1, 255, -1)

        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        _, sure_fg = cv2.threshold(dist_norm, self.ws_thresh, 255,
                                    cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)
        sure_bg = cv2.dilate(mask, kernel_c, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)

        n_markers, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        color_img = img.copy() if len(img.shape) == 3 else \
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(color_img, markers)

        ws_mask = np.zeros((h, w), dtype=np.uint8)
        for label in range(2, n_markers + 1):
            ws_mask[markers == label] = 255
        ws_mask[markers == -1] = 0

        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            ws_mask, connectivity=8)

        contours_list = []
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area or area / total_area > self.max_area_ratio:
                continue

            region = (labels == i).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            cnt = max(cnts, key=cv2.contourArea)

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity or circularity > self.max_circularity:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if hull_area > 0 and solidity < self.min_solidity:
                continue

            contours_list.append(cnt)
            cv2.drawContours(combined_mask, [cnt], -1, 255, -1)

        return contours_list, combined_mask

    def _detect_with_fastsam(self, img, model):
        """FastSAM辅助：暗度投票主检测 + FastSAM补充"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        h, w = img.shape[:2]

        main_contours, main_mask = self._detect_fallback(img)

        try:
            results = model(img, imgsz=self.imgsz, conf=self.conf,
                            iou=self.iou, device='cpu')
            if results[0].masks is None:
                return main_contours, main_mask

            masks = results[0].masks.data.cpu().numpy()
            fs_mask = np.zeros((h, w), dtype=np.uint8)
            for m in masks:
                m_resized = cv2.resize(m.astype(np.uint8), (w, h),
                                        interpolation=cv2.INTER_NEAREST)
                fs_mask = cv2.bitwise_or(fs_mask, m_resized)

            fs_contours = []
            if fs_mask.sum() > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fs_m = cv2.morphologyEx(fs_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                fs_m = cv2.morphologyEx(fs_m, cv2.MORPH_OPEN, kernel, iterations=1)
                total_area = h * w
                n, labels, stats, _ = cv2.connectedComponentsWithStats(fs_m, connectivity=8)
                for i in range(1, n):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area < self.min_area or area / total_area > self.max_area_ratio:
                        continue
                    region = (labels == i).astype(np.uint8) * 255
                    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        cnt = max(cnts, key=cv2.contourArea)
                        perim = cv2.arcLength(cnt, True)
                        if perim < 1:
                            continue
                        circ = 4 * np.pi * area / (perim * perim)
                        if circ < self.min_circularity or circ > self.max_circularity:
                            continue
                        hull = cv2.convexHull(cnt)
                        hull_a = cv2.contourArea(hull)
                        if hull_a > 0 and area / hull_a < self.min_solidity:
                            continue
                        fs_contours.append(cnt)

            all_contours = list(main_contours)
            combined_mask = main_mask.copy()
            for fs_cnt in fs_contours:
                fs_area = cv2.contourArea(fs_cnt)
                is_dup = False
                for main_cnt in main_contours:
                    fs_msk = np.zeros((h, w), dtype=np.uint8)
                    main_msk = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(fs_msk, [fs_cnt], -1, 255, -1)
                    cv2.drawContours(main_msk, [main_cnt], -1, 255, -1)
                    overlap = cv2.bitwise_and(fs_msk, main_msk)
                    if np.sum(overlap > 0) / max(fs_area, 1) > 0.3:
                        is_dup = True
                        break
                if not is_dup:
                    all_contours.append(fs_cnt)
                    cv2.drawContours(combined_mask, [fs_cnt], -1, 255, -1)

            return all_contours, combined_mask

        except Exception:
            return main_contours, main_mask
