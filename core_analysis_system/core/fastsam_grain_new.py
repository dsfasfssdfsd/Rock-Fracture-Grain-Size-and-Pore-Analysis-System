"""
基于FastSAM主检测的粒度检测器
核心算法：FastSAM分割为主（轮廓更贴边），DoG纹理检测回退
支持子进程调用（解决GUI中PyTorch DLL加载失败问题）
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


class FastSAMGrainDetectorNew:
    """粒度检测器：FastSAM主检测 + DoG回退"""

    def __init__(self,
                 min_area=200, max_area_ratio=0.25,
                 min_circularity=0.05, min_solidity=0.4,
                 conf=0.05, iou=0.5, imgsz=640,
                 dog_sigma1=7, dog_sigma2=21,
                 close_kernel=5, close_iter=2,
                 open_kernel=5, open_iter=1,
                 use_vote=True, vote_thresh=2):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_circularity = min_circularity
        self.min_solidity = min_solidity
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        # DoG参数（回退时用）
        self.dog_sigma1 = dog_sigma1
        self.dog_sigma2 = dog_sigma2
        self.close_kernel = close_kernel
        self.close_iter = close_iter
        self.open_kernel = open_kernel
        self.open_iter = open_iter
        self.use_vote = use_vote
        self.vote_thresh = vote_thresh
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
                model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'FastSAM-x.pt')
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
            raise RuntimeError("FastSAM模型加载失败")
        return self._model

    def detect(self, img):
        """检测粒度 - FastSAM主检测
        支持自适应图像尺寸：参数会根据图像大小自动缩放
        """
        h, w = img.shape[:2]
        base_area = 424 * 494
        cur_area = h * w
        scale_factor = (cur_area / base_area) ** 0.5

        orig_params = {
            'min_area': self.min_area,
            'dog_sigma1': self.dog_sigma1,
            'dog_sigma2': self.dog_sigma2,
            'close_kernel': self.close_kernel,
            'open_kernel': self.open_kernel,
        }

        try:
            if scale_factor != 1.0:
                self.min_area = max(50, int(self.min_area * scale_factor * scale_factor))
                self.dog_sigma1 = max(1.0, self.dog_sigma1 * scale_factor)
                self.dog_sigma2 = max(2.0, self.dog_sigma2 * scale_factor)
                self.close_kernel = max(3, int(self.close_kernel * scale_factor))
                if self.close_kernel % 2 == 0:
                    self.close_kernel += 1
                self.open_kernel = max(3, int(self.open_kernel * scale_factor))
                if self.open_kernel % 2 == 0:
                    self.open_kernel += 1

            model = self._load_model()
            if model is None:
                raise RuntimeError("FastSAM模型加载失败")
            return self._detect_fastsam_main(img, model)
        finally:
            self.min_area = orig_params['min_area']
            self.dog_sigma1 = orig_params['dog_sigma1']
            self.dog_sigma2 = orig_params['dog_sigma2']
            self.close_kernel = orig_params['close_kernel']
            self.open_kernel = orig_params['open_kernel']

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
            if circ < self.min_circularity:
                continue
            hull = cv2.convexHull(cnt)
            hull_a = cv2.contourArea(hull)
            solid = area / hull_a if hull_a > 0 else 0
            if hull_a > 0 and solid < self.min_solidity:
                continue
            keep.append(cnt)
        return keep

    def _detect_fastsam_main(self, img, model):
        """FastSAM检测 + 形状过滤
        model可以是FastSAM对象、'daemon'或'subprocess'字符串
        """
        h, w = img.shape[:2]

        if isinstance(model, str) and model == 'daemon':
            # 常驻子进程方式
            from core.fastsam_daemon import run_fastsam_daemon
            fastsam_contours = run_fastsam_daemon(
                img, conf=self.conf, iou=self.iou, imgsz=self.imgsz)
            if fastsam_contours is None:
                raise RuntimeError("FastSAM常驻子进程推理失败")
        elif isinstance(model, str) and model == 'subprocess':
            # 一次性子进程方式
            from core.fastsam_subprocess import run_fastsam_subprocess
            fastsam_contours = run_fastsam_subprocess(
                img, conf=self.conf, iou=self.iou, imgsz=self.imgsz)
            if fastsam_contours is None:
                raise RuntimeError("FastSAM子进程推理失败")
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

    def _get_dog_mask(self, gray):
        """高斯差分(DoG)检测纹理丰富区域"""
        h, w = gray.shape
        if self.use_vote:
            base_ratio = self.dog_sigma2 / self.dog_sigma1
            scale_factors = [0.5, 0.75, 1.0, 1.3]
            vote = np.zeros((h, w), dtype=np.uint8)
            for sf in scale_factors:
                s1 = self.dog_sigma1 * sf
                s2 = self.dog_sigma2 * sf
                g1 = cv2.GaussianBlur(gray, (0, 0), s1)
                g2 = cv2.GaussianBlur(gray, (0, 0), s2)
                dog = cv2.absdiff(g1, g2)
                _, th = cv2.threshold(dog, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                vote += (th > 0).astype(np.uint8)
            result = np.zeros((h, w), dtype=np.uint8)
            result[vote >= self.vote_thresh] = 255
            return result
        else:
            g1 = cv2.GaussianBlur(gray, (0, 0), self.dog_sigma1)
            g2 = cv2.GaussianBlur(gray, (0, 0), self.dog_sigma2)
            dog = cv2.absdiff(g1, g2)
            _, th = cv2.threshold(dog, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return th

    def _dog_post_process(self, img, binary):
        """DoG结果后处理：形态学 + 孔洞填充 + 连通域过滤"""
        h, w = img.shape[:2]

        kernel_c = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.close_kernel, self.close_kernel))
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_c,
                                iterations=self.close_iter)

        kernel_o = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.open_kernel, self.open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_o,
                                iterations=self.open_iter)

        contours_fill, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_fill:
            if cv2.contourArea(cnt) > 100:
                cv2.drawContours(mask, [cnt], -1, 255, -1)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)

        contours_list = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area or area / (h * w) > self.max_area_ratio:
                continue
            region = (labels == i).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            contours_list.append(cnt)

        return self._filter_contours(contours_list, h, w)

    def _detect_fallback(self, img):
        """回退：纯DoG方法（无FastSAM）"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        h, w = img.shape[:2]
        dog_mask = self._get_dog_mask(gray)
        contours = self._dog_post_process(img, dog_mask)
        combined = np.zeros((h, w), dtype=np.uint8)
        for cnt in contours:
            cv2.drawContours(combined, [cnt], -1, 255, -1)
        return contours, combined
