import os, sys, tempfile
import subprocess
import json
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QSplitter, QStatusBar, QLabel, QMessageBox,
                             QFileDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication
from ui.image_viewer import ImageViewer
from ui.param_panel import ParamPanel
from ui.toolbar import ToolBar
from ui.report_dialog import ReportDialog
from core.image_manager import ImageManager
from core.preprocessing import Preprocessor
from core.fracture import FractureDetector
from core.pore import PoreDetector
from core.grain import GrainDetector
from core.dl_fracture import DLFractureDetector
from core.unet_inference import UNetInference

STYLESHEET = """
QMainWindow {
    background-color: #f0f2f5;
}
QToolBar {
    background-color: #ffffff;
    border-bottom: 1px solid #e4e7ed;
    spacing: 10px;
    padding: 6px 16px;
}
QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 6px 16px;
    color: #303133;
    font-size: 13px;
    min-height: 24px;
}
QToolButton:hover {
    background-color: #f0f2f5;
    border-color: #e4e7ed;
}
QToolButton:pressed {
    background-color: #e8eaed;
}
QToolBar::separator {
    width: 1px;
    margin: 6px 4px;
    background-color: #e4e7ed;
}
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 8px;
    margin-top: 14px;
    padding: 20px 14px 14px 14px;
    font-weight: 600;
    font-size: 13px;
    color: #303133;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 12px;
    color: #409eff;
}
QGroupBox#subGroup {
    border: none;
    border-left: 3px solid #409eff;
    border-radius: 0;
    margin-top: 6px;
    padding: 8px 8px 4px 8px;
    background: #fafbfc;
}
QGroupBox#subGroup::title {
    color: #606266;
    font-weight: 500;
    font-size: 12px;
    padding: 0 8px;
}
QLabel {
    color: #606266;
    font-size: 13px;
}
QDoubleSpinBox, QSpinBox {
    border: 1px solid #dcdfe6;
    border-radius: 4px;
    padding: 4px 8px;
    background: #ffffff;
    min-height: 22px;
    font-size: 13px;
}
QDoubleSpinBox:focus, QSpinBox:focus {
    border-color: #409eff;
}
QComboBox {
    border: 1px solid #dcdfe6;
    border-radius: 6px;
    padding: 6px 12px;
    background: #ffffff;
    min-height: 28px;
    font-size: 13px;
}
QComboBox:focus { border-color: #409eff; }
QComboBox::drop-down {
    width: 28px;
    border: none;
}
QComboBox::down-arrow {
    image: none;
    width: 0;
}
QComboBox QAbstractItemView {
    border: 1px solid #e4e7ed;
    border-radius: 4px;
    selection-background-color: #ecf5ff;
    selection-color: #409eff;
    outline: none;
}
QCheckBox {
    spacing: 8px;
    font-size: 13px;
    color: #606266;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #e4e7ed;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    background: #409eff;
    border-radius: 8px;
    margin: -6px 0;
}
QSlider::handle:horizontal:hover {
    background: #66b1ff;
}
QScrollBar:vertical {
    width: 6px;
    background: transparent;
}
QScrollBar::handle:vertical {
    background: #c0c4cc;
    border-radius: 3px;
    min-height: 30px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QStatusBar {
    background: #ffffff;
    border-top: 1px solid #e4e7ed;
    color: #909399;
    font-size: 12px;
    min-height: 20px;
}
QGraphicsView {
    border: 1px solid #e8eaed;
    border-radius: 8px;
    background: #f5f7fa;
}
QPushButton {
    background-color: #409eff;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    font-size: 13px;
    min-height: 32px;
}
QPushButton:hover {
    background-color: #66b1ff;
}
QPushButton:pressed {
    background-color: #3a8ee6;
}
QPushButton#btn_secondary {
    background-color: #ffffff;
    color: #606266;
    border: 1px solid #dcdfe6;
}
QPushButton#btn_secondary:hover {
    background-color: #f0f2f5;
}
QPushButton#btn_small {
    background-color: #f5f7fa;
    color: #409eff;
    border: 1px solid #d9ecff;
    border-radius: 4px;
    padding: 4px 12px;
    font-size: 12px;
    min-height: 24px;
}
QPushButton#btn_small:hover {
    background-color: #ecf5ff;
    border-color: #409eff;
}
QPushButton#btn_small:pressed {
    background-color: #d9ecff;
}
QTextEdit {
    border: 1px solid #e8eaed;
    border-radius: 8px;
    background: #ffffff;
    padding: 12px;
    font-size: 13px;
    color: #303133;
}
QSplitter::handle {
    width: 1px;
    background: #e4e7ed;
}
QWidget#leftPanel {
    background-color: #f5f7fa;
    border-right: 1px solid #e4e7ed;
    padding: 8px;
}
"""

FastSAMGrainDetectorNew = None
FastSAMPoreDetector = None
try:
    from core.fastsam_detector import FastSAMPoreDetector
    from core.fastsam_grain_new import FastSAMGrainDetectorNew
except Exception:
    pass


class MainWindow(QMainWindow):
    COLOR_MAP = {
        "fracture": (0, 0, 255),
        "pore": (0, 255, 0),
        "grain": (255, 0, 0),
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("岩石裂缝粒度孔洞分析系统")
        self.resize(1400, 900)

        self._img_mgr = ImageManager()
        self._preprocessor = Preprocessor()
        self._fracture_detector = FractureDetector()
        self._pore_detector = PoreDetector()
        self._grain_detector = GrainDetector()
        self._dl_detector = DLFractureDetector(
            model_dir=os.path.join(os.path.dirname(__file__), '..', 'core'))
        self._unet_detector = UNetInference()
        self._fastsam_grain = FastSAMGrainDetectorNew() if FastSAMGrainDetectorNew else None
        self._fastsam_pore = FastSAMPoreDetector() if FastSAMPoreDetector else None
        self._dl_loaded = False
        self._unet_loaded = False
        self._current_mask = None
        self._analysis_result = None
        self._current_display = None

        self._load_dl_model()
        self._setup_ui()
        self._connect_signals()
        self._setup_style()

    def _load_dl_model(self):
        try:
            self._dl_detector.load()
            self._dl_loaded = True
        except Exception:
            self._dl_loaded = False
        self._unet_loaded = self._ensure_unet_loaded()

    def _ensure_unet_loaded(self):
        """确保UNet常驻子进程已启动"""
        try:
            from core.unet_daemon import get_unet_daemon
            daemon = get_unet_daemon()
            return daemon._ensure_started()
        except Exception:
            return False

    def _run_unet_subprocess(self, img_bgr):
        """使用常驻子进程运行UNet"""
        try:
            from core.unet_daemon import run_unet_daemon
            mask = run_unet_daemon(img_bgr, threshold=0.3, tile_size=384)
            if mask is not None:
                return mask
        except Exception:
            pass
        # 回退到一次性子进程
        script = os.path.join(os.path.dirname(__file__), '..', 'core', 'unet_infer_subprocess.py')
        tmp_img = os.path.join(tempfile.gettempdir(), 'unet_input.png')
        tmp_out = os.path.join(tempfile.gettempdir(), 'unet_output.png')
        cv2.imwrite(tmp_img, img_bgr)
        subprocess.run([sys.executable, script, tmp_img, tmp_out, '0.3'],
                       check=True, capture_output=True, timeout=120)
        mask = cv2.imread(tmp_out, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        return mask

    def _setup_ui(self):
        self._toolbar = ToolBar(self)
        self.addToolBar(self._toolbar)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        self._param_panel = ParamPanel()

        left_widget = QWidget()
        left_widget.setObjectName("leftPanel")
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(0)
        left_layout.addWidget(self._param_panel)
        left_layout.addStretch()

        self._viewer = ImageViewer()

        splitter.addWidget(left_widget)
        splitter.addWidget(self._viewer)
        splitter.setSizes([300, 1100])
        main_layout.addWidget(splitter)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_label = QLabel("就绪")
        self._status.addPermanentWidget(self._status_label)

    def _connect_signals(self):
        self._toolbar.act_open.triggered.connect(self._open_image)
        self._toolbar.act_save.triggered.connect(self._save_result)
        self._toolbar.act_analyze.triggered.connect(self._analyze)
        self._toolbar.act_auto_fracture.triggered.connect(self._auto_analyze_fracture)
        self._toolbar.act_auto_pore.triggered.connect(self._auto_analyze_pore)
        self._toolbar.act_auto_grain.triggered.connect(self._auto_analyze_grain)
        self._toolbar.act_reset.triggered.connect(self._reset)
        self._toolbar.act_report.triggered.connect(self._show_report)
        self._param_panel.params_changed.connect(self._on_params_changed)
        self._param_panel.preprocess_applied.connect(self._on_preprocess_applied)

    # ---------- 样式 ----------

    def _setup_style(self):
        QApplication.instance().setStyleSheet(STYLESHEET)
        font = QFont("Microsoft YaHei UI", 9)
        QApplication.instance().setFont(font)

    # ---------- 图像加载 ----------

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图像", "",
            "图像文件 (*.bmp *.jpg *.jpeg *.png *.tif *.tiff)")
        if not path:
            return
        try:
            self._img_mgr.load_image(path)
            # 统一缩放到1000x1000
            resized = cv2.resize(self._img_mgr.original, (1000, 1000),
                                 interpolation=cv2.INTER_AREA)
            self._img_mgr._original = resized
            self._img_mgr._processed = resized.copy()
            self._current_display = self._img_mgr._processed.copy()
            self._viewer.set_image(self._current_display)
            self._current_mask = None
            self._analysis_result = None
            self._last_preprocess_params = None
            h, w = self._img_mgr.original.shape[:2]
            self._status_label.setText(f"已加载: {os.path.basename(path)} ({w}x{h})")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法加载图像:\n{e}")

    # ---------- 保存 ----------

    def _save_result(self):
        img = self._viewer.get_scene_image()
        if img is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存图像", "", "PNG (*.png);;JPEG (*.jpg)")
        if path:
            ext = os.path.splitext(path)[1]
            if not ext:
                ext = '.png'
                path += ext
            success, buf = cv2.imencode(ext, img)
            if success:
                buf.tofile(path)
                self._status_label.setText(f"已保存: {os.path.basename(path)}")
            else:
                QMessageBox.warning(self, "错误", "保存失败")

    # ---------- 手动分析（带预处理） ----------

    def _apply_preprocess(self, img, pp):
        # 灰度化处理
        if pp.get("grayscale_enabled"):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if pp.get("contrast_enabled"):
            alpha = pp.get("contrast_alpha", 1.0)
            beta = pp.get("contrast_beta", 0)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        if pp.get("gaussian_enabled"):
            k = pp.get("gaussian_kernel", 3)
            if k % 2 == 0:
                k += 1
            sigma = pp.get("gaussian_sigma", 0.5)
            img = cv2.GaussianBlur(img, (k, k), sigma)

        return img

    @staticmethod
    def _morph_skeleton(binary):
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skel = np.zeros_like(binary)
        while True:
            eroded = cv2.erode(binary, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(binary, temp)
            skel = cv2.bitwise_or(skel, temp)
            binary = eroded
            if cv2.countNonZero(binary) == 0:
                break
        return skel

    def _adjust_brightness_contrast(self, img, brightness=0, contrast=1.0):
        if contrast != 1.0:
            img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
        if brightness != 0:
            img = cv2.convertScaleAbs(img, alpha=1.0, beta=brightness)
        return img

    def _analyze(self):
        if self._img_mgr.original is None:
            QMessageBox.warning(self, "提示", "请先打开图像")
            return

        mode = self._param_panel.mode
        params = self._param_panel.get_params()
        pp = params.get("preprocess", {})

        try:
            base = self._img_mgr.original.copy()
            processed = self._apply_preprocess(base.copy(), pp)

            if mode == "fracture":
                method_label = "传统多尺度投票"
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0.5)
                mask = self._dl_detector._classical_predict(gray)

                h, w = processed.shape[:2]
                margin = max(2, int(min(h, w) * 0.01))
                edges = np.zeros((h, w), dtype=np.uint8)
                edges[margin:-margin, margin:-margin] = 255
                mask = cv2.bitwise_and(mask, mask, mask=edges)

                self._current_mask = mask
                display = base.copy()
                display[mask > 0] = self.COLOR_MAP["fracture"]
                self._current_display = display
                self._viewer.set_image(display)

                # 统计裂缝条数（连通域数量）
                n, _ = cv2.connectedComponents(mask)
                frac_count = max(0, n - 1)  # 减去背景

                self._analysis_result = {"type": "fracture", "count": frac_count}
                self._status_label.setText(f"裂缝分析完成 | 识别到 {frac_count} 条裂缝（{method_label}）")

            elif mode == "pore":
                pore_size_scale = params.get("pore_size_scale", 1.0)
                # 使用原图进行孔洞检测（不受预处理影响）
                contours = self._pore_detector.detect_auto(base, pore_size_scale=pore_size_scale)
                mask = np.zeros(base.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, contours, -1, 255, -1)
                self._current_mask = mask

                display = base.copy()
                color = self.COLOR_MAP["pore"]
                for cnt in contours:
                    cv2.drawContours(display, [cnt], -1, color, 2)
                self._current_display = display
                self._viewer.set_image(display)

                self._analysis_result = {"type": "pore", "count": len(contours)}
                self._status_label.setText(f"孔洞分析完成 | 检测到 {len(contours)} 个孔洞")

            elif mode == "grain":
                if self._fastsam_grain is None:
                    QMessageBox.warning(self, "提示", "检测器未初始化")
                    return
                contours, mask = self._fastsam_grain.detect(processed)
                self._current_mask = mask

                overlay = self._make_overlay(mask, self.COLOR_MAP["grain"])
                display = self._blend_overlay(base, overlay, 0.4)
                self._current_display = display
                self._viewer.set_image(display)

                self._analysis_result = {"type": "grain", "count": len(contours)}
                self._status_label.setText(f"粒度分析完成 | 检测到 {len(contours)} 个颗粒")

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "分析错误", f"{str(e)}\n{traceback.format_exc()}")

    # ---------- 自动分析 ----------

    def _auto_analyze_fracture(self):
        if self._img_mgr.original is None:
            QMessageBox.warning(self, "提示", "请先打开图像")
            return

        self._status_label.setText("自动裂缝分析中(UNet)...")
        QApplication.processEvents()

        try:
            # 检测用原图（不使用预处理参数）
            img_orig = self._img_mgr.original.copy()
            h_o, w_o = img_orig.shape[:2]

            method_label = "深度学习UNet"

            # 确保UNet已加载
            if not self._unet_loaded:
                self._unet_loaded = self._ensure_unet_loaded()

            if self._unet_loaded:
                mask = self._run_unet_subprocess(img_orig)
            else:
                # UNet加载失败，回退到传统方法
                method_label = "传统多尺度投票(UNet未加载)"
                gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0.5)
                mask = self._dl_detector._classical_predict(gray)

            # 边缘裁剪（基于原图）
            margin = max(2, int(min(h_o, w_o) * 0.01))
            edges = np.zeros((h_o, w_o), dtype=np.uint8)
            edges[margin:-margin, margin:-margin] = 255
            mask = cv2.bitwise_and(mask, mask, mask=edges)

            # 显示用原图（不使用预处理后的图）
            display = img_orig.copy()

            self._current_mask = mask

            display[mask > 0] = [0, 0, 255]

            self._current_display = display
            self._viewer.set_image(display)
            QApplication.processEvents()

            # 统计裂缝条数（连通域数量）
            n, _ = cv2.connectedComponents(mask)
            frac_count = max(0, n - 1)  # 减去背景

            self._analysis_result = {"type": "fracture", "count": frac_count}
            self._status_label.setText(f"{method_label}自动裂缝分析完成 | 识别到 {frac_count} 条裂缝")
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "自动裂缝分析错误", f"{str(e)}\n{traceback.format_exc()}")

    def _auto_analyze_pore(self):
        if self._img_mgr.original is None:
            QMessageBox.warning(self, "提示", "请先打开图像")
            return
        if self._fastsam_pore is None:
            QMessageBox.warning(self, "提示", "检测器未初始化")
            return

        self._status_label.setText("自动孔洞分析中...")
        QApplication.processEvents()

        try:
            # 检测用原图
            img_orig = self._img_mgr.original.copy()
            h_o, w_o = img_orig.shape[:2]
            contours, mask = self._fastsam_pore.detect(img_orig)

            # 显示用缩放图(1033x1200)
            display = self._img_mgr._processed.copy()
            h_d, w_d = display.shape[:2]

            # 轮廓坐标缩放到显示尺寸
            sx = w_d / w_o
            sy = h_d / h_o
            scaled_contours = []
            for cnt in contours:
                scaled_cnt = cnt * np.array([sx, sy])
                scaled_contours.append(scaled_cnt.astype(np.int32))

            self._current_mask = mask

            if len(scaled_contours) == 0:
                self._current_display = display.copy()
                self._viewer.set_image(display)
                self._analysis_result = {"type": "pore", "count": 0}
                self._status_label.setText("自动孔洞分析完成 | 未检测到孔洞")
                return

            color = self.COLOR_MAP["pore"]
            for cnt in scaled_contours:
                cv2.drawContours(display, [cnt], -1, color, 2)

            self._current_display = display
            self._viewer.set_image(display)
            QApplication.processEvents()

            self._analysis_result = {"type": "pore", "count": len(contours), "contours": contours}
            self._status_label.setText(f"自动孔洞分析完成 | 检测到 {len(contours)} 个孔洞")
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "自动孔洞分析错误", f"{str(e)}\n{traceback.format_exc()}")

    def _auto_analyze_grain(self):
        if self._img_mgr.original is None:
            QMessageBox.warning(self, "提示", "请先打开图像")
            return
        if self._fastsam_grain is None:
            QMessageBox.warning(self, "提示", "检测器未初始化")
            return

        self._status_label.setText("自动粒度分析中...")
        QApplication.processEvents()

        try:
            # 检测用原图
            img_orig = self._img_mgr.original.copy()
            h_o, w_o = img_orig.shape[:2]
            contours, mask = self._fastsam_grain.detect(img_orig)

            # 显示用缩放图(1033x1200)
            display = self._img_mgr._processed.copy()
            h_d, w_d = display.shape[:2]

            # 轮廓坐标缩放到显示尺寸
            sx = w_d / w_o
            sy = h_d / h_o
            scaled_contours = []
            for cnt in contours:
                scaled_cnt = cnt * np.array([sx, sy])
                scaled_contours.append(scaled_cnt.astype(np.int32))

            self._current_mask = mask

            if len(scaled_contours) == 0:
                self._current_display = display.copy()
                self._viewer.set_image(display)
                self._analysis_result = {"type": "grain", "count": 0}
                self._status_label.setText("自动粒度分析完成 | 未检测到颗粒")
                return

            color = (255, 0, 0)
            for cnt in scaled_contours:
                cv2.drawContours(display, [cnt], -1, color, 3)

            self._current_display = display
            self._viewer.set_image(display)
            QApplication.processEvents()

            self._analysis_result = {"type": "grain", "count": len(contours), "contours": contours}
            self._status_label.setText(f"自动粒度分析完成 | 检测到 {len(contours)} 个颗粒")
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "自动粒度分析错误", f"{str(e)}\n{traceback.format_exc()}")

    # ---------- 工具 ----------

    def _make_overlay(self, mask, color_bgr):
        overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
        overlay[mask > 0] = color_bgr
        return overlay

    def _blend_overlay(self, img, overlay, alpha=0.4):
        mask = np.any(overlay > 0, axis=2)
        result = img.copy().astype(np.float32)
        for c in range(3):
            result[mask, c] = result[mask, c] * (1 - alpha) + overlay[mask, c].astype(np.float32) * alpha
        return result.astype(np.uint8)

    def _on_params_changed(self, params):
        if params.get('mode') != getattr(self, '_prev_mode', None):
            self._prev_mode = params.get('mode')

    def _on_preprocess_applied(self, pp):
        if self._img_mgr.original is None:
            return
        processed = self._apply_preprocess(self._img_mgr.original.copy(), pp)
        self._img_mgr.processed = processed
        self._current_display = processed.copy()
        self._current_mask = None
        self._viewer.clear_overlays()
        self._viewer.set_image(self._current_display)
        self._status_label.setText("预处理已应用")

    def _reset(self):
        self._img_mgr.reset()
        self._current_display = self._img_mgr.original.copy() if self._img_mgr.original is not None else None
        self._viewer.set_image(self._current_display)
        self._current_mask = None
        self._analysis_result = None
        self._status_label.setText("已重置")

    def _show_report(self):
        if self._img_mgr.original is None:
            QMessageBox.warning(self, "提示", "请先打开图像并完成分析")
            return
        data = {
            "文件名": os.path.basename(self._img_mgr.filename) if self._img_mgr.filename else "",
            "图像尺寸": f"{self._img_mgr.original.shape[1]}x{self._img_mgr.original.shape[0]}",
            "分析模式": self._param_panel.mode,
        }
        if self._analysis_result:
            data["分析结果"] = f"{self._analysis_result['count']}"
        dlg = ReportDialog(data, self)
        dlg.exec_()
