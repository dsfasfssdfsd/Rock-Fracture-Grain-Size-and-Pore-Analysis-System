from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QLabel,
                             QSpinBox, QDoubleSpinBox, QFormLayout,
                             QCheckBox, QSlider, QHBoxLayout, QComboBox,
                             QPushButton)
from PyQt5.QtCore import Qt, pyqtSignal


class ParamPanel(QWidget):
    params_changed = pyqtSignal(dict)
    preprocess_applied = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "fracture"
        self._pp_controls = {}
        self._setup_ui()

    def _setup_ui(self):
        self.setMinimumWidth(250)
        self.setMaximumWidth(350)
        layout = QVBoxLayout(self)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["裂缝分析"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        # 只剩一个模式，隐藏下拉框
        self._mode_combo.setVisible(False)
        mode_label = QLabel("分析模式:")
        mode_label.setVisible(False)
        layout.addWidget(mode_label)
        layout.addWidget(self._mode_combo)

        self._group_fracture = QGroupBox("裂缝分析参数")
        self._group_pore = QGroupBox("孔洞分析参数")
        self._group_grain = QGroupBox("粒度分析参数")

        for g in (self._group_fracture, self._group_pore, self._group_grain):
            g.setLayout(QVBoxLayout())

        self._init_preprocess_ui(self._group_fracture)
        self._init_fracture_ui()
        self._init_preprocess_ui(self._group_pore)
        self._init_pore_ui()
        self._init_preprocess_ui(self._group_grain)
        self._init_grain_ui()

        layout.addWidget(self._group_fracture)
        layout.addWidget(self._group_pore)
        layout.addWidget(self._group_grain)
        layout.addStretch()

        self._update_visible()

    # ---------- 预处理控件 ----------

    def _init_preprocess_ui(self, group):
        pp_box = QGroupBox("预处理")
        pp_box.setObjectName("subGroup")
        form = QFormLayout(pp_box)

        # 1. 灰度化处理
        grayscale_enabled = QCheckBox("启用")
        grayscale_enabled.setChecked(False)
        form.addRow("灰度化处理:", grayscale_enabled)

        # 2. 对比度/亮度调节
        contrast_enabled = QCheckBox("启用")
        contrast_enabled.setChecked(False)
        form.addRow("对比度调节:", contrast_enabled)

        contrast_alpha = QDoubleSpinBox()
        contrast_alpha.setRange(0.5, 3.0)
        contrast_alpha.setValue(1.0)
        contrast_alpha.setSingleStep(0.1)
        form.addRow("  对比度:", contrast_alpha)

        contrast_beta = QSpinBox()
        contrast_beta.setRange(-100, 100)
        contrast_beta.setValue(0)
        contrast_beta.setSingleStep(5)
        form.addRow("  亮度:", contrast_beta)

        # 3. 高斯降噪
        gaussian_enabled = QCheckBox("启用")
        gaussian_enabled.setChecked(False)
        form.addRow("高斯降噪:", gaussian_enabled)

        gaussian_kernel = QSpinBox()
        gaussian_kernel.setRange(3, 15)
        gaussian_kernel.setValue(3)
        gaussian_kernel.setSingleStep(2)
        form.addRow("  核大小:", gaussian_kernel)

        gaussian_sigma = QDoubleSpinBox()
        gaussian_sigma.setRange(0.1, 5.0)
        gaussian_sigma.setValue(0.5)
        gaussian_sigma.setSingleStep(0.1)
        form.addRow("  Sigma:", gaussian_sigma)

        # 4. 按钮
        btn_layout = QHBoxLayout()
        btn_apply = QPushButton("应用参数")
        btn_apply.setObjectName("btn_small")
        btn_apply.clicked.connect(lambda: self._emit_apply())
        btn_reset = QPushButton("重置参数")
        btn_reset.setObjectName("btn_small")
        btn_reset.clicked.connect(lambda: self._reset_preprocess(group))
        btn_layout.addWidget(btn_apply)
        btn_layout.addWidget(btn_reset)
        form.addRow(btn_layout)

        group.layout().addWidget(pp_box)
        self._pp_controls[group] = {
            "grayscale_enabled": grayscale_enabled,
            "contrast_enabled": contrast_enabled,
            "contrast_alpha": contrast_alpha,
            "contrast_beta": contrast_beta,
            "gaussian_enabled": gaussian_enabled,
            "gaussian_kernel": gaussian_kernel,
            "gaussian_sigma": gaussian_sigma,
        }

    # ---------- 裂缝分析 ----------

    def _init_fracture_ui(self):
        box = QGroupBox("分析参数")
        box.setObjectName("subGroup")
        form = QFormLayout(box)

        tip_label = QLabel("自动分析使用UNet深度学习模型\n手动分析使用传统多尺度投票")
        tip_label.setWordWrap(True)
        tip_label.setStyleSheet("color: #909399; font-size: 12px;")
        form.addRow(tip_label)

        # 一键预设按钮
        btn_preset = QPushButton("一键预处理（推荐）")
        btn_preset.setObjectName("btn_primary")
        btn_preset.clicked.connect(self._apply_fracture_preset)
        form.addRow(btn_preset)

        self._group_fracture.layout().addWidget(box)

    # ---------- 孔洞分析 ----------

    def _init_pore_ui(self):
        box = QGroupBox("分析参数")
        form = QFormLayout(box)

        self._pore_size_scale = QDoubleSpinBox()
        self._pore_size_scale.setRange(0.3, 3.0)
        self._pore_size_scale.setValue(1.0)
        self._pore_size_scale.setSingleStep(0.1)
        self._pore_size_scale.valueChanged.connect(self._emit_params)

        form.addRow("孔洞大小调节:", self._pore_size_scale)

        tip_label = QLabel("值越小识别越小的孔洞\n值越大识别越大的孔洞")
        tip_label.setWordWrap(True)
        tip_label.setStyleSheet("color: #909399; font-size: 12px;")
        form.addRow(tip_label)

        self._group_pore.layout().addWidget(box)

    # ---------- 粒度分析 ----------

    def _init_grain_ui(self):
        box = QGroupBox("分析参数")
        form = QFormLayout(box)

        self._grain_sigma = QDoubleSpinBox()
        self._grain_sigma.setRange(0.5, 5.0)
        self._grain_sigma.setValue(1.0)
        self._grain_sigma.setSingleStep(0.5)
        self._grain_sigma.valueChanged.connect(self._emit_params)

        form.addRow("高通滤波 Sigma:", self._grain_sigma)
        self._group_grain.layout().addWidget(box)

    # ---------- 公共 ----------

    def _on_mode_changed(self, text):
        self._mode = {"裂缝分析": "fracture", "孔洞分析": "pore", "粒度分析": "grain"}.get(text, "fracture")
        self._update_visible()
        self._emit_params()

    def _update_visible(self):
        self._group_fracture.setVisible(self._mode == "fracture")
        self._group_pore.setVisible(self._mode == "pore")
        self._group_grain.setVisible(self._mode == "grain")

    def _emit_params(self):
        self.params_changed.emit(self.get_params())

    def _emit_apply(self):
        self.preprocess_applied.emit(self.get_preprocess_params())

    def _reset_preprocess(self, group):
        ctrl = self._pp_controls.get(group)
        if not ctrl:
            return
        ctrl["grayscale_enabled"].setChecked(False)
        ctrl["contrast_enabled"].setChecked(False)
        ctrl["contrast_alpha"].setValue(1.0)
        ctrl["contrast_beta"].setValue(0)
        ctrl["gaussian_enabled"].setChecked(False)
        ctrl["gaussian_kernel"].setValue(3)
        ctrl["gaussian_sigma"].setValue(0.5)
        self._emit_apply()

    def _apply_fracture_preset(self):
        """裂缝分析一键预处理预设"""
        ctrl = self._pp_controls.get(self._group_fracture)
        if not ctrl:
            return
        ctrl["grayscale_enabled"].setChecked(True)
        ctrl["contrast_enabled"].setChecked(True)
        ctrl["contrast_alpha"].setValue(1.90)
        ctrl["contrast_beta"].setValue(60)
        ctrl["gaussian_enabled"].setChecked(True)
        ctrl["gaussian_kernel"].setValue(11)
        ctrl["gaussian_sigma"].setValue(2.30)
        self._emit_apply()

    def _mode_group(self):
        return {"fracture": self._group_fracture,
                "pore": self._group_pore,
                "grain": self._group_grain}.get(self._mode)

    def get_preprocess_params(self):
        ctrl = self._pp_controls.get(self._mode_group(), {})
        if not ctrl:
            return {}
        return {
            "grayscale_enabled": ctrl["grayscale_enabled"].isChecked(),
            "contrast_enabled": ctrl["contrast_enabled"].isChecked(),
            "contrast_alpha": ctrl["contrast_alpha"].value(),
            "contrast_beta": ctrl["contrast_beta"].value(),
            "gaussian_enabled": ctrl["gaussian_enabled"].isChecked(),
            "gaussian_kernel": ctrl["gaussian_kernel"].value(),
            "gaussian_sigma": ctrl["gaussian_sigma"].value(),
        }

    def get_params(self):
        params = {"mode": self._mode, "preprocess": self.get_preprocess_params()}
        if self._mode == "pore":
            params.update(pore_size_scale=self._pore_size_scale.value())
        elif self._mode == "grain":
            params.update(sigma=self._grain_sigma.value())
        return params

    @property
    def mode(self):
        return self._mode
