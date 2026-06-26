from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QDoubleSpinBox, QSpinBox, QPushButton,
                             QCheckBox, QFormLayout, QSlider)
from PyQt5.QtCore import Qt


class PreprocessingDialog(QDialog):
    def __init__(self, parent=None, last_params=None):
        super().__init__(parent)
        self.setWindowTitle("图像预处理")
        self.setModal(True)
        self.resize(350, 400)
        self._setup_ui()
        if last_params:
            self.set_params(last_params)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        gray_group = QGroupBox("灰度转换")
        gray_form = QFormLayout(gray_group)
        self._to_gray = QCheckBox("转为灰度图")
        self._to_gray.setChecked(False)
        gray_form.addRow(self._to_gray)
        layout.addWidget(gray_group)

        adj_group = QGroupBox("亮度 / 对比度调节")
        adj_form = QFormLayout(adj_group)

        self._brightness = QSlider(Qt.Horizontal)
        self._brightness.setRange(-100, 100)
        self._brightness.setValue(0)
        self._brightness_label = QLabel("0")
        self._brightness.valueChanged.connect(lambda v: self._brightness_label.setText(str(v)))
        bh = QHBoxLayout()
        bh.addWidget(self._brightness)
        bh.addWidget(self._brightness_label)
        adj_form.addRow("亮度:", bh)

        self._contrast = QSlider(Qt.Horizontal)
        self._contrast.setRange(0, 300)
        self._contrast.setValue(100)
        self._contrast_label = QLabel("1.0")
        self._contrast.valueChanged.connect(lambda v: self._contrast_label.setText(f"{v/100:.1f}"))
        ch = QHBoxLayout()
        ch.addWidget(self._contrast)
        ch.addWidget(self._contrast_label)
        adj_form.addRow("对比度:", ch)

        layout.addWidget(adj_group)

        filter_group = QGroupBox("滤波增强")
        filter_form = QFormLayout(filter_group)

        self._auto_levels = QCheckBox("自动色阶")
        self._auto_levels.setChecked(True)
        filter_form.addRow(self._auto_levels)

        self._gaussian_sigma = QDoubleSpinBox()
        self._gaussian_sigma.setRange(0.0, 5.0)
        self._gaussian_sigma.setValue(1.5)
        self._gaussian_sigma.setSingleStep(0.5)
        self._gaussian_sigma.setSpecialValueText("关闭")
        filter_form.addRow("高斯滤波 Sigma:", self._gaussian_sigma)

        self._clahe = QCheckBox("CLAHE 增强")
        self._clahe.setChecked(False)
        filter_form.addRow(self._clahe)

        self._clahe_clip = QDoubleSpinBox()
        self._clahe_clip.setRange(0.5, 10.0)
        self._clahe_clip.setValue(2.0)
        filter_form.addRow("CLAHE 限幅:", self._clahe_clip)

        layout.addWidget(filter_group)

        btn_layout = QHBoxLayout()
        self._btn_reset = QPushButton("重置")
        self._btn_reset.clicked.connect(self._reset_defaults)
        self._btn_ok = QPushButton("应用")
        self._btn_ok.clicked.connect(self.accept)
        self._btn_cancel = QPushButton("取消")
        self._btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self._btn_reset)
        btn_layout.addStretch()
        btn_layout.addWidget(self._btn_ok)
        btn_layout.addWidget(self._btn_cancel)
        layout.addLayout(btn_layout)

    def _reset_defaults(self):
        self.set_params({})

    def set_params(self, params):
        self._to_gray.setChecked(params.get("to_gray", False))
        self._brightness.setValue(params.get("brightness", 0))
        self._contrast.setValue(int(params.get("contrast", 1.0) * 100))
        self._auto_levels.setChecked(params.get("auto_levels", True))
        self._gaussian_sigma.setValue(params.get("gaussian_sigma", 1.5))
        self._clahe.setChecked(params.get("clahe", False))
        self._clahe_clip.setValue(params.get("clahe_clip", 2.0))

    def get_params(self):
        return {
            "to_gray": self._to_gray.isChecked(),
            "brightness": self._brightness.value(),
            "contrast": self._contrast.value() / 100.0,
            "auto_levels": self._auto_levels.isChecked(),
            "gaussian_sigma": self._gaussian_sigma.value(),
            "clahe": self._clahe.isChecked(),
            "clahe_clip": self._clahe_clip.value(),
        }
