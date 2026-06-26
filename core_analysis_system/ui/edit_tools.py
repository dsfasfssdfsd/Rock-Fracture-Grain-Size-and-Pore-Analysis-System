from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QPushButton)
from PyQt5.QtCore import pyqtSignal


class EditToolsPanel(QWidget):
    tool_selected = pyqtSignal(str)
    apply_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        group = QGroupBox("编辑工具")
        vbox = QVBoxLayout(group)

        self._btn_eraser = QPushButton("橡皮擦")
        self._btn_eraser.setCheckable(True)
        self._btn_eraser.clicked.connect(lambda: self._select_tool("eraser"))
        vbox.addWidget(self._btn_eraser)

        self._btn_dilate = QPushButton("膨胀")
        self._btn_dilate.clicked.connect(lambda: self._apply("dilate"))
        vbox.addWidget(self._btn_dilate)

        self._btn_erode = QPushButton("腐蚀")
        self._btn_erode.clicked.connect(lambda: self._apply("erode"))
        vbox.addWidget(self._btn_erode)

        self._btn_denoise = QPushButton("去噪 (面积<10px)")
        self._btn_denoise.clicked.connect(lambda: self._apply("denoise"))
        vbox.addWidget(self._btn_denoise)

        self._btn_fill = QPushButton("孔洞填充")
        self._btn_fill.clicked.connect(lambda: self._apply("fill_holes"))
        vbox.addWidget(self._btn_fill)

        layout.addWidget(group)

    def _select_tool(self, tool):
        self._btn_eraser.setChecked(tool == "eraser")
        self.tool_selected.emit(tool)

    def _apply(self, op):
        self.apply_requested.emit(op, {})
