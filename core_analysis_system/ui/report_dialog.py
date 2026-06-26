from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QPushButton, QFileDialog, QMessageBox)
from core.report import ReportGenerator


class ReportDialog(QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("分析报告预览")
        self.resize(600, 500)
        self._data = data
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._preview_content()
        layout.addWidget(self._text)
        btn_layout = QHBoxLayout()
        self._btn_export = QPushButton("导出 Word 报告")
        self._btn_export.clicked.connect(self._export)
        self._btn_close = QPushButton("关闭")
        self._btn_close.setObjectName("btn_secondary")
        self._btn_close.clicked.connect(self.close)
        btn_layout.addWidget(self._btn_export)
        btn_layout.addWidget(self._btn_close)
        layout.addLayout(btn_layout)

    def _preview_content(self):
        text = "=== 分析报告 ===\n\n"
        for key, val in self._data.items():
            if isinstance(val, list):
                text += f"\n{key}:\n"
                for item in val:
                    text += f"  {item}\n"
            else:
                text += f"{key}: {val}\n"
        self._text.setText(text)

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存报告", "", "Word (*.docx)")
        if not path:
            return
        report = ReportGenerator()
        for key, val in self._data.items():
            report.add_paragraph(f"{key}: {val}")
        report.save(path)
        QMessageBox.information(self, "完成", f"报告已保存至:\n{path}")
