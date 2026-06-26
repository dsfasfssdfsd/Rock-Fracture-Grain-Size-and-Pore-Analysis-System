from PyQt5.QtWidgets import QToolBar, QAction


class ToolBar(QToolBar):
    def __init__(self, parent=None):
        super().__init__("工具栏", parent)
        self.setMovable(False)
        self._setup_actions()

    def _setup_actions(self):
        self._act_open = QAction("打开图像", self)
        self.addAction(self._act_open)

        self._act_save = QAction("保存结果", self)
        self.addAction(self._act_save)

        self.addSeparator()

        self._act_analyze = QAction("开始分析", self)
        self.addAction(self._act_analyze)

        self._act_auto_fracture = QAction("自动裂缝分析", self)
        self.addAction(self._act_auto_fracture)

        self._act_auto_pore = QAction("自动孔洞分析", self)
        self.addAction(self._act_auto_pore)

        self._act_auto_grain = QAction("自动粒度分析", self)
        self.addAction(self._act_auto_grain)

        self.addSeparator()

        self._act_reset = QAction("重置", self)
        self.addAction(self._act_reset)

        self._act_report = QAction("生成报告", self)
        self.addAction(self._act_report)

    @property
    def act_open(self):
        return self._act_open

    @property
    def act_save(self):
        return self._act_save

    @property
    def act_analyze(self):
        return self._act_analyze

    @property
    def act_auto_fracture(self):
        return self._act_auto_fracture

    @property
    def act_auto_pore(self):
        return self._act_auto_pore

    @property
    def act_auto_grain(self):
        return self._act_auto_grain

    @property
    def act_reset(self):
        return self._act_reset

    @property
    def act_report(self):
        return self._act_report
