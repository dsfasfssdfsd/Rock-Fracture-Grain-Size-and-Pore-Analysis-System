import cv2
import numpy as np
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter


class ImageViewer(QGraphicsView):
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self._pixmap_item = None
        self._overlay_items = []
        self._zoom = 1.0

        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def set_image(self, img_bgr):
        self.scene.clear()
        self._overlay_items = []
        self._pixmap_item = None
        if img_bgr is None:
            return
        h, w, ch = img_bgr.shape
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self._pixmap_item)
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self._zoom = 1.0
        self.viewport().update()
        self.zoom_changed.emit(100)

    def add_overlay(self, overlay_bgr, opacity=128):
        if self._pixmap_item is None:
            return
        h, w = overlay_bgr.shape[:2]
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        alpha = np.full((h, w, 1), opacity, dtype=np.uint8)
        rgba = np.concatenate([overlay_rgb, alpha], axis=2)
        qimg = QImage(rgba.data, w, h, rgba.strides[0], QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)
        item = QGraphicsPixmapItem(pixmap)
        item.setTransformationMode(Qt.SmoothTransformation)
        self.scene.addItem(item)
        self._overlay_items.append(item)

    def clear_overlays(self):
        for item in self._overlay_items:
            self.scene.removeItem(item)
        self._overlay_items = []

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self._zoom *= factor
        self.zoom_changed.emit(int(self._zoom * 100))

    def get_scene_image(self):
        if self._pixmap_item is None:
            return None
        rect = self._pixmap_item.boundingRect()
        pixmap = QPixmap(int(rect.width()), int(rect.height()))
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        self.scene.render(painter, QRectF(pixmap.rect()), rect)
        painter.end()
        qimg = pixmap.toImage()
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        arr = np.array(ptr).reshape(qimg.height(), qimg.width(), 4)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
