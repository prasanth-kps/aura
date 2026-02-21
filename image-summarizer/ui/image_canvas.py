"""
image_canvas.py — Image display canvas with context window overlay.

This widget displays the loaded image and hosts the draggable context
window rectangle. It also draws a semi-transparent dimming overlay
over the parts of the image OUTSIDE the context window, making it
visually clear which region is being analyzed.

Built on QGraphicsView/QGraphicsScene which provides:
    - Hardware-accelerated rendering
    - Smooth zooming and panning
    - Proper coordinate transformations
    - Built-in item interaction (drag, resize, etc.)
"""

from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsPathItem, QWidget, QVBoxLayout,
)
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QPixmap, QPen, QColor, QBrush, QPainterPath,
    QWheelEvent, QResizeEvent, QImage,
)
from PIL import Image
import numpy as np

from ui.context_rect import ContextRect


class ImageCanvas(QWidget):
    """
    The image display area with interactive context window.

    Signals:
        region_changed(x1, y1, x2, y2): Emitted when context window
            position/size is finalized (after debounce).
        region_moving(x1, y1, x2, y2): Emitted in real-time during
            drag/resize (for coordinate display).
    """

    # Forwarded signals from ContextRect
    region_changed = pyqtSignal(float, float, float, float)
    region_moving = pyqtSignal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._pixmap_item = None
        self._context_rect = None
        self._dimming_overlay = None
        self._image_size = (0, 0)

        self._setup_ui()

    def _setup_ui(self):
        """Set up the QGraphicsView and scene."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create scene (holds all graphical items)
        self._scene = QGraphicsScene()
        self._scene.setBackgroundBrush(QBrush(QColor(30, 30, 30)))

        # Create view (displays the scene with zoom/pan)
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(self._view.renderHints())
        self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Enable smooth scrolling
        self._view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)

        layout.addWidget(self._view)

    def load_image(self, image: Image.Image):
        """
        Load a PIL Image into the canvas.

        Args:
            image: PIL Image (RGB). Stored as a QPixmap in the scene.
        """
        # Convert PIL Image to QPixmap
        self._image_size = image.size
        qimage = self._pil_to_qimage(image)
        pixmap = QPixmap.fromImage(qimage)

        # Clear previous items
        self._scene.clear()
        self._pixmap_item = None
        self._context_rect = None
        self._dimming_overlay = None

        # Add image to scene
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)  # Bottom layer

        # Set scene rect to image size
        self._scene.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))

        # Add dimming overlay (above image, below context rect)
        self._dimming_overlay = QGraphicsPathItem()
        self._dimming_overlay.setBrush(QBrush(QColor(0, 0, 0, 100)))
        self._dimming_overlay.setPen(QPen(Qt.PenStyle.NoPen))
        self._dimming_overlay.setZValue(1)
        self._scene.addItem(self._dimming_overlay)

        # Create context window rectangle
        iw, ih = pixmap.width(), pixmap.height()
        # Default size: 30% of the smaller image dimension
        rect_size = int(min(iw, ih) * 0.3)
        rect_x = (iw - rect_size) // 2
        rect_y = (ih - rect_size) // 2

        image_bounds = QRectF(0, 0, iw, ih)
        self._context_rect = ContextRect(
            rect_x, rect_y, rect_size, rect_size,
            image_bounds=image_bounds,
            debounce_ms=300,
        )
        self._context_rect.setZValue(2)  # Top layer
        self._scene.addItem(self._context_rect)

        # Connect signals
        self._context_rect.signals.region_changed.connect(self._on_region_changed)
        self._context_rect.signals.region_moving.connect(self._on_region_moving)

        # Fit image in view
        self._view.fitInView(
            self._pixmap_item,
            Qt.AspectRatioMode.KeepAspectRatio,
        )

        # Initial dimming overlay update
        self._update_dimming_overlay()

    def load_image_from_path(self, path: str):
        """Load an image from a file path."""
        image = Image.open(path).convert("RGB")
        self.load_image(image)

    def get_region(self):
        """
        Get the current context window region.

        Returns:
            Tuple of (x1, y1, x2, y2) in image pixel coordinates,
            or None if no image is loaded.
        """
        if self._context_rect is None:
            return None
        return self._context_rect.get_region()

    def get_image_size(self):
        """Return the loaded image dimensions as (width, height)."""
        return self._image_size

    # ================================================================
    # Dimming Overlay
    # ================================================================

    def _update_dimming_overlay(self):
        """
        Update the dimming overlay to darken everything OUTSIDE the context window.

        This creates a visual "spotlight" effect where only the context window
        area is fully visible, and the rest of the image is dimmed.

        We achieve this using a QPainterPath with a "hole":
            1. Start with a path that covers the entire image
            2. Subtract the context window rectangle
            3. The result: everything except the context window is filled
        """
        if self._dimming_overlay is None or self._context_rect is None:
            return

        if self._pixmap_item is None:
            return

        # Full image rectangle
        image_rect = self._pixmap_item.boundingRect()

        # Context window rectangle (in scene coordinates)
        region = self._context_rect.get_region()
        context_rect = QRectF(region[0], region[1], region[2] - region[0], region[3] - region[1])

        # Create path with hole
        path = QPainterPath()
        path.addRect(image_rect)       # Full image
        path.addRect(context_rect)     # Hole (subtracted via even-odd fill rule)

        # Set fill rule to create the "hole" effect
        path.setFillRule(Qt.FillRule.OddEvenFill)

        self._dimming_overlay.setPath(path)

    # ================================================================
    # Signal Handlers
    # ================================================================

    def _on_region_changed(self, x1, y1, x2, y2):
        """Handle finalized context window position change."""
        self._update_dimming_overlay()
        self.region_changed.emit(x1, y1, x2, y2)

    def _on_region_moving(self, x1, y1, x2, y2):
        """Handle real-time context window movement."""
        self._update_dimming_overlay()
        self.region_moving.emit(x1, y1, x2, y2)

    # ================================================================
    # Zoom Support
    # ================================================================

    def wheelEvent(self, event: QWheelEvent):
        """Zoom in/out with scroll wheel."""
        zoom_factor = 1.15

        if event.angleDelta().y() > 0:
            self._view.scale(zoom_factor, zoom_factor)
        else:
            self._view.scale(1 / zoom_factor, 1 / zoom_factor)

    def resizeEvent(self, event: QResizeEvent):
        """Re-fit image when the widget is resized."""
        super().resizeEvent(event)
        if self._pixmap_item is not None:
            self._view.fitInView(
                self._pixmap_item,
                Qt.AspectRatioMode.KeepAspectRatio,
            )

    def fit_in_view(self):
        """Reset zoom to fit the full image in view."""
        if self._pixmap_item is not None:
            self._view.fitInView(
                self._pixmap_item,
                Qt.AspectRatioMode.KeepAspectRatio,
            )

    # ================================================================
    # Utility
    # ================================================================

    @staticmethod
    def _pil_to_qimage(pil_image: Image.Image) -> QImage:
        """
        Convert a PIL Image to QImage.

        This handles the format conversion needed to display PIL images
        in Qt's graphics framework.
        """
        pil_image = pil_image.convert("RGB")
        data = np.array(pil_image)

        height, width, channels = data.shape
        bytes_per_line = channels * width

        qimage = QImage(
            data.tobytes(),
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        # QImage doesn't copy the data by default, so we need to copy
        return qimage.copy()
