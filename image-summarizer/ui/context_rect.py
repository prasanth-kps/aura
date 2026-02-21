"""
context_rect.py — Draggable, resizable rectangle (the "Context Window").

This is the core interactive element of the application. It's a QGraphicsRectItem
that the user can:
    - Click and drag to MOVE to any position over the image
    - Drag edges to RESIZE horizontally or vertically
    - Drag corners to RESIZE in both dimensions simultaneously
    - See visual feedback (cursor changes, highlight on hover)

The rectangle emits a signal whenever it changes position or size,
which the main window uses to trigger re-summarization (with debouncing).

Implementation details:
    - Built on Qt's QGraphicsView framework (hardware-accelerated rendering)
    - Handles are invisible hit-test zones at edges/corners (10px wide)
    - The rectangle is constrained to stay within the image bounds
    - A dimming overlay shows which part of the image is OUTSIDE the context window
"""

from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsItem
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QPen, QColor, QBrush, QCursor, QPainter


class ContextRectSignals(QObject):
    """
    Signals emitted by the context rectangle.

    We need a separate QObject because QGraphicsRectItem doesn't inherit QObject
    and therefore can't emit signals directly.
    """
    # Emitted when the rect changes position or size (after debounce)
    region_changed = pyqtSignal(float, float, float, float)  # x1, y1, x2, y2

    # Emitted on every move (for real-time coordinate display)
    region_moving = pyqtSignal(float, float, float, float)


class ContextRect(QGraphicsRectItem):
    """
    A draggable, resizable rectangle overlay on the image.

    Usage:
        rect = ContextRect(100, 100, 200, 200, image_bounds)
        scene.addItem(rect)

        # Connect to signals
        rect.signals.region_changed.connect(on_region_finalized)
        rect.signals.region_moving.connect(on_region_moving)
    """

    # Size of the invisible edge/corner hit zones (pixels)
    HANDLE_SIZE = 12

    # Minimum dimensions (pixels) — prevents collapsing to zero
    MIN_WIDTH = 40
    MIN_HEIGHT = 40

    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        image_bounds: QRectF,
        debounce_ms: int = 300,
    ):
        """
        Create a new context window rectangle.

        Args:
            x, y: Top-left corner position in scene coordinates
            width, height: Initial size in pixels
            image_bounds: QRectF defining the image area (constrains the rect)
            debounce_ms: Milliseconds to wait after last move before emitting
                        region_changed signal (prevents excessive inference calls)
        """
        super().__init__(0, 0, width, height)
        self.setPos(x, y)

        # Store image boundaries for constraining movement
        self._image_bounds = image_bounds

        # --- Visual styling ---
        # Blue border with semi-transparent fill
        self.setPen(QPen(QColor(0, 120, 255, 200), 2.5, Qt.PenStyle.SolidLine))
        self.setBrush(QBrush(QColor(0, 120, 255, 25)))

        # --- Interaction flags ---
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

        # --- Resize state ---
        self._resizing = False
        self._resize_edges = ()      # Which edges are being dragged
        self._drag_start_pos = None  # Mouse position at drag start
        self._drag_start_rect = None  # Rect geometry at drag start
        self._dragging = False       # Whether we're in a move drag

        # --- Signals and debounce ---
        self.signals = ContextRectSignals()
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(debounce_ms)
        self._debounce_timer.timeout.connect(self._emit_region_changed)

    # ================================================================
    # Public API
    # ================================================================

    def get_region(self):
        """
        Get the current context window region in scene coordinates.

        Returns:
            Tuple of (x1, y1, x2, y2) — top-left and bottom-right corners
        """
        rect = self.rect()
        pos = self.pos()
        return (
            pos.x() + rect.x(),
            pos.y() + rect.y(),
            pos.x() + rect.x() + rect.width(),
            pos.y() + rect.y() + rect.height(),
        )

    def set_region(self, x1, y1, x2, y2):
        """Set the context window to specific coordinates."""
        self.setPos(x1, y1)
        self.setRect(0, 0, x2 - x1, y2 - y1)
        self._emit_region_changed()

    def set_image_bounds(self, bounds: QRectF):
        """Update the image bounds (call when image changes)."""
        self._image_bounds = bounds

    # ================================================================
    # Edge/Corner Detection
    # ================================================================

    def _detect_edges(self, local_pos: QPointF):
        """
        Detect which edges/corners the mouse is near.

        Args:
            local_pos: Mouse position in item-local coordinates

        Returns:
            Tuple of edge names, e.g. ("top",), ("bottom", "right"), or ()
            Empty tuple means the mouse is in the interior (drag to move).
        """
        rect = self.rect()
        h = self.HANDLE_SIZE
        edges = []

        if abs(local_pos.y() - rect.top()) < h:
            edges.append("top")
        if abs(local_pos.y() - rect.bottom()) < h:
            edges.append("bottom")
        if abs(local_pos.x() - rect.left()) < h:
            edges.append("left")
        if abs(local_pos.x() - rect.right()) < h:
            edges.append("right")

        return tuple(edges)

    def _cursor_for_edges(self, edges):
        """Return the appropriate cursor for the given edge combination."""
        cursor_map = {
            ("top",):              Qt.CursorShape.SizeVerCursor,
            ("bottom",):           Qt.CursorShape.SizeVerCursor,
            ("left",):             Qt.CursorShape.SizeHorCursor,
            ("right",):            Qt.CursorShape.SizeHorCursor,
            ("top", "left"):       Qt.CursorShape.SizeFDiagCursor,
            ("bottom", "right"):   Qt.CursorShape.SizeFDiagCursor,
            ("top", "right"):      Qt.CursorShape.SizeBDiagCursor,
            ("bottom", "left"):    Qt.CursorShape.SizeBDiagCursor,
        }
        return cursor_map.get(edges, Qt.CursorShape.OpenHandCursor)

    # ================================================================
    # Mouse Event Handlers
    # ================================================================

    def hoverMoveEvent(self, event):
        """Update cursor based on which edge/corner is hovered."""
        edges = self._detect_edges(event.pos())
        cursor = self._cursor_for_edges(edges)
        self.setCursor(QCursor(cursor))
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        """Reset cursor when mouse leaves the rectangle."""
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        """
        Handle mouse press — start either a resize or move operation.

        If the press is near an edge/corner → start resizing
        If the press is in the interior → start moving (default behavior)
        """
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        edges = self._detect_edges(event.pos())

        if edges:
            # Start resize
            self._resizing = True
            self._resize_edges = edges
            self._drag_start_pos = event.pos()
            self._drag_start_rect = QRectF(self.rect())
            self.setCursor(QCursor(self._cursor_for_edges(edges)))
            event.accept()
        else:
            # Start move
            self._dragging = True
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """
        Handle mouse move — either resize or move the rectangle.
        """
        if self._resizing:
            self._handle_resize(event)
        else:
            # Let Qt handle the move (ItemIsMovable flag)
            super().mouseMoveEvent(event)
            # Constrain position to image bounds
            self._constrain_to_bounds()

        # Emit real-time position for coordinate display
        region = self.get_region()
        self.signals.region_moving.emit(*region)

        # Start/restart debounce timer
        self._debounce_timer.start()

    def mouseReleaseEvent(self, event):
        """Handle mouse release — finalize resize or move."""
        if self._resizing:
            self._resizing = False
            self._resize_edges = ()
            edges = self._detect_edges(event.pos())
            self.setCursor(QCursor(self._cursor_for_edges(edges)))
        else:
            self._dragging = False
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            super().mouseReleaseEvent(event)

        # Emit final position
        self._emit_region_changed()

    # ================================================================
    # Resize Logic
    # ================================================================

    def _handle_resize(self, event):
        """
        Resize the rectangle based on which edge(s) are being dragged.

        The resize is computed as a delta from the initial mouse position
        and applied to the initial rectangle geometry. This prevents
        accumulated floating-point errors from causing jitter.
        """
        delta = event.pos() - self._drag_start_pos
        new_rect = QRectF(self._drag_start_rect)

        # Apply delta to the appropriate edges
        if "left" in self._resize_edges:
            new_left = new_rect.left() + delta.x()
            # Enforce minimum width
            if new_rect.right() - new_left >= self.MIN_WIDTH:
                new_rect.setLeft(new_left)

        if "right" in self._resize_edges:
            new_right = new_rect.right() + delta.x()
            if new_right - new_rect.left() >= self.MIN_WIDTH:
                new_rect.setRight(new_right)

        if "top" in self._resize_edges:
            new_top = new_rect.top() + delta.y()
            if new_rect.bottom() - new_top >= self.MIN_HEIGHT:
                new_rect.setTop(new_top)

        if "bottom" in self._resize_edges:
            new_bottom = new_rect.bottom() + delta.y()
            if new_bottom - new_rect.top() >= self.MIN_HEIGHT:
                new_rect.setBottom(new_bottom)

        self.setRect(new_rect)

    def _constrain_to_bounds(self):
        """
        Ensure the rectangle stays within the image boundaries.

        Called after every move to prevent the context window from
        extending beyond the image edges.
        """
        if self._image_bounds is None:
            return

        pos = self.pos()
        rect = self.rect()
        bounds = self._image_bounds

        # Constrain position
        new_x = pos.x()
        new_y = pos.y()

        if new_x + rect.left() < bounds.left():
            new_x = bounds.left() - rect.left()
        if new_y + rect.top() < bounds.top():
            new_y = bounds.top() - rect.top()
        if new_x + rect.right() > bounds.right():
            new_x = bounds.right() - rect.right()
        if new_y + rect.bottom() > bounds.bottom():
            new_y = bounds.bottom() - rect.bottom()

        self.setPos(new_x, new_y)

    # ================================================================
    # Signal Emission
    # ================================================================

    def _emit_region_changed(self):
        """Emit the region_changed signal with current coordinates."""
        region = self.get_region()
        self.signals.region_changed.emit(*region)

    # ================================================================
    # Custom Painting
    # ================================================================

    def paint(self, painter: QPainter, option, widget=None):
        """
        Custom paint to draw the context window with visual enhancements.

        Draws:
            1. The semi-transparent fill
            2. The blue border
            3. Corner handles (small squares at corners)
            4. Size label (width × height)
        """
        rect = self.rect()

        # Draw fill and border (default behavior)
        super().paint(painter, option, widget)

        # Draw corner handles
        handle_size = 8
        handle_color = QColor(0, 120, 255, 220)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(handle_color))

        corners = [
            (rect.left(), rect.top()),
            (rect.right() - handle_size, rect.top()),
            (rect.left(), rect.bottom() - handle_size),
            (rect.right() - handle_size, rect.bottom() - handle_size),
        ]
        for cx, cy in corners:
            painter.drawRect(QRectF(cx, cy, handle_size, handle_size))

        # Draw midpoint handles on edges
        mid_handles = [
            (rect.center().x() - handle_size / 2, rect.top()),           # top-center
            (rect.center().x() - handle_size / 2, rect.bottom() - handle_size),  # bottom-center
            (rect.left(), rect.center().y() - handle_size / 2),          # left-center
            (rect.right() - handle_size, rect.center().y() - handle_size / 2),   # right-center
        ]
        painter.setBrush(QBrush(QColor(0, 120, 255, 150)))
        for mx, my in mid_handles:
            painter.drawRect(QRectF(mx, my, handle_size, handle_size))

        # Draw size label
        w = int(rect.width())
        h = int(rect.height())
        label = f"{w} × {h}"
        painter.setPen(QPen(QColor(255, 255, 255, 200)))
        painter.setFont(painter.font())

        label_rect = QRectF(
            rect.left() + 4,
            rect.bottom() - 20,
            rect.width() - 8,
            18,
        )
        # Background for label
        painter.setBrush(QBrush(QColor(0, 0, 0, 120)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(label_rect, 3, 3)

        # Text
        painter.setPen(QPen(QColor(255, 255, 255, 220)))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)
