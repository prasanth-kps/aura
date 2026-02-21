"""
main_window.py — Main application window for Edge AI Image Summarizer.

This is the top-level window that assembles all UI components:
    - Toolbar with load, summarize, mode selection, and settings
    - Image canvas (left panel) with the interactive context window
    - Summary panel (right panel) with results and metadata
    - Status bar with inference timing and model information

Layout:
    ┌─────────────────────────────────────────────────────────┐
    │  [Load Image] [Summarize] [Mode: ▼] [Fit] [Status...] │
    ├──────────────────────────────┬──────────────────────────┤
    │                              │  Region Summary:         │
    │                              │  ┌────────────────────┐  │
    │    Image Canvas              │  │ [Cropped Thumb]    │  │
    │    ┌──────────┐              │  │                    │  │
    │    │ Context  │              │  │ Summary text...    │  │
    │    │ Window   │              │  │                    │  │
    │    └──────────┘              │  └────────────────────┘  │
    │                              │                          │
    │                              │  Details:                │
    │                              │  • Scene: person (0.82)  │
    │                              │  • Attr: outdoor (0.71)  │
    │                              │  • Time: 23ms (NPU)      │
    └──────────────────────────────┴──────────────────────────┘
"""

import os
import sys
from PIL import Image

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QLabel, QTextEdit, QComboBox,
    QFileDialog, QGroupBox, QGridLayout, QApplication,
    QFrame, QScrollArea, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QAction

from ui.image_canvas import ImageCanvas
from engine.inference_engine import InferenceEngine
import numpy as np


class InferenceWorker(QThread):
    """
    Run AI inference in a background thread.

    This prevents the UI from freezing during model inference.
    The summarization call blocks for 20-500ms depending on the model
    and hardware, which would cause visible UI jank if run on the main thread.
    """
    finished = pyqtSignal(dict)  # Emits the result dictionary
    error = pyqtSignal(str)       # Emits error message if inference fails

    def __init__(self, engine: InferenceEngine, image: Image.Image, bbox: tuple):
        super().__init__()
        self.engine = engine
        self.image = image
        self.bbox = bbox

    def run(self):
        try:
            result = self.engine.summarize(self.image, self.bbox)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge AI Image Summarizer — Qualcomm AI Hub")
        self.setMinimumSize(1100, 650)
        self.resize(1300, 750)

        # State
        self._current_image = None  # PIL Image
        self._current_image_path = None
        self._worker = None

        # Initialize AI engine
        self._init_engine()

        # Build UI
        self._setup_ui()
        self._setup_menu()

        # Show initial status
        self._update_status("Ready. Load an image to begin.", "gray")

    # ================================================================
    # Initialization
    # ================================================================

    def _init_engine(self):
        """
        Initialize the AI inference engine.

        Tries to find compiled models in the models/ directory.
        Falls back to HuggingFace CPU models if not found.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, "models")

        # Check for compiled ONNX models
        clip_model = os.path.join(models_dir, "clip_optimized.onnx")
        clip_text_emb = os.path.join(models_dir, "clip_text_embeddings.npy")
        blip_model = os.path.join(models_dir, "blip_vision_optimized.onnx")

        self._engine = InferenceEngine(
            mode="clip",
            clip_model_path=clip_model if os.path.exists(clip_model) else None,
            clip_text_embeddings_path=clip_text_emb if os.path.exists(clip_text_emb) else None,
            blip_model_path=blip_model if os.path.exists(blip_model) else None,
            use_npu=True,
        )

    # ================================================================
    # UI Setup
    # ================================================================

    def _setup_ui(self):
        """Build the complete UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # --- Toolbar ---
        toolbar = self._create_toolbar()
        main_layout.addLayout(toolbar)

        # --- Main content (splitter: canvas | summary) ---
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Image canvas
        self._canvas = ImageCanvas()
        self._canvas.region_changed.connect(self._on_region_finalized)
        self._canvas.region_moving.connect(self._on_region_moving)
        self._splitter.addWidget(self._canvas)

        # Right: Summary panel
        right_panel = self._create_summary_panel()
        self._splitter.addWidget(right_panel)

        # Set initial splitter proportions (70% canvas, 30% summary)
        self._splitter.setSizes([900, 400])
        self._splitter.setStretchFactor(0, 7)
        self._splitter.setStretchFactor(1, 3)

        main_layout.addWidget(self._splitter)

        # --- Status bar ---
        self._status_label = QLabel()
        self._status_label.setStyleSheet("padding: 4px; color: #888;")
        main_layout.addWidget(self._status_label)

    def _create_toolbar(self) -> QHBoxLayout:
        """Create the top toolbar with action buttons."""
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        # Load Image button
        self._btn_load = QPushButton("  Load Image")
        self._btn_load.setMinimumHeight(36)
        self._btn_load.setStyleSheet("""
            QPushButton {
                background-color: #0078D4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1084D8; }
            QPushButton:pressed { background-color: #006CBD; }
        """)
        self._btn_load.clicked.connect(self._load_image)
        toolbar.addWidget(self._btn_load)

        # Summarize button
        self._btn_summarize = QPushButton("  Summarize Region")
        self._btn_summarize.setMinimumHeight(36)
        self._btn_summarize.setEnabled(False)
        self._btn_summarize.setStyleSheet("""
            QPushButton {
                background-color: #107C10;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1B8C1B; }
            QPushButton:pressed { background-color: #0E6B0E; }
            QPushButton:disabled { background-color: #555; color: #999; }
        """)
        self._btn_summarize.clicked.connect(self._run_summarization)
        toolbar.addWidget(self._btn_summarize)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #555;")
        toolbar.addWidget(sep)

        # Mode selector
        toolbar.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["clip", "blip", "hybrid"])
        self._mode_combo.setCurrentText("clip")
        self._mode_combo.setMinimumHeight(32)
        self._mode_combo.setToolTip(
            "clip: Fast zero-shot (recommended)\n"
            "blip: Natural language captions\n"
            "hybrid: Both models combined"
        )
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        toolbar.addWidget(self._mode_combo)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color: #555;")
        toolbar.addWidget(sep2)

        # Fit to view button
        btn_fit = QPushButton("Fit View")
        btn_fit.setMinimumHeight(32)
        btn_fit.clicked.connect(lambda: self._canvas.fit_in_view())
        toolbar.addWidget(btn_fit)

        # Auto-summarize checkbox (summarize on every context window move)
        self._auto_summarize = False
        self._btn_auto = QPushButton("Auto: OFF")
        self._btn_auto.setMinimumHeight(32)
        self._btn_auto.setCheckable(True)
        self._btn_auto.setToolTip("When ON, summarizes automatically after moving the context window")
        self._btn_auto.toggled.connect(self._toggle_auto_summarize)
        toolbar.addWidget(self._btn_auto)

        toolbar.addStretch()

        # Model info label
        models = self._engine.get_active_models()
        model_text = ", ".join(models) if models else "No models"
        self._model_info_label = QLabel(f"Models: {model_text}")
        self._model_info_label.setStyleSheet("color: #888; font-size: 11px;")
        toolbar.addWidget(self._model_info_label)

        return toolbar

    def _create_summary_panel(self) -> QWidget:
        """Create the right-side summary panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        # --- Header ---
        header = QLabel("Region Summary")
        header.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        header.setStyleSheet("color: #DDD; padding: 4px 0;")
        layout.addWidget(header)

        # --- Cropped region thumbnail ---
        self._thumbnail_label = QLabel()
        self._thumbnail_label.setFixedHeight(150)
        self._thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumbnail_label.setStyleSheet("""
            QLabel {
                background-color: #1E1E1E;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)
        self._thumbnail_label.setText("Cropped region will appear here")
        layout.addWidget(self._thumbnail_label)

        # --- Summary text ---
        summary_group = QGroupBox("AI Summary")
        summary_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 6px;
                color: #AAA;
            }
        """)
        summary_layout = QVBoxLayout(summary_group)

        self._summary_text = QTextEdit()
        self._summary_text.setReadOnly(True)
        self._summary_text.setPlaceholderText(
            "Move the blue context window over any region of the image "
            "and click 'Summarize Region' to see an AI-generated description.\n\n"
            "You can also enable 'Auto' mode for continuous summarization."
        )
        self._summary_text.setStyleSheet("""
            QTextEdit {
                background-color: #1A1A2E;
                color: #E0E0E0;
                border: none;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
                line-height: 1.5;
            }
        """)
        self._summary_text.setMinimumHeight(100)
        summary_layout.addWidget(self._summary_text)
        layout.addWidget(summary_group)

        # --- Details panel ---
        details_group = QGroupBox("Details")
        details_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 6px;
                color: #AAA;
            }
        """)
        details_layout = QVBoxLayout(details_group)

        self._details_text = QTextEdit()
        self._details_text.setReadOnly(True)
        self._details_text.setMaximumHeight(200)
        self._details_text.setStyleSheet("""
            QTextEdit {
                background-color: #111;
                color: #AAA;
                border: none;
                border-radius: 4px;
                padding: 6px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        details_layout.addWidget(self._details_text)
        layout.addWidget(details_group)

        # --- Coordinates display ---
        self._coords_label = QLabel("Region: — × —")
        self._coords_label.setStyleSheet("color: #777; font-size: 11px; padding: 2px;")
        layout.addWidget(self._coords_label)

        layout.addStretch()

        return panel

    def _setup_menu(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Image...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._load_image)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        fit_action = QAction("&Fit to Window", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(lambda: self._canvas.fit_in_view())
        view_menu.addAction(fit_action)

    # ================================================================
    # Actions
    # ================================================================

    def _load_image(self):
        """Open file dialog and load an image."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff);;All Files (*)",
        )
        if not path:
            return

        try:
            self._current_image = Image.open(path).convert("RGB")
            self._current_image_path = path
            self._canvas.load_image(self._current_image)
            self._btn_summarize.setEnabled(True)

            # Clear previous results
            self._summary_text.clear()
            self._details_text.clear()
            self._thumbnail_label.clear()
            self._thumbnail_label.setText("Cropped region will appear here")

            filename = os.path.basename(path)
            w, h = self._current_image.size
            self._update_status(f"Loaded: {filename} ({w}×{h}px) — Drag the blue box, then Summarize", "#4CAF50")

        except Exception as e:
            self._update_status(f"Error loading image: {e}", "#F44336")

    def _run_summarization(self):
        """Run AI summarization on the current context window region."""
        if self._current_image is None:
            return

        region = self._canvas.get_region()
        if region is None:
            return

        # Prevent multiple concurrent inference calls
        if self._worker is not None and self._worker.isRunning():
            return

        self._update_status("Running AI inference...", "#FFC107")
        self._btn_summarize.setEnabled(False)
        QApplication.processEvents()

        # Run inference in background thread
        self._worker = InferenceWorker(self._engine, self._current_image, region)
        self._worker.finished.connect(self._on_inference_complete)
        self._worker.error.connect(self._on_inference_error)
        self._worker.start()

    def _on_inference_complete(self, result: dict):
        """Handle completed inference — update the summary panel."""
        self._btn_summarize.setEnabled(True)

        # Update summary text
        self._summary_text.setHtml(f"""
            <div style="font-size: 14px; line-height: 1.6; color: #E0E0E0;">
                {result['summary']}
            </div>
        """)

        # Update thumbnail
        if result.get("cropped_image"):
            self._update_thumbnail(result["cropped_image"])

        # Update details
        self._update_details(result)

        # Update status
        model_name = result.get("model_used", "unknown")
        total_ms = result.get("total_ms", 0)
        self._update_status(
            f"Inference complete: {total_ms}ms | Model: {model_name}",
            "#4CAF50",
        )

    def _on_inference_error(self, error_msg: str):
        """Handle inference error."""
        self._btn_summarize.setEnabled(True)
        self._summary_text.setPlainText(f"Error: {error_msg}")
        self._update_status(f"Inference error: {error_msg}", "#F44336")

    # ================================================================
    # UI Update Helpers
    # ================================================================

    def _update_thumbnail(self, cropped_image: Image.Image):
        """Display the cropped region as a thumbnail."""
        # Resize to fit the label
        thumb = cropped_image.copy()
        thumb.thumbnail((280, 140), Image.BICUBIC)

        # Convert to QPixmap
        data = np.array(thumb.convert("RGB"))
        h, w, ch = data.shape
        qimage = QImage(data.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        self._thumbnail_label.setPixmap(pixmap)

    def _update_details(self, result: dict):
        """Update the details panel with inference metadata."""
        details = result.get("details", {})

        lines = []
        lines.append(f"Model: {result.get('model_used', 'N/A')}")
        lines.append(f"Total time: {result.get('total_ms', 0)}ms")
        lines.append("")

        # CLIP details
        if "scenes" in details:
            lines.append("--- Scene Matches ---")
            for desc, score in details["scenes"]:
                bar = "█" * int(score * 30)
                lines.append(f"  {score:.3f} {bar} {desc}")

        if "attributes" in details:
            lines.append("\n--- Attributes ---")
            for desc, score in details["attributes"]:
                bar = "█" * int(score * 30)
                lines.append(f"  {score:.3f} {bar} {desc}")

        if "actions" in details:
            lines.append("\n--- Actions ---")
            for desc, score in details["actions"]:
                bar = "█" * int(score * 30)
                lines.append(f"  {score:.3f} {bar} {desc}")

        # Timing breakdown
        if "clip_ms" in details:
            lines.append(f"\nCLIP inference: {details['clip_ms']}ms")
        if "blip_ms" in details:
            lines.append(f"BLIP inference: {details['blip_ms']}ms")

        self._details_text.setPlainText("\n".join(lines))

    def _update_status(self, text: str, color: str = "#888"):
        """Update the status bar text and color."""
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"padding: 4px; color: {color}; font-size: 12px;")

    # ================================================================
    # Event Handlers
    # ================================================================

    def _on_region_finalized(self, x1, y1, x2, y2):
        """Handle context window position finalized (after debounce)."""
        self._update_coords(x1, y1, x2, y2)
        if self._auto_summarize and self._current_image is not None:
            self._run_summarization()

    def _on_region_moving(self, x1, y1, x2, y2):
        """Handle real-time context window movement (for coordinate display)."""
        self._update_coords(x1, y1, x2, y2)

    def _update_coords(self, x1, y1, x2, y2):
        """Update the coordinates display."""
        w = int(x2 - x1)
        h = int(y2 - y1)
        self._coords_label.setText(
            f"Region: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)}) — {w}×{h}px"
        )

    def _on_mode_changed(self, mode: str):
        """Handle summarization mode change."""
        self._update_status(f"Switching to {mode} mode...", "#FFC107")
        QApplication.processEvents()

        self._engine.change_mode(mode)

        models = self._engine.get_active_models()
        model_text = ", ".join(models) if models else "No models"
        self._model_info_label.setText(f"Models: {model_text}")

        self._update_status(f"Mode: {mode} | Models: {model_text}", "#4CAF50")

    def _toggle_auto_summarize(self, checked: bool):
        """Toggle auto-summarize mode."""
        self._auto_summarize = checked
        self._btn_auto.setText(f"Auto: {'ON' if checked else 'OFF'}")
        if checked:
            self._btn_auto.setStyleSheet("""
                QPushButton { background-color: #107C10; color: white;
                              border-radius: 4px; padding: 4px 12px; }
            """)
        else:
            self._btn_auto.setStyleSheet("")
