"""
main.py — Entry point for the Edge AI Image Summarizer application.

Usage:
    python main.py                  # Launch with default settings (CLIP mode)
    python main.py --mode blip      # Launch with BLIP captioning
    python main.py --mode hybrid    # Launch with both CLIP + BLIP
    python main.py --no-npu         # Disable NPU, run on CPU only
    python main.py --help           # Show all options

This application runs entirely on-device using models from Qualcomm AI Hub.
No internet connection is required after initial model download.
"""

import sys
import os
import argparse

# Add project root to path so imports work
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Edge AI Image Summarizer — On-device image region summarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Launch with CLIP (fast, recommended)
  python main.py --mode blip        Launch with BLIP captioning
  python main.py --mode hybrid      Launch with CLIP + BLIP
  python main.py --no-npu           Force CPU-only inference
  python main.py --image photo.jpg  Launch and load an image immediately

Models:
  Place compiled ONNX models in the models/ directory:
    models/clip_optimized.onnx           - CLIP image encoder (from Qualcomm AI Hub)
    models/clip_text_embeddings.npy      - Pre-computed text embeddings
    models/blip_vision_optimized.onnx    - BLIP vision encoder (optional)

  If no compiled models are found, the app falls back to HuggingFace
  models running on CPU (slower but always works).

  To export models for NPU acceleration:
    python scripts/export_clip.py
    python scripts/precompute_text_embeddings.py
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["clip", "blip", "hybrid"],
        default="clip",
        help="Summarization mode (default: clip)",
    )
    parser.add_argument(
        "--no-npu",
        action="store_true",
        help="Disable NPU acceleration, use CPU only",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image to load at startup",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default=None,
        help="Path to compiled CLIP ONNX model (overrides default location)",
    )
    parser.add_argument(
        "--blip-model",
        type=str,
        default=None,
        help="Path to compiled BLIP ONNX model (overrides default location)",
    )

    return parser.parse_args()


def apply_dark_theme(app):
    """
    Apply a modern dark theme to the application.

    This matches the aesthetic of professional AI tools and looks good
    for hackathon demos.
    """
    app.setStyle("Fusion")

    stylesheet = """
    QMainWindow {
        background-color: #1E1E1E;
    }
    QWidget {
        background-color: #1E1E1E;
        color: #D4D4D4;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        font-size: 12px;
    }
    QMenuBar {
        background-color: #2D2D2D;
        color: #D4D4D4;
        border-bottom: 1px solid #3E3E3E;
    }
    QMenuBar::item:selected {
        background-color: #0078D4;
    }
    QMenu {
        background-color: #2D2D2D;
        color: #D4D4D4;
        border: 1px solid #3E3E3E;
    }
    QMenu::item:selected {
        background-color: #0078D4;
    }
    QPushButton {
        background-color: #333;
        color: #D4D4D4;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 5px 12px;
        min-height: 24px;
    }
    QPushButton:hover {
        background-color: #444;
        border-color: #666;
    }
    QPushButton:pressed {
        background-color: #222;
    }
    QPushButton:disabled {
        color: #666;
        background-color: #2A2A2A;
        border-color: #3A3A3A;
    }
    QComboBox {
        background-color: #333;
        color: #D4D4D4;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 4px 8px;
    }
    QComboBox::drop-down {
        border: none;
    }
    QComboBox QAbstractItemView {
        background-color: #2D2D2D;
        color: #D4D4D4;
        selection-background-color: #0078D4;
    }
    QSplitter::handle {
        background-color: #3E3E3E;
    }
    QSplitter::handle:horizontal {
        width: 3px;
    }
    QScrollBar:vertical {
        background-color: #1E1E1E;
        width: 10px;
    }
    QScrollBar::handle:vertical {
        background-color: #555;
        border-radius: 5px;
        min-height: 20px;
    }
    QScrollBar::handle:vertical:hover {
        background-color: #777;
    }
    QScrollBar:horizontal {
        background-color: #1E1E1E;
        height: 10px;
    }
    QScrollBar::handle:horizontal {
        background-color: #555;
        border-radius: 5px;
        min-width: 20px;
    }
    QScrollBar::add-line, QScrollBar::sub-line {
        height: 0;
        width: 0;
    }
    QGroupBox {
        color: #AAA;
    }
    QLabel {
        color: #D4D4D4;
    }
    QTextEdit {
        background-color: #1A1A2E;
        color: #D4D4D4;
        border: 1px solid #333;
        border-radius: 4px;
    }
    """

    app.setStyleSheet(stylesheet)


def main():
    """Application entry point."""
    args = parse_args()

    # Print startup banner
    print("=" * 60)
    print("  Edge AI Image Summarizer")
    print("  Qualcomm AI Hub × Edge AI Developer Hackathon")
    print("=" * 60)
    print(f"  Mode:       {args.mode}")
    print(f"  NPU:        {'Enabled' if not args.no_npu else 'Disabled (CPU only)'}")
    print(f"  Image:      {args.image or 'None (load from UI)'}")
    print("=" * 60)
    print()

    # Create Qt application
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("Edge AI Image Summarizer")
    app.setOrganizationName("Edge AI Hackathon")

    # Apply dark theme
    apply_dark_theme(app)

    # Create and show main window
    from ui.main_window import MainWindow
    window = MainWindow()

    # Override engine settings from CLI args
    if args.mode != "clip":
        window._engine.change_mode(args.mode)
        window._mode_combo.setCurrentText(args.mode)

    # Load image from CLI if provided
    if args.image and os.path.exists(args.image):
        from PIL import Image
        window._current_image = Image.open(args.image).convert("RGB")
        window._current_image_path = args.image
        window._canvas.load_image(window._current_image)
        window._btn_summarize.setEnabled(True)

    window.show()

    print("[App] Window displayed. Ready for interaction.")
    print("[App] Use Ctrl+O to open an image, or drag the context window and click Summarize.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
