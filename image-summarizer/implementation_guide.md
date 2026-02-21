# Edge AI Image Summarizer — Full Implementation Guide
## Qualcomm AI Hub × Edge AI Developer Hackathon

---

## 1. Project Overview

**What you're building:** A desktop application (Windows on Snapdragon) that lets users upload an image, draw/resize/move a rectangular "context window" over it, and receive an AI-generated text summary describing *only what's inside that window* — all running **entirely on-device** using Qualcomm NPU acceleration.

**Core user flow:**
1. User loads an image into the app
2. A draggable, resizable rectangle (the "context window") overlays the image
3. User positions/resizes the rectangle over the region of interest
4. The cropped region is fed to an on-device vision-language model
5. A text summary of that region appears in a panel beside the image

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Desktop UI (PyQt6)                  │
│  ┌──────────────────────┐  ┌──────────────────────┐ │
│  │   Image Canvas        │  │   Summary Panel      │ │
│  │  ┌──────────┐        │  │                      │ │
│  │  │ Context  │ ← drag │  │  "A dog playing      │ │
│  │  │ Window   │   resize│  │   with a red ball    │ │
│  │  └──────────┘        │  │   on green grass..."  │ │
│  └──────────────────────┘  └──────────────────────┘ │
└─────────────┬───────────────────────────────────────┘
              │  (cropped region)
              ▼
┌─────────────────────────────────────────────────────┐
│            Inference Engine (Python)                  │
│                                                      │
│  Option A: CLIP (from QAI Hub) → zero-shot labels   │
│  Option B: BLIP (exported via ONNX) → captions      │
│  Option C: OpenAI CLIP + Small LLM → rich summaries │
│                                                      │
│  Runtime: ONNX Runtime + QNN Execution Provider      │
│           (Hexagon NPU acceleration)                 │
└─────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack — What to Use and Why

### 3.1 Hardware Target
| Component | Choice | Why |
|-----------|--------|-----|
| Device | **Copilot+ PC / Snapdragon X Elite** | Hackathon provides these devices; they have the Hexagon NPU for AI acceleration |
| Accelerator | **Qualcomm Hexagon NPU** | Purpose-built for AI inference — 4x faster than CPU, dramatically lower power |

### 3.2 AI Models from Qualcomm AI Hub

**Primary model — OpenAI CLIP (available on Qualcomm AI Hub):**
- **What it is:** A multi-modal model that jointly understands images and text. It has a ViT-B/16 image encoder and a text encoder.
- **Why use it:** It's already optimized and benchmarked on Qualcomm AI Hub for Snapdragon devices. 150M parameters, 571 MB. Supports zero-shot classification and image-text similarity.
- **Model page:** https://aihub.qualcomm.com/models/openai_clip
- **How it helps your app:** You crop the image to the context window, encode it with CLIP's vision encoder, then compare against a bank of descriptive text prompts to find the best description.

**Secondary option — Export BLIP yourself:**
- **What it is:** A dedicated image captioning model (vision encoder + text decoder) from Salesforce. Generates natural language captions.
- **Why use it:** Produces free-form text descriptions rather than selecting from fixed labels. More natural output.
- **How to get it on-device:** Export from PyTorch → ONNX → compile via Qualcomm AI Hub Workbench → deploy with QNN Execution Provider.

**Supplementary models (also on Qualcomm AI Hub):**
- **TrOCR** — If the context window contains text, use this for OCR to include readable text in the summary.
- **YOLOv7/v8** — Real-time object detection to identify objects within the context window.

### 3.3 Runtime & SDK Stack

| Layer | Technology | Significance |
|-------|-----------|--------------|
| **AI Runtime** | **ONNX Runtime + QNN Execution Provider** | Industry-standard runtime; the QNN EP offloads computation to the Hexagon NPU. This is the recommended path for Windows on Snapdragon. |
| **Model Optimization** | **Qualcomm AI Hub Workbench** | Cloud service that compiles, quantizes, and profiles models for specific Snapdragon chipsets. Produces optimized `.onnx` + QNN context binaries. |
| **Low-level SDK** | **Qualcomm AI Engine Direct (QNN SDK)** | The underlying SDK that QNN EP uses. Provides direct NPU/GPU/CPU access. You won't call this directly — ONNX Runtime wraps it. |
| **Quantization** | **INT8 (W8A8)** via AI Hub Workbench | Reduces model size and speeds up inference by ~2-4x with minimal accuracy loss. Essential for real-time context window updates. |

### 3.4 Application Framework

| Component | Technology | Why |
|-----------|-----------|-----|
| **UI Framework** | **PyQt6** (or PySide6) | Rich widget toolkit for Python. Has `QGraphicsView` / `QGraphicsScene` which make movable, resizable rectangles trivial. Cross-platform. |
| **Image handling** | **Pillow (PIL)** | Standard Python image library for cropping, resizing, format conversion |
| **Array ops** | **NumPy** | Preprocessing images into tensors for model input |
| **Language** | **Python 3.10+** (x64 on Snapdragon) | Note: Qualcomm AI Hub requires x64 Python on Windows ARM, not ARM64 Python |

---

## 4. Step-by-Step Implementation

### Phase 1: Environment Setup

```bash
# 1. Install x64 Python (CRITICAL — ARM64 Python won't work with QAI Hub)
# Download Python 3.10+ x64 installer from python.org

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install core dependencies
pip install PyQt6 Pillow numpy onnxruntime-qnn

# 4. Install Qualcomm AI Hub SDK
pip install qai-hub qai-hub-models

# 5. Configure API token (get from https://aihub.qualcomm.com)
qai-hub configure --api_token YOUR_API_TOKEN
```

### Phase 2: Get the CLIP Model from Qualcomm AI Hub

```python
# export_clip.py — Download and compile CLIP for Snapdragon X Elite
import qai_hub as hub
from qai_hub_models.models.openai_clip import Model

# Load the pre-trained CLIP model
torch_model = Model.from_pretrained()

# Export for Snapdragon X Elite
# This compiles and optimizes the model for the NPU
python_cmd = """
python -m qai_hub_models.models.openai_clip.export \
    --device "Snapdragon X Elite CRD" \
    --target-runtime onnx \
    --output-dir ./models/clip/
"""
# Run this command in your terminal

# Alternative: Compile manually
import torch

# Trace the model
sample_inputs = torch_model.sample_inputs()
traced_model = torch.jit.trace(
    torch_model,
    [torch.tensor(data[0]) for _, data in sample_inputs.items()]
)

# Submit compile job to AI Hub
compile_job = hub.submit_compile_job(
    model=traced_model,
    device=hub.Device("Snapdragon X Elite CRD"),
    options="--target_runtime onnx",
    input_specs=torch_model.get_input_spec(),
)

# Download compiled model
target_model = compile_job.get_target_model()
target_model.download("./models/clip_optimized.onnx")
```

### Phase 3: (Optional) Export BLIP for Image Captioning

```python
# export_blip.py — Export BLIP to ONNX and optimize via QAI Hub
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model.eval()

# Export vision encoder to ONNX
dummy_image = torch.randn(1, 3, 384, 384)
torch.onnx.export(
    model.vision_model,
    dummy_image,
    "blip_vision_encoder.onnx",
    input_names=["pixel_values"],
    output_names=["image_features"],
    dynamic_axes={"pixel_values": {0: "batch"}},
    opset_version=14,
)

# Then compile via Qualcomm AI Hub Workbench
import qai_hub as hub

compile_job = hub.submit_compile_job(
    model="blip_vision_encoder.onnx",
    device=hub.Device("Snapdragon X Elite CRD"),
    options="--target_runtime onnx",
    input_specs={"pixel_values": (1, 3, 384, 384)},
)
compile_job.download_target_model("./models/blip_vision_optimized.onnx")

# The text decoder can run on CPU (it's autoregressive, hard to NPU-accelerate)
```

### Phase 4: Build the On-Device Inference Engine

```python
# inference_engine.py
import onnxruntime as ort
import numpy as np
from PIL import Image

class ImageSummarizer:
    """On-device image summarization using CLIP + descriptive prompts."""

    def __init__(self, model_path="./models/clip_optimized.onnx"):
        # Configure ONNX Runtime with QNN Execution Provider (NPU)
        provider_options = [{
            "backend_path": "QnnHtp.dll",  # Hexagon Tensor Processor
            "htp_performance_mode": "burst",
        }]

        self.session = ort.InferenceSession(
            model_path,
            providers=["QNNExecutionProvider"],
            provider_options=provider_options,
        )

        # Pre-defined descriptive prompts for zero-shot summarization
        self.description_prompts = [
            # Scene types
            "a photograph of a person",
            "a photograph of an animal",
            "a photograph of a building",
            "a photograph of a landscape",
            "a photograph of food",
            "a photograph of text or a document",
            "a photograph of a vehicle",
            "a photograph of indoor furniture",
            # Actions
            "people walking",
            "people sitting",
            "people talking",
            "an object in motion",
            # Attributes
            "something colorful",
            "something dark or shadowy",
            "a close-up detail",
            "a wide scenic view",
        ]

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Resize and normalize image for CLIP (224x224)."""
        image = image.convert("RGB").resize((224, 224), Image.BICUBIC)
        pixel_values = np.array(image, dtype=np.float32) / 255.0

        # CLIP normalization
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        pixel_values = (pixel_values - mean) / std

        # HWC -> CHW -> NCHW
        pixel_values = pixel_values.transpose(2, 0, 1)
        pixel_values = np.expand_dims(pixel_values, 0).astype(np.float32)
        return pixel_values

    def summarize_region(self, image: Image.Image,
                         bbox: tuple) -> str:
        """
        Summarize a cropped region of the image.
        bbox: (x1, y1, x2, y2) — coordinates of the context window
        """
        # Crop to context window
        cropped = image.crop(bbox)

        # Preprocess
        input_tensor = self.preprocess_image(cropped)

        # Run inference on NPU
        # For CLIP: get image embeddings, compare with text prompts
        outputs = self.session.run(None, {
            self.session.get_inputs()[0].name: input_tensor
        })

        image_features = outputs[0]  # Image embedding

        # Compare with text prompts (text encoding can be pre-computed)
        # Build summary from top matching descriptions
        # (Simplified — full CLIP needs text encoder too)
        summary = self._build_summary(image_features, cropped)
        return summary

    def _build_summary(self, features, cropped_image):
        """Build natural language summary from model outputs."""
        # In a full implementation, you'd:
        # 1. Encode all text prompts with CLIP text encoder
        # 2. Compute cosine similarity
        # 3. Select top-k matching descriptions
        # 4. Compose them into a natural summary

        # For hackathon, you might combine:
        # - CLIP similarity scores for scene understanding
        # - YOLOv7 detections for specific objects
        # - TrOCR for any text in the region
        return "Summary placeholder"
```

### Phase 5: Build the Interactive UI

```python
# main_app.py — The complete desktop application
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QGraphicsPixmapItem, QTextEdit,
    QHBoxLayout, QVBoxLayout, QWidget, QPushButton,
    QFileDialog, QLabel, QSplitter
)
from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer
from PyQt6.QtGui import (
    QPixmap, QPen, QColor, QBrush, QImage, QCursor
)
from PIL import Image
from inference_engine import ImageSummarizer


class ResizableRect(QGraphicsRectItem):
    """
    A draggable, resizable rectangle — the "context window".

    Users can:
    - Click and drag to move it
    - Drag corners/edges to resize
    - See real-time visual feedback

    This is the core interactive element of the app.
    """

    HANDLE_SIZE = 10  # pixels for resize handles

    def __init__(self, x, y, w, h, on_change_callback=None):
        super().__init__(x, y, w, h)

        # Visual styling
        self.setPen(QPen(QColor(0, 120, 255), 2, Qt.PenStyle.SolidLine))
        self.setBrush(QBrush(QColor(0, 120, 255, 40)))  # Semi-transparent fill

        # Make it interactive
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        self.setAcceptHoverEvents(True)

        # Resize state
        self._resizing = False
        self._resize_edge = None
        self._start_rect = None
        self._start_pos = None

        # Callback when rect changes (triggers re-summarization)
        self._on_change = on_change_callback

        # Debounce timer — don't re-run inference on every pixel of movement
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(300)  # 300ms debounce
        if on_change_callback:
            self._debounce_timer.timeout.connect(on_change_callback)

    def get_rect_in_scene(self):
        """Return the rectangle coordinates in scene space."""
        r = self.rect()
        pos = self.pos()
        return (
            pos.x() + r.x(),
            pos.y() + r.y(),
            pos.x() + r.x() + r.width(),
            pos.y() + r.y() + r.height()
        )

    def _detect_edge(self, pos):
        """Detect which edge/corner the mouse is near."""
        r = self.rect()
        h = self.HANDLE_SIZE
        edges = []
        if abs(pos.y() - r.top()) < h:    edges.append("top")
        if abs(pos.y() - r.bottom()) < h: edges.append("bottom")
        if abs(pos.x() - r.left()) < h:   edges.append("left")
        if abs(pos.x() - r.right()) < h:  edges.append("right")
        return tuple(edges) if edges else None

    def hoverMoveEvent(self, event):
        """Change cursor based on which edge is hovered."""
        edge = self._detect_edge(event.pos())
        if edge:
            cursors = {
                ("top",):    Qt.CursorShape.SizeVerCursor,
                ("bottom",): Qt.CursorShape.SizeVerCursor,
                ("left",):   Qt.CursorShape.SizeHorCursor,
                ("right",):  Qt.CursorShape.SizeHorCursor,
                ("top", "left"):     Qt.CursorShape.SizeFDiagCursor,
                ("bottom", "right"): Qt.CursorShape.SizeFDiagCursor,
                ("top", "right"):    Qt.CursorShape.SizeBDiagCursor,
                ("bottom", "left"):  Qt.CursorShape.SizeBDiagCursor,
            }
            self.setCursor(QCursor(cursors.get(edge, Qt.CursorShape.SizeAllCursor)))
        else:
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        edge = self._detect_edge(event.pos())
        if edge:
            self._resizing = True
            self._resize_edge = edge
            self._start_rect = self.rect()
            self._start_pos = event.pos()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing:
            delta = event.pos() - self._start_pos
            r = QRectF(self._start_rect)

            if "left" in self._resize_edge:
                r.setLeft(r.left() + delta.x())
            if "right" in self._resize_edge:
                r.setRight(r.right() + delta.x())
            if "top" in self._resize_edge:
                r.setTop(r.top() + delta.y())
            if "bottom" in self._resize_edge:
                r.setBottom(r.bottom() + delta.y())

            # Enforce minimum size
            if r.width() >= 50 and r.height() >= 50:
                self.setRect(r)
                self._debounce_timer.start()
            event.accept()
        else:
            super().mouseMoveEvent(event)
            self._debounce_timer.start()

    def mouseReleaseEvent(self, event):
        if self._resizing:
            self._resizing = False
            self._resize_edge = None
            if self._on_change:
                self._on_change()
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class ImageSummarizerApp(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge AI Image Summarizer")
        self.setMinimumSize(1200, 700)

        # Initialize the AI engine
        self.summarizer = ImageSummarizer()
        self.current_image = None
        self.pixmap_item = None
        self.context_rect = None

        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Top toolbar
        toolbar = QHBoxLayout()
        load_btn = QPushButton("📂 Load Image")
        load_btn.clicked.connect(self._load_image)
        toolbar.addWidget(load_btn)

        summarize_btn = QPushButton("🔍 Summarize Region")
        summarize_btn.clicked.connect(self._run_summarization)
        toolbar.addWidget(summarize_btn)

        self.status_label = QLabel("Load an image to begin")
        toolbar.addWidget(self.status_label)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Main content: image canvas + summary panel
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Image canvas with context window
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(self.view.renderHints())
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        splitter.addWidget(self.view)

        # Right: Summary output
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("Region Summary:"))
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText(
            "Move the context window over a region and click "
            "'Summarize Region'..."
        )
        right_layout.addWidget(self.summary_text)
        splitter.addWidget(right_panel)

        splitter.setSizes([800, 400])
        layout.addWidget(splitter)

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return

        self.current_image = Image.open(path).convert("RGB")
        pixmap = QPixmap(path)

        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

        # Add the context window (default: centered, 200x200)
        iw, ih = pixmap.width(), pixmap.height()
        rw, rh = min(200, iw // 2), min(200, ih // 2)
        rx, ry = (iw - rw) // 2, (ih - rh) // 2

        self.context_rect = ResizableRect(
            rx, ry, rw, rh,
            on_change_callback=None  # or self._run_summarization for auto
        )
        self.scene.addItem(self.context_rect)
        self.status_label.setText(f"Loaded: {path.split('/')[-1]} — drag the blue box")

    def _run_summarization(self):
        if not self.current_image or not self.context_rect:
            return

        self.status_label.setText("⏳ Summarizing...")
        QApplication.processEvents()

        # Get context window coordinates
        bbox = self.context_rect.get_rect_in_scene()

        # Clamp to image bounds
        iw, ih = self.current_image.size
        bbox = (
            max(0, int(bbox[0])),
            max(0, int(bbox[1])),
            min(iw, int(bbox[2])),
            min(ih, int(bbox[3]))
        )

        # Run on-device inference
        summary = self.summarizer.summarize_region(self.current_image, bbox)

        self.summary_text.setText(summary)
        self.status_label.setText("✅ Summary ready (on-device inference)")


def main():
    app = QApplication(sys.argv)
    window = ImageSummarizerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

---

## 5. Inference Strategy — Choosing Your Approach

You have three viable approaches, ranked by complexity and output quality:

### Approach A: CLIP Zero-Shot Classification (Simplest — Recommended for Hackathon)

```
Image Region → CLIP Image Encoder → Image Embedding
                                           ↓
Text Prompts → CLIP Text Encoder → Text Embeddings
                                           ↓
                              Cosine Similarity → Top-K Matches
                                           ↓
                              Template: "This region shows {match1},
                                         {match2}, and {match3}"
```

**Pros:** Fully available on Qualcomm AI Hub. Fast. Simple pipeline.
**Cons:** Output is selected from pre-written prompts, not truly generative.

**Implementation:**
```python
# Pre-compute text embeddings for all description prompts
# At runtime, only the image encoder runs on NPU (fast!)
# Cosine similarity is just a dot product (trivial CPU work)

def summarize_with_clip(image_features, text_features, prompts, top_k=5):
    similarities = np.dot(image_features, text_features.T)
    top_indices = np.argsort(similarities[0])[::-1][:top_k]
    descriptions = [prompts[i] for i in top_indices]
    return "This region contains: " + ", ".join(descriptions)
```

### Approach B: BLIP Image Captioning (Medium Complexity)

```
Image Region → BLIP Vision Encoder (NPU) → Visual Features
                                                  ↓
                         BLIP Text Decoder (CPU) → Generated Caption
```

**Pros:** Generates natural language captions. More impressive output.
**Cons:** Requires exporting BLIP yourself to ONNX. Text decoder runs on CPU.

### Approach C: CLIP + Small LLM Pipeline (Most Impressive)

```
Image Region → CLIP → Labels + Detected Objects + OCR Text
                                    ↓
              Small LLM (Phi-3-mini / Llama 3.2 1B) → Rich Summary
```

**Pros:** Best quality output. Uses multiple Qualcomm AI Hub models.
**Cons:** Most complex. LLM adds latency.

---

## 6. How ONNX Runtime + QNN Execution Provider Works

This is the critical piece that makes everything run on the NPU:

```python
import onnxruntime as ort

# Without QNN EP (runs on CPU — slow)
session_cpu = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

# With QNN EP (runs on Hexagon NPU — fast!)
session_npu = ort.InferenceSession(
    "model.onnx",
    providers=["QNNExecutionProvider"],
    provider_options=[{
        "backend_path": "QnnHtp.dll",        # Hexagon Tensor Processor backend
        "htp_performance_mode": "burst",      # Maximum performance mode
    }]
)

# Same API, dramatically different performance
output = session_npu.run(None, {"input": input_data})
```

**What happens under the hood:**
1. ONNX Runtime loads the model graph
2. QNN EP analyzes which operations can run on the NPU
3. Supported ops are compiled into a QNN context binary (cached for reuse)
4. At inference time, data flows: CPU → NPU → CPU with minimal copies
5. The Hexagon NPU runs quantized matrix operations at high throughput

**For pre-compiled context binaries** (faster startup):
```python
# If you compiled via AI Hub Workbench with --target_runtime precompiled_qnn_onnx
# The model directory contains both .onnx and .bin (context binary)
session = ort.InferenceSession(
    "model_dir.onnx/",  # Directory, not file
    providers=["QNNExecutionProvider"],
    provider_options=[{"backend_path": "QnnHtp.dll"}]
)
```

---

## 7. Model Optimization Pipeline

```
Step 1: Get Model (PyTorch/HuggingFace)
         ↓
Step 2: Export to ONNX (torch.onnx.export)
         ↓
Step 3: Upload to Qualcomm AI Hub Workbench
         ↓
Step 4: Quantize (FP32 → INT8 using calibration data)
         ↓
Step 5: Compile for target device (Snapdragon X Elite)
         ↓
Step 6: Profile on cloud-hosted device (verify latency)
         ↓
Step 7: Download optimized model (.onnx + QNN context binary)
         ↓
Step 8: Deploy locally with ONNX Runtime + QNN EP
```

```python
# Complete pipeline in code:
import qai_hub as hub

# Step 1-2: Already have the ONNX model (or use AI Hub Models)
from qai_hub_models.models.openai_clip import Model
torch_model = Model.from_pretrained()

# Step 3-5: Compile for target
compile_job = hub.submit_compile_job(
    model="clip_image_encoder.onnx",
    device=hub.Device("Snapdragon X Elite CRD"),
    options="--target_runtime precompiled_qnn_onnx",
    input_specs={"pixel_values": (1, 3, 224, 224)},
)

# Step 6: Profile (verify it actually runs fast)
profile_job = hub.submit_profile_job(
    model=compile_job.get_target_model(),
    device=hub.Device("Snapdragon X Elite CRD"),
)
print(profile_job)  # Shows latency, memory, throughput

# Step 7: Download
compile_job.download_target_model("./models/clip_npu/")
```

---

## 8. Project File Structure

```
image-summarizer/
├── main.py                     # App entry point
├── ui/
│   ├── main_window.py          # QMainWindow layout
│   ├── image_canvas.py         # QGraphicsView + QGraphicsScene
│   └── context_rect.py         # ResizableRect widget
├── engine/
│   ├── inference_engine.py     # ONNX Runtime + QNN EP wrapper
│   ├── clip_summarizer.py      # CLIP-based zero-shot summarization
│   ├── blip_captioner.py       # (Optional) BLIP captioning
│   └── preprocessing.py        # Image crop, resize, normalize
├── models/
│   ├── clip_image_encoder/     # Compiled ONNX + QNN context binary
│   ├── clip_text_encoder/      # Pre-computed text embeddings
│   └── prompts.json            # Description prompt bank
├── scripts/
│   ├── export_clip.py          # Export CLIP from AI Hub
│   └── export_blip.py          # Export BLIP to ONNX
├── requirements.txt
└── README.md
```

---

## 9. Key Implementation Tips for the Hackathon

1. **Start with CLIP from AI Hub** — It's pre-optimized and you can get it running in minutes. Avoid spending hours exporting custom models.

2. **Debounce inference calls** — When the user drags the context window, don't run inference on every mouse-move event. Use a 300ms timer that resets on each move.

3. **Pre-compute text embeddings** — CLIP's text encoder only needs to run once (when the app starts). Cache the text embeddings for all your description prompts. Only the image encoder runs per-interaction.

4. **Show the cropped region** — Display a thumbnail of the cropped context window alongside the summary. This helps judges understand what's happening.

5. **Fallback to CPU** — If QNN EP fails (driver issues, unsupported ops), fall back to CPU or DirectML gracefully:
    ```python
    providers = ["QNNExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    ```

6. **Log inference latency** — Show the inference time in the UI. Judges love seeing "Inference: 23ms (NPU)" vs "Inference: 450ms (CPU)".

7. **Multi-model pipeline** — For the best demo, chain CLIP (scene) + YOLOv7 (objects) + TrOCR (text) for a rich summary. All three are on Qualcomm AI Hub.

---

## 10. Summary of Key Technologies

| Technology | Role | Why It Matters |
|-----------|------|---------------|
| **Qualcomm AI Hub** | Model marketplace + optimization platform | Pre-optimized models for Snapdragon; cloud compilation/profiling |
| **OpenAI CLIP** | Vision-language understanding | Zero-shot image understanding without task-specific training |
| **BLIP** | Image captioning | Generates natural language descriptions of images |
| **ONNX Runtime** | Model inference runtime | Cross-platform, supports multiple backends (CPU/GPU/NPU) |
| **QNN Execution Provider** | NPU acceleration | Routes computation to Hexagon NPU for 4x faster inference |
| **QNN SDK (AI Engine Direct)** | Low-level NPU access | The engine under QNN EP; handles quantized tensor operations |
| **PyQt6** | Desktop UI framework | Provides QGraphicsView for interactive, draggable context window |
| **INT8 Quantization** | Model compression | Reduces model from 571MB → ~143MB, faster inference with minimal accuracy loss |

---

*This guide is designed for the Qualcomm Edge AI Developer Hackathon. All AI models run entirely on-device — no cloud required. The Hexagon NPU provides the performance needed for interactive, real-time context window summarization.*
