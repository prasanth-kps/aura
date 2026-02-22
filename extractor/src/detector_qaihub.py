from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch


COCO_80_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


class QualcommYoloDetector:
    """
    Qualcomm-first detector:
    - Primary: AI Hub YOLOv11-Detection
    - Fallback: AI Hub YOLOv8-Detection
    """

    def __init__(
        self,
        prefer_model: str = "yolov11",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.60,
        runtime: str = "auto",
        input_size: int = 960,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = int(input_size)
        self.device = torch.device("cpu")
        self.runtime = runtime
        self.model_name = ""
        self.model = self._load(prefer_model)
        self.model.to(self.device)
        self.model.eval()
        self.backend = "cpu"
        self.ort_session = None
        self.ort_input_name = "images"

        if runtime not in {"auto", "cpu", "qnn"}:
            raise ValueError(f"Unsupported runtime '{runtime}'. Use one of: auto, cpu, qnn.")
        if self.input_size < 320 or self.input_size > 1280:
            raise ValueError("input_size must be in range [320, 1280].")

        if runtime in {"auto", "qnn"}:
            self._try_enable_qnn()
            if runtime == "qnn" and self.backend != "qnn":
                raise RuntimeError(
                    "Runtime 'qnn' was requested but QNNExecutionProvider could not be initialized."
                )

    def _load(self, prefer_model: str):
        order = ["yolov11", "yolov8"] if prefer_model == "yolov11" else ["yolov8", "yolov11"]
        errors = {}

        for model_name in order:
            try:
                if model_name == "yolov11":
                    from qai_hub_models.models.yolov11_det.model import YoloV11Detector

                    model = YoloV11Detector.from_pretrained(include_postprocessing=True)
                else:
                    from qai_hub_models.models.yolov8_det.model import YoloV8Detector

                    model = YoloV8Detector.from_pretrained(include_postprocessing=True)

                self.model_name = model_name
                return model
            except Exception as exc:  # noqa: BLE001
                errors[model_name] = str(exc)

        raise RuntimeError(f"Unable to load Qualcomm detector models: {errors}")

    @staticmethod
    def _label_from_class_id(class_id: int) -> str:
        if 0 <= class_id < len(COCO_80_LABELS):
            return COCO_80_LABELS[class_id]
        return f"class_{class_id}"

    def _onnx_export_path(self) -> Path:
        cache_dir = Path.home() / ".aura_cache" / "onnx"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{self.model_name}_det_post_{self.input_size}.onnx"

    def _try_enable_qnn(self) -> None:
        try:
            import onnxruntime as ort
        except Exception:
            return

        providers = ort.get_available_providers()
        if "QNNExecutionProvider" not in providers:
            return

        onnx_path = self._onnx_export_path()
        if not onnx_path.exists():
            dummy = torch.randn(1, 3, self.input_size, self.input_size)
            torch.onnx.export(
                self.model,
                dummy,
                str(onnx_path),
                input_names=["images"],
                output_names=["boxes", "scores", "classes"],
                opset_version=17,
            )

        session = ort.InferenceSession(
            str(onnx_path),
            providers=["QNNExecutionProvider", "CPUExecutionProvider"],
        )
        self.ort_session = session
        self.ort_input_name = session.get_inputs()[0].name
        self.backend = "qnn"

    def _class_aware_nms(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
    ) -> list[int]:
        keep_global: list[int] = []

        unique_classes = np.unique(classes.astype(np.int32))
        for cls in unique_classes:
            idxs = np.where(classes == cls)[0]
            if idxs.size == 0:
                continue

            cls_boxes = boxes_xyxy[idxs]
            cls_scores = scores[idxs].astype(float).tolist()

            # cv2.dnn.NMSBoxes expects x,y,w,h.
            cls_boxes_xywh = []
            for box in cls_boxes:
                x1, y1, x2, y2 = box.tolist()
                cls_boxes_xywh.append(
                    [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]
                )

            raw_keep = cv2.dnn.NMSBoxes(
                bboxes=cls_boxes_xywh,
                scores=cls_scores,
                score_threshold=self.conf_threshold,
                nms_threshold=self.iou_threshold,
            )

            if raw_keep is None or len(raw_keep) == 0:
                continue

            if isinstance(raw_keep, np.ndarray):
                local_keep = raw_keep.flatten().tolist()
            else:
                local_keep = []
                for item in raw_keep:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        local_keep.append(int(item[0]))
                    else:
                        local_keep.append(int(item))

            for local_idx in local_keep:
                if 0 <= local_idx < len(idxs):
                    keep_global.append(int(idxs[local_idx]))

        return sorted(set(keep_global))

    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        scale_x = w / float(self.input_size)
        scale_y = h / float(self.input_size)

        if self.ort_session is not None:
            inp_np = np.transpose(resized_rgb, (2, 0, 1)).astype(np.float32)[None, ...] / 255.0
            try:
                boxes, scores, classes = self.ort_session.run(
                    None, {self.ort_input_name: inp_np}
                )
                boxes = boxes[0]
                scores = scores[0]
                classes = classes[0].astype(np.int32)
            except Exception as exc:
                if self.runtime == "qnn":
                    raise RuntimeError(f"QNN inference failed: {exc}") from exc
                # Auto fallback to CPU path.
                self.ort_session = None
                self.backend = "cpu"
                inp = torch.from_numpy(resized_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                inp = inp.to(self.device)
                with torch.no_grad():
                    boxes_t, scores_t, classes_t = self.model(inp)
                boxes = boxes_t[0].detach().cpu().numpy()
                scores = scores_t[0].detach().cpu().numpy()
                classes = classes_t[0].detach().cpu().numpy().astype(np.int32)
        else:
            inp = torch.from_numpy(resized_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            inp = inp.to(self.device)

            with torch.no_grad():
                try:
                    boxes_t, scores_t, classes_t = self.model(inp)
                except RuntimeError as exc:
                    # One automatic fallback at runtime if the primary model path fails.
                    if self.model_name == "yolov11":
                        self.model = self._load("yolov8")
                        self.model.to(self.device)
                        self.model.eval()
                        boxes_t, scores_t, classes_t = self.model(inp)
                    else:
                        raise RuntimeError(
                            f"Inference failed for model '{self.model_name}': {exc}"
                        ) from exc

            boxes = boxes_t[0].detach().cpu().numpy()
            scores = scores_t[0].detach().cpu().numpy()
            classes = classes_t[0].detach().cpu().numpy().astype(np.int32)

        if boxes.size == 0:
            return []

        pre_keep = np.where(scores >= self.conf_threshold)[0]
        if pre_keep.size == 0:
            return []

        boxes = boxes[pre_keep]
        scores = scores[pre_keep]
        classes = classes[pre_keep]

        nms_keep = self._class_aware_nms(boxes, scores, classes)
        if not nms_keep:
            return []

        out: list[Detection] = []
        for idx in nms_keep:
            x1, y1, x2, y2 = boxes[idx].tolist()
            x1 = max(0, min(w - 1, int(round(x1 * scale_x))))
            y1 = max(0, min(h - 1, int(round(y1 * scale_y))))
            x2 = max(0, min(w - 1, int(round(x2 * scale_x))))
            y2 = max(0, min(h - 1, int(round(y2 * scale_y))))
            if x2 <= x1 or y2 <= y1:
                continue

            cls_id = int(classes[idx])
            label = self._label_from_class_id(cls_id)

            out.append(
                Detection(
                    label=label,
                    confidence=float(scores[idx]),
                    bbox=(x1, y1, x2, y2),
                )
            )

        return out
