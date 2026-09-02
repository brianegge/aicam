"""Ultralytics YOLO (v8/v11) detection through ONNX Runtime.

Darknet has no GPU path on Apple silicon, so models retrained on claw-mini are
trained with Ultralytics instead. Their exported graph is shaped differently
from the darknet-lineage YOLOv4 models:

- one output ``[1, 4 + num_classes, N]`` rather than a ``boxes``/``confs`` pair
- boxes as ``cx, cy, w, h`` in **input pixels**, not normalised corners
- no separate objectness score; the class score is the confidence

Everything before and after that (the 608x608 input, NMS, and the prediction
dicts detect.py consumes) is shared, so the two backends are interchangeable
per model via ``backend=`` in the model's config section.
"""

import logging

import numpy as np

from detection_base import ConfiguredInputObjectDetection, nms_cpu, resolve_providers

logger = logging.getLogger(__name__)

# Matching the YOLOv4 path, which hardcodes these rather than using
# prob_threshold, so swapping a model does not silently change how much gets
# through to detect.py's own per-class thresholds.
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.6


class ONNXRuntimeUltralyticsObjectDetection(ConfiguredInputObjectDetection):
    """Ultralytics YOLO inference through ONNX Runtime."""

    def __init__(self, config, labels):
        super(ONNXRuntimeUltralyticsObjectDetection, self).__init__(config, labels)
        import onnxruntime

        model_filename = config.get("onnx")
        requested = [
            p.strip() for p in config.get("providers", "").split(",") if p.strip()
        ]
        providers = resolve_providers(requested, onnxruntime.get_available_providers())
        logger.info(
            "Loading %s (ultralytics) with providers %s for classes %s",
            model_filename,
            ",".join(providers),
            ",".join(labels),
        )
        self.session = onnxruntime.InferenceSession(model_filename, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.is_fp16 = self.session.get_inputs()[0].type == "tensor(float16)"
        self.conf_threshold = float(config.get("conf_threshold", CONF_THRESHOLD))
        self.nms_threshold = float(config.get("nms_threshold", NMS_THRESHOLD))

    def predict(self, preprocessed_image):
        np_image = preprocessed_image
        assert (
            1,
            self.channels,
            self.model_height,
            self.model_width,
        ) == np_image.shape, "Image must be resized to model shape"

        if self.is_fp16:
            np_image = np_image.astype(np.float16)

        return self.session.run(None, {self.input_name: np_image})

    def postprocess(self, prediction_outputs):
        """Decode ``[1, 4 + num_classes, N]`` into prediction dicts."""
        raw = np.asarray(prediction_outputs[0], dtype=np.float32)
        # [1, 4+nc, N] -> [N, 4+nc]
        predictions = np.squeeze(raw, axis=0).T
        num_classes = len(self.labels)
        assert predictions.shape[1] == 4 + num_classes, (
            "Model has %d outputs per box, expected %d for classes %s"
            % (predictions.shape[1], 4 + num_classes, ",".join(self.labels))
        )

        scores = predictions[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        keep = confidences > self.conf_threshold
        if not np.any(keep):
            return []
        boxes = predictions[keep, :4]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        # cx, cy, w, h in input pixels -> normalised x1, y1, x2, y2, which is
        # what nms_cpu and the prediction dicts both work in.
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - w / 2) / self.model_width
        y1 = (cy - h / 2) / self.model_height
        x2 = (cx + w / 2) / self.model_width
        y2 = (cy + h / 2) / self.model_height
        corners = np.stack((x1, y1, x2, y2), axis=1)

        results = []
        for class_id in range(num_classes):
            mask = class_ids == class_id
            if not np.any(mask):
                continue
            cls_boxes = corners[mask]
            cls_conf = confidences[mask]
            for i in nms_cpu(cls_boxes, cls_conf, self.nms_threshold):
                box = cls_boxes[i]
                left = float(np.clip(box[0], 0.0, 1.0))
                top = float(np.clip(box[1], 0.0, 1.0))
                right = float(np.clip(box[2], 0.0, 1.0))
                bottom = float(np.clip(box[3], 0.0, 1.0))
                results.append(
                    {
                        "probability": round(float(cls_conf[i]), 8),
                        "tagId": int(class_id),
                        "tagName": self.labels[class_id],
                        "boundingBox": {
                            "left": round(left, 8),
                            "top": round(top, 8),
                            "width": round(right - left, 8),
                            "height": round(bottom - top, 8),
                        },
                    }
                )

        results.sort(key=lambda p: p["probability"], reverse=True)
        return results
