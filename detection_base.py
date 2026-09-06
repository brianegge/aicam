"""Pieces shared by every ONNX detection backend.

The model input pipeline, provider selection and NMS are identical whether the
graph is YOLOv4 (darknet lineage, boxes/confs output pair) or Ultralytics
YOLO11 (single [1, 4+nc, N] tensor). Only the output decoding differs, so that
is all a backend has to implement.
"""

import logging

import numpy as np
from PIL import Image

from object_detection import ObjectDetection

logger = logging.getLogger(__name__)

# Providers to try, in order, when the config does not name any.
DEFAULT_PROVIDERS = ("CoreMLExecutionProvider", "CPUExecutionProvider")


def resolve_providers(requested, available):
    """Pick the execution providers to use, preserving the requested order.

    Unavailable providers are dropped rather than fatal, so a config written for
    a machine with CoreML still starts on one without it.
    """
    if requested:
        chosen = [p for p in requested if p in available]
        missing = [p for p in requested if p not in available]
        if missing:
            logger.warning(
                "Ignoring unavailable execution providers: %s", ", ".join(missing)
            )
    else:
        chosen = [p for p in DEFAULT_PROVIDERS if p in available]
    if not chosen:
        chosen = ["CPUExecutionProvider"]
    return chosen


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # logger.debug(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


class ConfiguredInputObjectDetection(ObjectDetection):
    """Detector whose input geometry comes from a config section.

    camera.py hands every model a 608x608 RGB array, so preprocessing is just a
    channels-first float conversion; the resize is a no-op unless a model is
    configured at some other size.
    """

    def __init__(self, config, labels):
        super(ConfiguredInputObjectDetection, self).__init__(
            labels, float(config.get("prob_threshold"))
        )
        self.model_width = int(config.get("width"))
        self.model_height = int(config.get("height"))
        self.channels = int(config.get("channels"))
        self.is_fp16 = False

    def preprocess(self, image):
        if isinstance(image, Image.Image):
            if image.size != (self.model_width, self.model_height):
                logger.debug(
                    "Resizing from {} to {}".format(
                        image.size, (self.model_width, self.model_height)
                    )
                )
                image = image.resize(
                    (self.model_width, self.model_height), Image.BILINEAR
                )
        img_in = np.array(image)
        if self.channels == 3:
            # channels first
            img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        else:
            img_in = img_in.astype(np.float32)
            # add channel dimension
            img_in = np.expand_dims(img_in, axis=0)
        # add batch dimension
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        img_in = np.ascontiguousarray(img_in)
        return img_in
