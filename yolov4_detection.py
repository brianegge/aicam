"""Backend-independent YOLOv4 detection.

The models aicam ships (``ipcams_color``, ``ipcams_grey``, ``vehicles``) emit the
YOLOv4 output pair ``boxes[1, N, 1, 4]`` / ``confs[1, N, num_classes]``, which is
not what the Custom Vision postprocess in :mod:`object_detection` expects. The
box decoding lives here so the TensorRT backend (Jetson) and the ONNX Runtime
backend (anything else) stay byte-for-byte identical in how they turn model
output into predictions.
"""

import logging

import numpy as np
from PIL import Image

from object_detection import ObjectDetection

logger = logging.getLogger(__name__)

# Providers to try, in order, when the config does not name any.
DEFAULT_PROVIDERS = ("CoreMLExecutionProvider", "CPUExecutionProvider")


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


class Yolov4ObjectDetection(ObjectDetection):
    """Shared preprocess/postprocess for the YOLOv4 models.

    Subclasses only have to implement :meth:`predict`, returning the raw
    ``[boxes, confs]`` pair shaped ``(1, N, 1, 4)`` and ``(1, N, num_classes)``.
    """

    def __init__(self, config, labels):
        super(Yolov4ObjectDetection, self).__init__(
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

    def post_processing(self, conf_thresh, nms_thresh, output):
        # [batch, num, 1, 4]
        box_array = output[0]
        num_classes = len(self.labels)
        # [batch, num, num_classes]
        confs = output[1]

        if type(box_array).__name__ != "ndarray":
            box_array = box_array.cpu().detach().numpy()
            confs = confs.cpu().detach().numpy()

        assert num_classes == confs.shape[2]

        # [batch, num, 4]
        box_array = box_array[:, :, 0]

        # [batch, num, num_classes] --> [batch, num]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)

        bboxes_batch = []
        for i in range(box_array.shape[0]):
            argwhere = max_conf[i] > conf_thresh
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]

            bboxes = []
            # nms for each class
            for j in range(num_classes):
                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]

                keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

                if keep.size > 0:
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]

                    for k in range(ll_box_array.shape[0]):
                        bboxes.append(
                            [
                                ll_box_array[k, 0],
                                ll_box_array[k, 1],
                                ll_box_array[k, 2],
                                ll_box_array[k, 3],
                                ll_max_conf[k],
                                ll_max_conf[k],
                                ll_max_id[k],
                            ]
                        )

            bboxes_batch.append(bboxes)

        assert (
            len(bboxes_batch) == 1
        ), "We only expect to be doing one batch at a time now"

        return bboxes_batch[0]

    def postprocess(self, prediction_outputs):
        """Extract bounding boxes from the model outputs.

        Args:
            prediction_outputs: Output from the object detection model. (H x W x C)
        """
        selected_boxes = self.post_processing(0.4, 0.6, prediction_outputs)

        return [
            {
                "probability": round(float(selected_boxes[i][4]), 8),
                "tagId": int(selected_boxes[i][6]),
                "tagName": self.labels[selected_boxes[i][6]],
                "boundingBox": {
                    "left": round(float(selected_boxes[i][0]), 8),
                    "top": round(float(selected_boxes[i][1]), 8),
                    "width": round(
                        float(selected_boxes[i][2]) - float(selected_boxes[i][0]), 8
                    ),
                    "height": round(
                        float(selected_boxes[i][3]) - float(selected_boxes[i][1]), 8
                    ),
                },
            }
            for i in range(len(selected_boxes))
        ]


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


class ONNXRuntimeYolov4ObjectDetection(Yolov4ObjectDetection):
    """YOLOv4 inference through ONNX Runtime.

    Used everywhere TensorRT is unavailable. On Apple silicon the CoreML
    provider runs the graph on the ANE/GPU; note it computes in fp16, so
    confidences can differ from the CPU provider in the third decimal place.
    """

    def __init__(self, config, labels):
        super(ONNXRuntimeYolov4ObjectDetection, self).__init__(config, labels)
        import onnxruntime

        model_filename = config.get("onnx")
        requested = [
            p.strip() for p in config.get("providers", "").split(",") if p.strip()
        ]
        providers = resolve_providers(requested, onnxruntime.get_available_providers())
        logger.info(
            "Loading %s with providers %s for classes %s",
            model_filename,
            ",".join(providers),
            ",".join(labels),
        )
        self.session = onnxruntime.InferenceSession(model_filename, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.is_fp16 = self.session.get_inputs()[0].type == "tensor(float16)"

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

        outputs = self.session.run(None, {self.input_name: np_image})
        num_classes = len(self.labels)
        return [
            outputs[0].reshape(1, -1, 1, 4),
            outputs[1].reshape(1, -1, num_classes),
        ]
