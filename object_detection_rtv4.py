import atexit
import logging
import os

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image

import common
from object_detection import ObjectDetection

TRT_LOGGER = trt.Logger()
logger = logging.getLogger(__name__)

# Initialize CUDA once and share context across all models
cuda.init()
_cuda_device = cuda.Device(0)
_cuda_context = _cuda_device.make_context()


def _cleanup_cuda():
    """Clean up CUDA context at exit."""
    global _cuda_context
    if _cuda_context is not None:
        try:
            _cuda_context.pop()
        except cuda.LogicError:
            pass  # Context already popped or invalid
        _cuda_context = None


atexit.register(_cleanup_cuda)


class ONNXTensorRTv4ObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime"""

    def __init__(
        self, config, labels
    ):  # , prob_threshold=0.10, model_height=768, model_width=1344, channels=3):
        super(ONNXTensorRTv4ObjectDetection, self).__init__(
            labels, float(config.get("prob_threshold"))
        )
        self.model_width = int(config.get("width"))
        self.model_height = int(config.get("height"))
        self.channels = int(config.get("channels"))
        model_filename = config.get("onnx")
        engine_file_path = model_filename + ".engine"
        # Use shared CUDA context
        self.cfx = _cuda_context
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
        if os.path.exists(engine_file_path) and os.path.getctime(
            engine_file_path
        ) > os.path.getctime(model_filename):
            # If a serialized engine exists, use it instead of building an engine.
            logger.info(
                "Reading engine from file {} for classes {}".format(
                    engine_file_path, ",".join(labels)
                )
            )
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        else:
            logger.info("Compiling model {}".format(os.path.basename(model_filename)))
            self.engine = self.get_engine(model_filename, engine_file_path)
        self.is_fp16 = False  # network.get_input(0).type == 'tensor(float16)'
        self.input_name = "input"  # network.get_input(0).name
        self.context = self.engine.create_execution_context()
        self.context.set_binding_shape(
            0, (1, self.channels, self.model_height, self.model_width)
        )

    def __del__(self):
        # Clean up TensorRT resources; shared CUDA context is cleaned up at exit
        try:
            del self.context
            del self.engine
        except (AttributeError, cuda.LogicError):
            pass  # Already cleaned up or context invalid

    def get_engine(self, onnx_file_path, engine_file_path):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            # builder.max_workspace_size = 1 << 28  # 256MiB
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 20
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                logger.warning(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(
                        onnx_file_path
                    )
                )
                exit(0)
            logger.info("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                logger.info("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    logger.error("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return None
            logger.info(
                "Creating model with shape {},{},{}".format(
                    self.channels, self.model_height, self.model_width
                )
            )
            network.get_input(0).shape = [
                1,
                self.channels,
                self.model_height,
                self.model_width,
            ]  # NCWH
            logger.info("Completed parsing of ONNX file")
            logger.info(
                "Building an engine from file {}; this may take a while...".format(
                    onnx_file_path
                )
            )
            plan = builder.build_serialized_network(network, config)
            with trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(plan)
            # engine = builder.build_cuda_engine(network)
            if engine:
                logger.info("Completed creating Engine")
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    def preprocess(self, image):
        # opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        # resized = cv2.resize(opencv_image, (self.model_width, self.model_height), interpolation=cv2.INTER_LINEAR)
        # img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
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
        # logger.debug("Shape of the network input: ", img_in.shape)
        return img_in

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

        self.cfx.push()
        try:
            inputs, outputs, bindings, stream = common.allocate_buffers(self.engine)
            # Do inference
            inputs[0].host = np_image
            trt_outputs = do_inference(
                self.context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
            )
        finally:
            self.cfx.pop()  # very important
        # logger.debug('Len of outputs: ', len(trt_outputs))
        num_classes = len(self.labels)
        trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
        trt_outputs[1] = trt_outputs[1].reshape(1, -1, num_classes)
        return trt_outputs

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

    def post_processing(self, conf_thresh, nms_thresh, output):

        # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
        # num_anchors = 9
        # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        # strides = [8, 16, 32]
        # anchor_step = len(anchors) // num_anchors

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


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # prediction_start = timer()
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # prediction_time = timer() - prediction_start
    # logger.info("Inference in {: 0.3f}", prediction_time)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


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
