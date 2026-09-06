import atexit
import logging
import os

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

import common
from yolov4_detection import Yolov4ObjectDetection

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


class ONNXTensorRTv4ObjectDetection(Yolov4ObjectDetection):
    """Object Detection class for ONNX Runtime"""

    def __init__(
        self, config, labels
    ):  # , prob_threshold=0.10, model_height=768, model_width=1344, channels=3):
        super(ONNXTensorRTv4ObjectDetection, self).__init__(config, labels)
        model_filename = config.get("onnx")
        engine_file_path = model_filename + ".engine"
        # Use shared CUDA context
        self.cfx = _cuda_context
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
        if os.path.exists(engine_file_path) and os.path.getmtime(
            engine_file_path
        ) > os.path.getmtime(model_filename):
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
