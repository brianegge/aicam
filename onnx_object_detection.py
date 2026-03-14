import os
import tempfile

import numpy as np
import onnx
import onnxruntime

from object_detection import ObjectDetection


class ONNXRuntimeObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime"""

    def __init__(self, model_filename, labels, prob_threshold=0.10, scale=4):
        super(ONNXRuntimeObjectDetection, self).__init__(
            labels, prob_threshold=prob_threshold, scale=scale
        )
        model = onnx.load(model_filename)
        with tempfile.TemporaryDirectory() as dirpath:
            temp = os.path.join(dirpath, os.path.basename(model_filename))
            model.graph.input[0].type.tensor_type.shape.dim[-1].dim_param = "dim1"
            model.graph.input[0].type.tensor_type.shape.dim[-2].dim_param = "dim2"
            onnx.save(model, temp)
            self.session = onnxruntime.InferenceSession(temp)
        self.input_name = self.session.get_inputs()[0].name
        self.is_fp16 = self.session.get_inputs()[0].type == "tensor(float16)"

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[
            np.newaxis, :, :, (2, 1, 0)
        ]  # RGB -> BGR
        inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))

        if self.is_fp16:
            inputs = inputs.astype(np.float16)

        outputs = self.session.run(None, {self.input_name: inputs})
        return np.squeeze(outputs).transpose((1, 2, 0)).astype(np.float32)
