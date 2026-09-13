#!/usr/bin/env python3
"""Re-export the ipcams classifier as a fully INT8-quantized tflite.

The previous export used a plain TFLiteConverter.from_keras_model(), which
keeps a float32 input. Frigate hands the classifier the raw crop straight from
cv2:

    input = np.expand_dims(resized_crop, axis=0)   # uint8
    self.interpreter.set_tensor(..., input)

so it threw "Got value of type UINT8 but expected type FLOAT32" on the first
animal -- and because that runs inside embeddings_maintainer, the exception
killed the thread that also does LPR. Plates stopped for ~4.5 hours.

This reproduces frigate/util/classification.py exactly: full integer
quantization, uint8 in AND out, with a representative dataset built the same
way (BGR->RGB, resize 224, /255 float32).
"""
import os, glob, sys
import numpy as np, tensorflow as tf, cv2

SRC   = os.path.expanduser(os.environ.get('SRC_MODEL','~/train/ipcams-clf-grouped'))
DATA  = os.path.expanduser(os.environ.get('DATA_DIR','~/train/crops-grouped/train'))
OUT   = os.path.expanduser(os.environ.get('OUT_DIR','~/train/ipcams-clf-uint8'))
os.makedirs(OUT, exist_ok=True)

# The trained keras model was not saved separately, so rebuild it from the
# float tflite? No -- retrain is unnecessary: we saved only the tflite. Load
# the SavedModel if present, else rebuild architecture and load weights.
keras_path = os.path.join(SRC, 'model.keras')
if not os.path.exists(keras_path):
    print('ERROR: %s not found -- the training run must save the keras model' % keras_path)
    sys.exit(2)
model = tf.keras.models.load_model(keras_path)

def representative():
    paths = []
    for root, _, files in os.walk(DATA):
        for f in files:
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                paths.append(os.path.join(root,f))
    paths.sort()
    step = max(1, len(paths)//300)
    for p in paths[::step][:300]:
        img = cv2.imread(p)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(224,224))
        yield [np.array(img, dtype=np.float32)[None,...] / 255.0]

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.uint8
conv.inference_output_type = tf.uint8
tfl = conv.convert()
open(os.path.join(OUT,'model.tflite'),'wb').write(tfl)
import shutil; shutil.copy(os.path.join(SRC,'labelmap.txt'), os.path.join(OUT,'labelmap.txt'))

it = tf.lite.Interpreter(model_path=os.path.join(OUT,'model.tflite')); it.allocate_tensors()
i0, o0 = it.get_input_details()[0], it.get_output_details()[0]
print('exported %s (%d bytes)' % (os.path.join(OUT,'model.tflite'), len(tfl)))
print('  input : %s %s' % (i0['shape'], i0['dtype'].__name__))
print('  output: %s %s' % (o0['shape'], o0['dtype'].__name__))
assert i0['dtype'] == np.uint8, 'input is not uint8'
print('  input dtype is uint8 -- matches what Frigate passes')
