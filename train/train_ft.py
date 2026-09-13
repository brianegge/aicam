#!/usr/bin/env python3
"""alpha=1.0 MobileNetV2 with fine-tuning, for the ipcams2 object classifier.

Frigate loads a .tflite through its own Interpreter and only requires a
224x224x3 input and an N-class softmax matching labelmap.txt -- it does not
care how the model was trained. So we are not bound to the frozen alpha=0.35
backbone its built-in trainer uses, which topped out at 0.68 val accuracy and
0.20 recall on raccoon.

Two phases, which is what makes fine-tuning work rather than destroy the
pretrained features:
  1. frozen base, train the head only, so the randomly-initialised head does
     not push large gradients back into good ImageNet weights.
  2. unfreeze the top blocks at a 10x lower learning rate.

BatchNorm layers stay frozen throughout: their running statistics are computed
over ImageNet, and updating them on 3.5k small crops in batches of 32 is a
well-known way to make fine-tuning worse rather than better.
"""
import os, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA = os.path.expanduser(os.environ.get('DATA_DIR','~/train/ipcams2-crops'))
OUT  = os.path.expanduser(os.environ.get('OUT_DIR','~/train/ipcams-classifier-a10'))
HEAD_EPOCHS = int(os.environ.get('HEAD_EPOCHS','8'))
FT_EPOCHS   = int(os.environ.get('FT_EPOCHS','30'))
UNFREEZE_FROM = int(os.environ.get('UNFREEZE_FROM','100'))

os.makedirs(OUT, exist_ok=True)
tf.random.set_seed(0); np.random.seed(0)

dg = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
tr = dg.flow_from_directory(DATA, target_size=(224,224), batch_size=32,
                            class_mode='categorical', subset='training', seed=0)
va = dg.flow_from_directory(DATA, target_size=(224,224), batch_size=32,
                            class_mode='categorical', subset='validation', seed=0)
nc = tr.num_classes
print('classes:', tr.class_indices, flush=True)

base = MobileNetV2(input_shape=(224,224,3), include_top=False,
                   weights='imagenet', alpha=1.0)
base.trainable = False
model = models.Sequential([
    base, layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'), layers.Dropout(0.3),
    layers.Dense(nc, activation='softmax'),
])
model.compile(optimizer=optimizers.Adam(1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])
print('--- phase 1: head only ---', flush=True)
h1 = model.fit(tr, validation_data=va, epochs=HEAD_EPOCHS, verbose=2)

print('--- phase 2: fine-tune from layer %d ---' % UNFREEZE_FROM, flush=True)
base.trainable = True
for l in base.layers[:UNFREEZE_FROM]:
    l.trainable = False
for l in base.layers:
    if isinstance(l, layers.BatchNormalization):
        l.trainable = False
trainable = sum(1 for l in base.layers if l.trainable)
print('trainable base layers: %d of %d' % (trainable, len(base.layers)), flush=True)

model.compile(optimizer=optimizers.Adam(1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])
cb = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6,
                                       restore_best_weights=True, verbose=1)]
h2 = model.fit(tr, validation_data=va, epochs=FT_EPOCHS, callbacks=cb, verbose=2)

idx = {v:k for k,v in tr.class_indices.items()}
with open(os.path.join(OUT,'labelmap.txt'),'w') as f:
    for i in range(nc): f.write(idx[i]+'\n')
tfl = tf.lite.TFLiteConverter.from_keras_model(model).convert()
with open(os.path.join(OUT,'model.tflite'),'wb') as f: f.write(tfl)

print('\nphase1 best val_acc: %.4f' % max(h1.history['val_accuracy']))
print('phase2 best val_acc: %.4f' % max(h2.history['val_accuracy']))
print('wrote', os.path.join(OUT,'model.tflite'),
      os.path.getsize(os.path.join(OUT,'model.tflite')), 'bytes')
