#!/usr/bin/env python3
"""Train the ipcams2 object classifier that Frigate will load.

Reproduces frigate/util/classification.py exactly, so the .tflite drops into
/config/model_cache/<name>/ and Frigate's own interpreter loads it unchanged:

    base = MobileNetV2(input_shape=(224,224,3), include_top=False,
                       weights='imagenet', alpha=0.35)
    base.trainable = False
    model = Sequential([base, GlobalAveragePooling2D(),
                        Dense(128, 'relu'), Dropout(0.3),
                        Dense(num_classes, 'softmax')])
    ImageDataGenerator(rescale=1/255, validation_split=0.2)

Trained here rather than in the Frigate container: that host is a 6-core
i5-8500 already running fifteen cameras, LPR on the GPU and a TensorRT
detector. The M4 does this in minutes and costs the cameras nothing.
"""
import json, os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA = os.path.expanduser('~/train/ipcams2-crops')
OUT  = os.path.expanduser('~/train/ipcams-classifier')
EPOCHS = int(os.environ.get('EPOCHS', '40'))
BATCH_SIZE = 32
LEARNING_RATE = 0.001

os.makedirs(OUT, exist_ok=True)
tf.random.set_seed(0)
np.random.seed(0)

datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_gen = datagen.flow_from_directory(DATA, target_size=(224,224),
    batch_size=BATCH_SIZE, class_mode='categorical', subset='training', seed=0)
val_gen = datagen.flow_from_directory(DATA, target_size=(224,224),
    batch_size=BATCH_SIZE, class_mode='categorical', subset='validation', seed=0)

num_classes = train_gen.num_classes
print('classes:', train_gen.class_indices)

base = MobileNetV2(input_shape=(224,224,3), include_top=False,
                   weights='imagenet', alpha=0.35)
base.trainable = False
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax'),
])
model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy', metrics=['accuracy'])

cb = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8,
                                       restore_best_weights=True, verbose=1)]
hist = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS,
                 callbacks=cb, verbose=2)

# labelmap.txt: one class per line, ordered by class index, as Frigate writes it.
index_to_class = {v: k for k, v in train_gen.class_indices.items()}
with open(os.path.join(OUT, 'labelmap.txt'), 'w') as f:
    for i in range(num_classes):
        f.write(index_to_class[i] + '\n')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite = converter.convert()
with open(os.path.join(OUT, 'model.tflite'), 'wb') as f:
    f.write(tflite)

best = max(hist.history['val_accuracy'])
print('\nbest val_accuracy: %.4f' % best)
print('wrote', os.path.join(OUT, 'model.tflite'),
      os.path.getsize(os.path.join(OUT, 'model.tflite')), 'bytes')

# Per-class recall on the validation split, which is what actually matters:
# the rare classes are the whole point and overall accuracy hides them.
val_gen.reset()
probs = model.predict(val_gen, verbose=0)
y_true = val_gen.classes[:len(probs)]
y_pred = probs.argmax(axis=1)
print('\n%-9s %6s %6s' % ('class','n','recall'))
for i in range(num_classes):
    m = y_true == i
    n = int(m.sum())
    if n:
        print('%-9s %6d %6.2f' % (index_to_class[i], n, float((y_pred[m]==i).mean())))
