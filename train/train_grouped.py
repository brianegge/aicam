#!/usr/bin/env python3
"""Train the ipcams2 classifier on the event-grouped split, deer prioritised.

Differences from the earlier runs:
  - train/ and val/ are pre-split BY EVENT GROUP, so no frame of an event can
    appear on both sides. (Measured: ipcams2 has essentially no near-duplicate
    frames -- 0 identical pairs, median dHash distance 32 -- so this confirms
    the split rather than repairing it.)
  - the validation set is NOT oversampled; only train/ carries variants, so
    val accuracy is measured on real crops, one per box.
  - class_weight prioritises deer, which is the class that matters here: deer
    eat the trees. Weights are inverse-frequency, then deer is scaled by
    DEER_WEIGHT on top.
"""
import os, collections, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT=os.path.expanduser('~/train/crops-grouped')
OUT =os.path.expanduser(os.environ.get('OUT_DIR','~/train/ipcams-clf-grouped'))
DEER_WEIGHT=float(os.environ.get('DEER_WEIGHT','2.0'))
HEAD_EPOCHS=int(os.environ.get('HEAD_EPOCHS','8'))
FT_EPOCHS=int(os.environ.get('FT_EPOCHS','30'))
UNFREEZE_FROM=int(os.environ.get('UNFREEZE_FROM','100'))
os.makedirs(OUT,exist_ok=True); tf.random.set_seed(0); np.random.seed(0)

dg=ImageDataGenerator(rescale=1.0/255)
tr=dg.flow_from_directory(os.path.join(ROOT,'train'),target_size=(224,224),
    batch_size=32,class_mode='categorical',seed=0)
va=dg.flow_from_directory(os.path.join(ROOT,'val'),target_size=(224,224),
    batch_size=32,class_mode='categorical',seed=0,shuffle=False)
nc=tr.num_classes
print('classes:',tr.class_indices,flush=True)

cnt=collections.Counter(tr.classes)
total=sum(cnt.values())
cw={i: total/(nc*cnt[i]) for i in range(nc)}
deer_i=tr.class_indices.get('deer')
if deer_i is not None: cw[deer_i]*=DEER_WEIGHT
print('class weights:',{list(tr.class_indices)[i]:round(w,3) for i,w in cw.items()},flush=True)

base=MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet',alpha=1.0)
base.trainable=False
model=models.Sequential([base,layers.GlobalAveragePooling2D(),
    layers.Dense(128,activation='relu'),layers.Dropout(0.3),
    layers.Dense(nc,activation='softmax')])
model.compile(optimizer=optimizers.Adam(1e-3),loss='categorical_crossentropy',metrics=['accuracy'])
print('--- phase 1: head ---',flush=True)
h1=model.fit(tr,validation_data=va,epochs=HEAD_EPOCHS,class_weight=cw,verbose=2)

base.trainable=True
for l in base.layers[:UNFREEZE_FROM]: l.trainable=False
for l in base.layers:
    if isinstance(l,layers.BatchNormalization): l.trainable=False
model.compile(optimizer=optimizers.Adam(1e-4),loss='categorical_crossentropy',metrics=['accuracy'])
print('--- phase 2: fine-tune ---',flush=True)
cb=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=6,restore_best_weights=True,verbose=1)]
h2=model.fit(tr,validation_data=va,epochs=FT_EPOCHS,class_weight=cw,callbacks=cb,verbose=2)

idx={v:k for k,v in tr.class_indices.items()}
with open(os.path.join(OUT,'labelmap.txt'),'w') as f:
    for i in range(nc): f.write(idx[i]+'\n')

# Keep the keras model: the first run saved only the tflite, so re-exporting
# later was impossible without a full retrain.
model.save(os.path.join(OUT,'model.keras'))

# Fully INT8 quantized, uint8 in AND out -- what frigate/util/classification.py
# produces. A plain from_keras_model() export keeps a float32 input, but
# Frigate passes the raw cv2 crop:
#     input = np.expand_dims(resized_crop, axis=0)   # uint8
# which raised "Got value of type UINT8 but expected type FLOAT32" inside
# embeddings_maintainer, killing the thread that also runs LPR.
import cv2
def _representative():
    paths=[]
    for root,_,files in os.walk(os.path.join(ROOT,'train')):
        for f in files:
            if f.lower().endswith(('.jpg','.jpeg','.png')): paths.append(os.path.join(root,f))
    paths.sort(); step=max(1,len(paths)//300)
    for pth in paths[::step][:300]:
        img=cv2.imread(pth)
        if img is None: continue
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB); img=cv2.resize(img,(224,224))
        yield [np.array(img,dtype=np.float32)[None,...]/255.0]

conv=tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations=[tf.lite.Optimize.DEFAULT]
conv.representative_dataset=_representative
conv.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type=tf.uint8
conv.inference_output_type=tf.uint8
open(os.path.join(OUT,'model.tflite'),'wb').write(conv.convert())

_it=tf.lite.Interpreter(model_path=os.path.join(OUT,'model.tflite')); _it.allocate_tensors()
_i=_it.get_input_details()[0]; _o=_it.get_output_details()[0]
print('tflite input : %s %s' % (_i['shape'], _i['dtype'].__name__))
print('tflite output: %s %s' % (_o['shape'], _o['dtype'].__name__))
assert _i['dtype']==np.uint8, 'input must be uint8 -- Frigate passes a raw crop'
print('input dtype verified uint8')
print('\nphase1 best val_acc %.4f' % max(h1.history['val_accuracy']))
print('phase2 best val_acc %.4f' % max(h2.history['val_accuracy']))

p=model.predict(va,verbose=0); y=va.classes[:len(p)]; pr=p.argmax(1)
print('\n%-9s %5s %7s %7s' % ('class','n','recall','prec'))
for i in range(nc):
    m=y==i; pm=pr==i
    print('%-9s %5d %7.2f %7.2f' % (idx[i],m.sum(),(pr[m]==i).mean() if m.sum() else 0,
                                     (y[pm]==i).mean() if pm.sum() else 0))
print('\nconfusion (row=true):')
print('%-9s'%'' + ''.join('%8s'%idx[j][:7] for j in range(nc)))
for i in range(nc):
    print('%-9s'%idx[i] + ''.join('%8d'%int(((y==i)&(pr==j)).sum()) for j in range(nc)))
