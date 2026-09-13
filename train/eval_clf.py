#!/usr/bin/env python3
"""Per-class recall on the same validation split, without the shuffle bug.

flow_from_directory shuffles by default, so predictions come back in a
different order than generator.classes -- pairing them makes a good model look
like chance. shuffle=False is the whole fix.
"""
import os, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA = os.path.expanduser('~/train/ipcams2-crops')
OUT  = os.path.expanduser(os.environ.get('CLF_DIR','~/train/ipcams-classifier'))

gen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2).flow_from_directory(
    DATA, target_size=(224,224), batch_size=32, class_mode='categorical',
    subset='validation', seed=0, shuffle=False)

interp = tf.lite.Interpreter(model_path=os.path.join(OUT,'model.tflite'))
interp.allocate_tensors()
inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]
labels = [l.strip() for l in open(os.path.join(OUT,'labelmap.txt'))]

y_true, y_pred, conf = [], [], []
n = gen.samples
seen = 0
for bx, by in gen:
    for i in range(len(bx)):
        interp.set_tensor(inp['index'], bx[i:i+1].astype(inp['dtype']))
        interp.invoke()
        p = interp.get_tensor(outp['index'])[0]
        y_pred.append(int(p.argmax())); conf.append(float(p.max()))
        y_true.append(int(by[i].argmax()))
        seen += 1
    if seen >= n: break
y_true, y_pred, conf = np.array(y_true), np.array(y_pred), np.array(conf)

print('\noverall accuracy: %.4f  (n=%d, chance=%.3f)' % ((y_true==y_pred).mean(), len(y_true), 1/len(labels)))
print('\n%-9s %5s %7s %7s %7s' % ('class','n','recall','prec','medconf'))
for i,l in enumerate(labels):
    m = y_true==i; pm = y_pred==i
    rec = (y_pred[m]==i).mean() if m.sum() else 0
    prec = (y_true[pm]==i).mean() if pm.sum() else 0
    mc = np.median(conf[pm]) if pm.sum() else 0
    print('%-9s %5d %7.2f %7.2f %7.2f' % (l, m.sum(), rec, prec, mc))

# Frigate needs 60% consensus over up to 16 crops; a per-frame threshold of
# 0.8 is the config default, so how often would a single crop even qualify?
for t in (0.5, 0.8):
    ok = conf >= t
    print('\nconf >= %.1f: %.0f%% of crops, accuracy on those %.2f' % (
        t, 100*ok.mean(), (y_true[ok]==y_pred[ok]).mean() if ok.sum() else 0))
print('\nconfusion (row=true, col=pred):')
print('%-9s' % '' + ''.join('%8s'%l[:7] for l in labels))
for i,l in enumerate(labels):
    row=[int(((y_true==i)&(y_pred==j)).sum()) for j in range(len(labels))]
    print('%-9s'%l + ''.join('%8d'%v for v in row))
