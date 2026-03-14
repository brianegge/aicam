#!/usr/bin/env python3

import argparse
import os
from pprint import pprint
from tempfile import gettempdir

import cv2
import numpy as np
from colorhash import ColorHash
from PIL import Image

from onnx_object_detection import ONNXRuntimeObjectDetection
from sort import Sort
from utils import draw_bbox

parser = argparse.ArgumentParser(description="Process video")
parser.add_argument("input", metavar="I", nargs="+", help="One or more input videos")
parser.add_argument("output", metavar="O", nargs=1, help="Output file")

args = parser.parse_args()

model_dir = "/Users/brianegge/downloads/5229026c5d694c2292f9f326d4b49620.ONNX/"
# Load labels
with open(model_dir + "labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]
od_model = ONNXRuntimeObjectDetection(model_dir + "model.onnx", labels)

fps = 15
# out_width=1920
# out_height=1080
out_width = 1920 * 2
out_height = 1080 * 2
output_tmp = os.path.join(gettempdir(), os.path.basename(args.output[0]))
print(output_tmp)
vid_writer = cv2.VideoWriter(
    output_tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, out_height)
)
colors = {
    "dog": "yellow",
    "cat": "orange",
    "person": "cyan",
    "raccoon": "brown",
    "deer": "chartreuse",
}
for label in labels:
    colors.setdefault(label, ColorHash(label).hex)

frame_count = 0
for path_video in args.input:
    cap = cv2.VideoCapture(path_video)
    _, img = cap.read()
    sort = Sort(max_age=30, min_hits=3)
    while img is not None:
        if frame_count % 15 == 0:
            print("f{:05d}=".format(frame_count), end="")
            image = Image.fromarray(img.astype("uint8"), "RGB")
            predictions = od_model.predict_image(image)
            predictions = list(filter(lambda p: p["probability"] > 0.6, predictions))
            pprint(predictions)
            trks = []
            for p in predictions:
                trks.append(
                    [
                        p["boundingBox"]["left"],
                        p["boundingBox"]["top"],
                        p["boundingBox"]["left"] + p["boundingBox"]["width"],
                        p["boundingBox"]["top"] + p["boundingBox"]["height"],
                        p["probability"],
                    ]
                )
            if len(trks) > 0:
                trks = np.array(trks)
            else:
                trks = np.empty((0, 5))
            dets = sort.update(trks)
            for det in dets:
                p = {
                    "boundingBox": {
                        "left": det[0],
                        "top": det[1],
                        "width": det[2] - det[0],
                        "height": det[3] - det[1],
                    }
                }
                # draw_bbox(image, p, color='red', label='tracker')
            for p in predictions:
                label = "{} {:.1f}%".format(p["tagName"], p["probability"] * 100)
                draw_bbox(image, p, colors[p["tagName"]], label=label)
            # for *xyxy, conf, cls in det:
            #    plot_one_box(xyxy, img, label=label, color=colors[int(cls)])
            image_out = cv2.resize(np.array(image), (out_width, out_height))
            vid_writer.write(np.array(image_out))
        _, img = cap.read()
        frame_count += 1
vid_writer.release()
os.system('ffmpeg -y -i "%s" "%s"' % (output_tmp, args.output[0]))
os.remove(output_tmp)
