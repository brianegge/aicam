# Training

Everything needed to rebuild a deployed model from the dataset. It is here so
that losing `~/aicam-models` costs GPU hours rather than a model — on
2026-09-13 a `git stash -u` during a deploy destroyed all five ONNX files, and
they were recoverable only because no `git gc` had run in between.

These run on claw-mini (`openclaw.home`). Darknet has no GPU path on Apple
silicon, so anything retrained here uses Ultralytics and needs
`backend=ultralytics` in `config.txt`.

## Rebuilding the deployed ipcams animal detector

`ipcams_v32_yolo11m_608.onnx`, live since 2026-09-11. Roughly 20 h on an M4.

1. **Fetch the dataset.** Roboflow `egge-public/ipcams2`, version 32, YOLOv8
   format, to `~/train/ipcams2-v32`. The API key is in `config.txt` under
   `[roboflow]` — it is not in this repo.

2. **Stretch it to 608 square** — `build_608.sh` (run on ubuntu24, where the
   dataset lives on `/mnt/storage/ipcams2`).

   `camera.py`'s `resize()` does an unconditional `cv2.resize` to 608x608, not
   a letterbox. Training on letterboxed images teaches an aspect ratio that
   never occurs live. YOLO labels are normalised, so a pure stretch leaves
   them correct and they are copied verbatim.

3. **Train** — `run_ipcams_y11m.sh`.

   Two settings are load-bearing and neither is obvious. `rect=False` is
   mandatory: Ultralytics' `build_transforms()` silently zeroes
   mosaic/mixup/cutmix when `rect` is true, and `args.yaml` still records the
   value you asked for, so a rect run looks comparable and is not. `batch=8`
   is the ceiling on a 16 GB mini while aicam is running — `batch=16` starved
   aicam into a full detection blackout.

4. **Export and score** — `after_ipcams_y11m.sh`. Waits on the training
   *process*, not on `best.pt`: Ultralytics rewrites `best.pt` after every
   improving epoch, so the file exists within minutes and means nothing.

5. **Choose thresholds** — `../sweep.py`, not the trainer's mAP. The
   validation mAP is not measured at the thresholds aicam acts on, and the two
   disagree badly across an architecture change: `dog=0.95`, tuned for
   yolov4's confidence distribution, gave 1.9% recall on a yolo11 model with
   mAP50 0.90.

6. **Install** into `~/aicam-models/` and restart. Keep the previous file —
   `ipcams_color_yolov4.onnx` and `ipcams_grey_yolov4.onnx` are the rollback
   for the current model.

## The split does not see everything

`compare_models.py` scores a held-out split, and the split is drawn from the
same review pipeline as the training data, so it contains the frames someone
thought worth labelling. It reported **deer precision 1.000** for a model that
went on to produce five distinct recurring deer false positives in the field,
because it holds none of that treeline-at-dusk scenery.

Read a clean precision number as "no false positives of a kind already in the
dataset". Field false positives are handled by `excludes/` and by `verify.py`.

## The rest of these files

Experiments, kept for how the decisions were reached rather than because
anything runs them:

| file | question it answered |
|------|----------------------|
| `build_arms.py`, `run_arms.sh`, `eval_arms.py` | is the colour/grey specialist split worth keeping? (no — identical F1, and each arm had a blind spot) |
| `run_mosaic.sh`, `eval_mosaic.py`, `eval_mosaic2.py` | does mosaic augmentation help here? |
| `train_ipcams_classifier.py`, `train_grouped.py`, `train_ft.py`, `eval_clf.py`, `classify_grey.py` | a crop classifier as a second stage |
| `export_crops.py`, `export_uint8.py`, `test_dist.py` | crop export and score distributions |
| `compare.py` | earlier vehicle-model comparison, superseded by `../compare_models.py` |
| `after_y11m.sh`, `train_ipcams.sh`, `eval_a10.py`, `hours.py` | earlier runs of the above |

Paths inside them are absolute and assume claw-mini. `compare.py`,
`after_y11m.sh` and `after_ipcams_y11m.sh` were repointed at `~/aicam-models`
when the models moved out of the checkout; the others do not load a deployed
model.
