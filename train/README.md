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

## Rebuilding the object classifier with the `none` class

`ipcams_animals`, live since 2026-09-21. Roughly 40 min on an M4. The previous
seven-class model was a softmax over animals only, so every crop it was handed
had to be one of them -- one pale rock on peach_tree came back as fox, dog and
deer on different frames of the same object.

**The class must be called exactly `none`.** Frigate's
`CustomObjectClassificationProcessor` drops the consensus label when it equals
that string; any other name (`background`, `negative`) is published as a
sub_label and every rock gets one.

1. **Mine the detector's own false positives** -- `mine_fp.py`, on claw-mini.
   Runs stock COCO `yolo11s` over ipcams2 and keeps animal-class boxes that
   match no ground-truth box. Those are exactly the crops the classifier will
   be handed and has no right answer for.

   **Eyeball every one.** 14 of 69 were real animals ipcams2 had not labelled,
   including a deer under a pink IR cast. Training on those teaches
   deer-is-none. No property of the box predicts it; only the picture does.
   The verdicts live in `build_background.py`'s `REJECT`, by index and reason.

2. **Build the source boxes** -- `build_background.py`, on ubuntu24 where the
   dataset lives. Combines the verified hard negatives with random squares
   that overlap neither a label nor a detection.

3. **Build the crops** -- `build_grouped.py` with the default `BACKGROUND=1`.

4. **Add Frigate's own false positives** -- `add_field_none.py`. Hold at least
   one confirmed object out (`HOLDOUT`, default `treeline_obj`) or there is
   nothing left to measure generalisation with.

5. **Train** -- `train_grouped.py` with `CROPS_DIR` and `OUT_DIR`, using
   `~/train/.tfvenv` (`.venv` is the Ultralytics one and has no TensorFlow).

6. **Score on field data** -- `eval_field.py`, not the validation split. See
   below for why, and step 7 for what it caught.

7. **Install** into `/opt/frigate/config/model_cache/ipcams_animals/` and
   restart. Keep the previous pair; `ipcams_animals-backup-20260921-080557`
   on frigate.home is the rollback for the current model.

### Scenery alone does not make a `none` class

The first build was 97% random grass, pavement and stone. Validation looked
fine -- `none` recall 0.95, precision 0.98, overall accuracy up on the
seven-class model -- and on the rock it went **backwards**: fox at 0.95-1.00,
more confident than the model it replaced. It had learned "none == flat
texture", which is the failure Frigate's own docs warn about.

What fixed it was raising the hard negatives from 3% of the class to roughly a
third: `NONE_MAX_PER_GROUP` (one boulder was detected 92 times in different
light -- diversity, not redundancy), `HARD_REPS`, `SIM_MIN` lowered to 40 px
because the live failure is a 56 px crop, and the field crops from step 4.
After that the rock reads `none` at 1.00 on all 20 frames, the held-out
tree-line object reads `none` with no animal label at all, and 16 crops of a
covered grill that the old model called `dog` read `none`.

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
| `train_ipcams_classifier.py`, `train_ft.py`, `eval_clf.py`, `classify_grey.py` | a crop classifier as a second stage; `train_grouped.py` is the one that ships |
| `export_crops.py`, `export_uint8.py`, `test_dist.py` | crop export and score distributions |
| `compare.py` | earlier vehicle-model comparison, superseded by `../compare_models.py` |
| `after_y11m.sh`, `train_ipcams.sh`, `eval_a10.py`, `hours.py` | earlier runs of the above |

Paths inside them are absolute and assume claw-mini. `compare.py`,
`after_y11m.sh` and `after_ipcams_y11m.sh` were repointed at `~/aicam-models`
when the models moved out of the checkout; the others do not load a deployed
model.
