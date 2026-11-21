# Reproducibility archive (GSS-CNN autofocus, IJIES revision)

Immutable revision: [GitHub URL, release tag and commit hash to be filled before submission].

## Files
- `splits_public.csv`: anonymised field code, capture session, slide group ("not recorded"), split (train 299, val 63, test 71, excluded 10).
- `field_key_map_PRIVATE.csv`: link from field code to the original dataset/field id. Not for public release.
- `classifier_selection.csv`: validation-selected confidence threshold, checkpoint file and test result for each of the 15 classifiers.
- `per_run_test.csv`: one row per (strategy, test field): requested angles, sampled (recorded) angles, stop reason, final requested/sampled angle, charged steps, distinct recorded frames, final error, success at 1.0 degree, estimated time.
- `baselines_per_field.csv`: per-field outcomes of full sweep, hill climbing, adaptive hill climbing, coarse-to-fine and GSS-only (tolerance 1.0 degree).
- `make_archive.py`: regenerates the tables above and checks that the per-run log reproduces the official replay (1,065 classifier runs).

## Regeneration order
`01_data_preparation/data_preparation.py` -> `02_backbone_training/` (seed and hyper-parameters in the paper, Table of settings) -> `03_pipeline_validation/cnn_inference_cache*.py`, `prepare_*simulation_data.py` -> `select_on_val.py` -> `rerun_unitfix.py` -> `report_train_replay.py` -> `retained_sharpness.py` -> `04_supporting_analysis/*` -> `05_reproducibility_archive/make_archive.py`.

## Conventions
- A step is a distinct commanded position with an image capture. The returned midpoint of an interval-terminated GSS search counts as one further step (`STEP_CONVENTION_R1-1.md`). Requests are mapped to the nearest recorded angle (`nearest_index`); requests are charged even if they map to an already used recorded angle.
- Final repositioning moves without capture (full sweep, hill climbing, coarse-to-fine) are not charged.
- Success: |returned angle - Brenner maximum| <= 1.0 degree (0.5 and 2.0 reported as sensitivity). Servo counts to degrees: 300/1023 (session 93FOV only).

## Timing assumptions
- Estimated time = steps x 0.65 s. 0.65 s is the step interval programmed in the dataset acquisition program (650 ms step timer); the instrument controller waits 400 ms for the motor to settle before capture.
- Median interval between consecutive saved images for 0.5 degree moves: 0.66 s (two sessions), 1.13 s (one session). These time stamps are file-save times; whether they include camera exposure and transfer was not separated.
- Measured GPU forward-pass latency (batch 1): FocusNet-Nano 0.79 ms, ViT-Tiny 3.51 ms, DenseNet-121 14.54 ms. End-to-end controller time on the instrument was not measured.

## Not included
Slide or specimen identifiers (not recorded), the raw images (see the dataset link in the paper), and live-instrument measurements.
