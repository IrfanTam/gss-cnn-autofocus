# GSS + classifier early termination for microscope autofocus — code

Code used for the experiments in the manuscript "Learning When to Stop: Classifier-Assisted
Early Termination of Golden Section Search for Robotic Microscope Autofocus".

Tested with Python 3.13.9, NVIDIA GeForce RTX 3050 (4 GB), PyTorch 2.9.1+cu126.

## Setup
    python -m venv .venv && .venv\Scripts\activate          (Windows)
    pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
    pip install -r requirements.txt

## Paths to edit first
The scripts were written on the authors' PC. Before running, edit these constants:
- `01_data_preparation/data_preparation.py`: `DATASET_MASTER` (folder with the five capture
  sessions) and `EXCLUDED_FOV_PATH` (use `01_data_preparation/excluded_fov.txt`).
- `03_pipeline_validation/*.py`: check the `DATA_PATH` / `CHECKPOINT_DIR` constants at the top.

## Run order
1. `cd 01_data_preparation && python data_preparation.py`
   Labels images (Focused > 0.55, Medium 0.30–0.55, Blur <= 0.30 of the per-FOV Brenner maximum),
   removes the 10 excluded FOV, splits by FOV (70/15/15, random_state=42) and builds the balanced set.
2. `cd 02_backbone_training && python run_all.py`
   Trains the 14 networks of the common script (AdamW, lr 1e-4, batch 64, 50 epochs, patience 10).
   YOLO26n-cls is trained with Ultralytics (arguments in `02_backbone_training/yolo26n_args.yaml`):
       yolo classify train model=yolo26n-cls.pt data=<dataset_balanced> epochs=50 patience=10 batch=64 imgsz=224 seed=0 name=yolo26n_balanced
   Optional: `python test_leftover_blur.py --model resnet18_balanced` (generalisation check on unused Blur images).
3. `cd 03_pipeline_validation`
       python prepare_simulation_data.py        # full focus stacks of the 71 test FOV
       python prepare_val_simulation_data.py    # full focus stacks of the 63 validation FOV
       python cnn_inference_cache.py            # predictions of all 15 models on test FOV
       python cnn_inference_cache_val.py        # ... on validation FOV
       python select_on_val.py                  # selects p_min and classifier on validation, reports test once
       python rerun_unitfix.py                  # unit-consistent replay, baselines, paired statistics
       python sensitivity_p_min.py              # p_min sweep (Fig. Pareto)
       python retained_sharpness.py             # sharpness retained at the stopping position
4. `cd 04_analysis && python tenengrad_check.py --fov ... --retained ...` (independent focus measure).

## Results included
`results/` holds the small result files used in the paper (comparison table, selection and
sharpness outputs, YOLO26n-cls training log). Image data, model checkpoints and prediction caches are not included.

## Notes
- Network training of the 14 networks does not set a global random seed; only the class balancing and the FOV split use `random_state=42`.
- Positions of the 93FOV session are logged in servo counts and converted with 300/1023 degrees per count.
