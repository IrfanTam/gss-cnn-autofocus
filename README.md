# GSS-CNN autofocus: code and result archive

Code and result archive for the manuscript "A Hybrid GSS-CNN Pipeline for Efficient Autofocus in Robotic Bright-Field Microscopy" (International Journal of Intelligent Engineering and Systems, Paper ID 20267653, revised version).

The study is a replay study: classifier-assisted golden section search (GSS) is evaluated on recorded focus stacks of 71 held-out fields of view (FOV). No live instrument run is included.

Immutable revision of this release: tag `v1.0-ijies-revision` (commit hash: see the GitHub release page).

Tested with Python 3.13.9, PyTorch 2.9.1 (CUDA 12.6), torchvision 0.24.1, timm 1.0.26, Ultralytics 8.4.161, scikit-learn 1.7.2, on an NVIDIA GeForce RTX 3050 (4 GB).

## Setup
    python -m venv .venv && .venv\Scripts\activate          (Windows)
    pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
    pip install -r requirements.txt

## Paths to edit first
The scripts were written on the authors' PC. Edit the path constants at the top of `01_data_preparation/data_preparation.py` (`DATASET_MASTER`, `EXCLUDED_FOV_PATH`, use `01_data_preparation/excluded_fov.txt`), of `03_pipeline_validation/*.py`, of `04_supporting_analysis/*.py` and of `05_reproducibility_archive/make_archive.py`.

## Folders
- `01_data_preparation`: labelling, FOV-level split, class balancing; list of the 10 excluded FOV.
- `02_backbone_training`: the 14 networks of the common training script, the two FocusNet variants, YOLO26n-cls arguments.
- `03_pipeline_validation`: replay simulator (`gss_simulator.py`), classifier prediction caches, validation-based selection, unit-consistent replay with baselines and paired statistics, p_min sweep, retained sharpness, non-learning stopping rules, replay on training FOV.
- `04_supporting_analysis`: paired confidence intervals, Spearman permutation test, triggering-frame labels, unimodality check, Tenengrad check, figure script.
- `05_reproducibility_archive`: anonymised split assignments, selected checkpoint and threshold per classifier, per-run table of all test searches, per-field baseline outcomes, `make_archive.py`.
- `results`: the small result files used in the manuscript tables. `docs/STEP_CONVENTION_R1-1.md`: step convention and re-run order.

## Run order
1. `01_data_preparation/data_preparation.py`: labels images (Focused P > 0.55, Medium 0.30 < P <= 0.55, Blur otherwise, P = Brenner score / per-FOV maximum; boundaries fixed a priori), removes the 10 excluded FOV, splits by FOV (70/15/15, seed 42, stratified by capture session) and builds the class-balanced set (seed 42).
2. `02_backbone_training/run_all.py`: trains the 14 networks (AdamW, learning rate 1e-4, batch size 64, weight decay 0.01, ReduceLROnPlateau factor 0.5 patience 5, up to 50 epochs, early stopping patience 10, class-weighted cross-entropy; checkpoint = epoch with the lowest validation loss). YOLO26n-cls is trained with Ultralytics, arguments in `yolo26n_args.yaml` (seed 0, default loss, no class weights, best.pt). Optional: `test_leftover_blur.py`.
3. `03_pipeline_validation`: `prepare_simulation_data.py`, `prepare_val_simulation_data.py`, `prepare_train_simulation_data.py`, then `cnn_inference_cache.py`, `cnn_inference_cache_val.py`, `cnn_inference_cache_train.py`, then `select_on_val.py` (p_min and classifier chosen on validation, test evaluated once), `rerun_unitfix.py` (replay, baselines, paired statistics), `sensitivity_p_min.py`, `retained_sharpness.py`, `nonlearning_stop.py`, `report_train_replay.py`.
4. `04_supporting_analysis`: `paired_ci.py <output_dir>`, `spearman_perm.py`, `trigger_labels.py`, `unimodality_check.py`, `tenengrad_check.py`, `make_figs_steps.py`.
5. `05_reproducibility_archive/make_archive.py`: writes the archive tables and checks by assertion that the per-run log reproduces the official replay (1,065 classifier runs on 71 FOV).

## Conventions
- A step is one position to which the stage is moved and at which an image is captured. The midpoint returned after interval termination counts as one further step. Every new requested angle is charged, even if it maps to an already used recorded angle. A strategy returns the sampled (nearest recorded) angle. The final repositioning move of the full sweep, hill climbing and coarse-to-fine is not charged.
- Success: |returned angle - Brenner maximum| <= 1.0 degree (0.5 and 2.0 as sensitivity). Servo counts of the 93FOV session are converted with 300/1023 degrees per count.
- Time: estimated as steps x 0.65 s (programmed step interval of the acquisition software: command, 400 ms settling, capture and save). Median save-to-save intervals of recorded 0.5 degree moves were 0.66 s (two sessions) and 1.13 s (one session). End-to-end controller time was not measured. Measured GPU forward-pass latency (batch 1): FocusNet-Nano 0.79 ms, ViT-Tiny 3.51 ms, DenseNet-121 14.54 ms.
- Random seeds: 42 for class balancing and the FOV split; 0 for YOLO26n-cls; fixed seeds for the bootstrap and permutation tests. Training of the 14 networks sets no global seed.

## Not included
Images, model checkpoints, prediction caches, slide or specimen identifiers (not recorded), the link between anonymised and original field identifiers (`field_key_map_PRIVATE.csv`, excluded by `.gitignore`), and live-instrument measurements. Trained weights are not part of this release.
