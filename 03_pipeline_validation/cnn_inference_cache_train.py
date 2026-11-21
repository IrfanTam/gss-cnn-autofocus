"""
Supplementary analysis (Step T2): CNN inference cache for the TRAIN FOV.
Same as cnn_inference_cache.py, reading output_train/simulation_fov_data_train.csv
and writing output_train/cnn_cache/<model>_predictions.parquet.
Resumable: models whose parquet already exists are skipped.

    python cnn_inference_cache_train.py
    python cnn_inference_cache_train.py --model resnet18
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import cnn_inference_cache as c  # noqa: E402

c.DATA_PATH = HERE / "output_train" / "simulation_fov_data_train.csv"
c.CACHE_DIR = HERE / "output_train" / "cnn_cache"

if __name__ == "__main__":
    c.main()
