"""
Step B. Same inference as cnn_inference_cache.py, but on the validation FOV.
Reads output_val/simulation_fov_data_val.csv and writes
output_val/cnn_cache/<model>_predictions.parquet. Requires torch + the trained
checkpoints (run on the lab/training PC).

    python cnn_inference_cache_val.py                # all 15 models
    python cnn_inference_cache_val.py --model resnet18
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import cnn_inference_cache as c  # noqa: E402

c.DATA_PATH = HERE / "output_val" / "simulation_fov_data_val.csv"
c.CACHE_DIR = HERE / "output_val" / "cnn_cache"

if __name__ == "__main__":
    c.main()
