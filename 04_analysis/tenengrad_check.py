"""Independent focus-measure check (Tenengrad, Sobel) used in Section 4.6.

Computes the Tenengrad score for every image in simulation_fov_data.csv, compares
the Tenengrad maximum with the Brenner maximum per FOV, and (optionally) the
sharpness retained at the stopping position of each run.

Usage (from 04_analysis, after running 03_pipeline_validation/retained_sharpness.py):
    python tenengrad_check.py --fov ../03_pipeline_validation/output/simulation_fov_data.csv \
        --retained ../03_pipeline_validation/output_selection/retained_sharpness_per_fov.csv
"""
import argparse
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool

def tenengrad(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return np.nan
    im = im.astype(np.float32)
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fov", required=True)
    ap.add_argument("--retained", default=None)
    ap.add_argument("--out", default="tenengrad_test.csv")
    a = ap.parse_args()
    d = pd.read_csv(a.fov)
    with Pool() as pool:
        d["tenengrad"] = pool.map(tenengrad, d.abs_filepath.tolist(), chunksize=50)
    d.to_csv(a.out, index=False)
    # Servo-count session: convert positions to degrees (300/1023), as in the paper.
    d.loc[d.dataset.str.startswith("Dataset_93"), "pos"] *= 300 / 1023
    rows = []
    for (ds, f), x in d.groupby(["dataset", "fov_id"]):
        pb = x.loc[x.brenner.idxmax(), "pos"]; pt = x.loc[x.tenengrad.idxmax(), "pos"]
        rows.append((abs(pb - pt), x.brenner.rank().corr(x.tenengrad.rank())))
    r = pd.DataFrame(rows, columns=["peak_diff_deg", "rank_corr"])
    print("FOV:", len(r), "| peak diff <=0.5 deg: %.1f%%" % ((r.peak_diff_deg <= 0.5).mean() * 100),
          "| max diff %.2f" % r.peak_diff_deg.max(), "| median rank corr %.3f" % r.rank_corr.median())
    if a.retained:
        rt = pd.read_csv(a.retained)
        ref = {k: (x.pos.values, x.tenengrad.values, x.tenengrad.max()) for k, x in d.groupby(["dataset", "fov_id"])}
        def f(row):
            pos, ten, m = ref[(row.dataset, row.fov_id)]
            return ten[np.abs(pos - row.final_pos).argmin()] / m
        rt["tenengrad_final"] = rt.apply(f, axis=1)
        cl = rt[rt.model != "gss_only"]
        print("classifier runs: mean %.3f median %.3f | >0.5: %.1f%%" % (
            cl.tenengrad_final.mean(), cl.tenengrad_final.median(), (cl.tenengrad_final > 0.5).mean() * 100))
        print("GSS-only mean %.3f" % rt[rt.model == "gss_only"].tenengrad_final.mean())
