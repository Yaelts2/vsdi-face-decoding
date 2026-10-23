# open_permutation_sliding.py
# Append the null weights of ALL finished permutation runs (made by run_permutation_test_slidingwin.py,
# windows 0-300 ms) to their model .mat files, for the overlap null in MATLAB.
# Every "...__withWeights__..." folder in PERM_ROOT is processed; models whose .mat already holds
# that run are skipped, so it is safe to run again.
#
# Added to the .mat (nothing existing is changed):
#   NullWeights_acrossT  (10000 x n_win x n_perm) single, NaN outside the ROI
#                        = same kind of map as Weights_acrossT, but from a shuffled-label model
#   null_w_win_idx       (n_win x 1) column of Weights_acrossT that each null window matches:
#                        NullWeights_acrossT(:, j, p)  <->  Weights_acrossT(:, null_w_win_idx(j))
#   null_w_centers       (n_win x 1) window centers (same units as 'centers')
#   null_trial_acc_perm  (n_perm x n_win) trial accuracy of each shuffled model (should be ~0.5)
#   perm_n, perm_seed, perm_dir
import sys
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.functions_scripts import save_results as sr

# =========================
# CONFIG
# =========================
# folder with the "...__withWeights__..." permutation folders
PERM_ROOT = Path(r"C:\project\vsdi-face-decoding\results\allareas\permutation_test_sliding_window")   # <- check

# folder of the model .mat files (the .mat name = source_model_name in config.json)
MAT_DIR = Path(r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab"
               r"\new project\new project results\saveDataModels")


def append_one(test_root):
    """append the null weights of ONE permutation folder to its model .mat; returns a status string"""
    perm = sr.load_sliding_window_permutation(str(test_root))
    cfg = perm["config"]
    model_mat = MAT_DIR / (cfg["source_model_name"] + ".mat")
    if not model_mat.exists():
        raise FileNotFoundError(f"model .mat not found: {model_mat.name}")

    D = {k: v for k, v in loadmat(str(model_mat)).items() if not k.startswith("__")}
    if "perm_dir" in D and str(np.asarray(D["perm_dir"]).ravel()[0]) == str(test_root):
        return "skipped (already appended)"

    null_w = np.asarray(perm["null_w_mean"], dtype=np.float32)        # (n_perm, n_win, n_roi)
    null_w_centers = np.asarray(perm["null_w_centers"], dtype=int)    # (n_win,)
    roi_mask = np.asarray(perm["roi_mask"], dtype=bool).ravel()       # (10000,)
    n_perm, n_win, n_roi = null_w.shape

    # null weights -> full image, NaN outside the ROI (same layout as Weights_acrossT)
    NullW_img = np.full((roi_mask.size, n_win, n_perm), np.nan, dtype=np.float32)
    NullW_img[roi_mask, :, :] = np.transpose(null_w, (2, 1, 0))

    # match windows + ROI with the real model
    mat_centers = np.asarray(D["centers"]).ravel().astype(int)
    null_w_win_idx = np.array([int(np.where(mat_centers == c)[0][0]) for c in null_w_centers]) + 1   # 1-based
    if not np.array_equal(np.isfinite(np.asarray(D["Weights_acrossT"])[:, 0]), roi_mask):
        raise ValueError("ROI of Weights_acrossT differs from the ROI of the permutation run")

    D["NullWeights_acrossT"] = NullW_img
    D["null_w_win_idx"] = null_w_win_idx.astype(np.float64).reshape(-1, 1)
    D["null_w_centers"] = null_w_centers.astype(np.float64).reshape(-1, 1)
    D["null_trial_acc_perm"] = np.asarray(perm["null_trial_acc"], dtype=np.float64)   # (n_perm, n_win)
    D["perm_n"] = np.float64(n_perm)
    D["perm_seed"] = np.float64(cfg["perm_seed"])
    D["perm_dir"] = str(test_root)

    # backup once, write to tempdir, then move
    backup = model_mat.with_name(model_mat.stem + "__backup_before_perm.mat")
    if not backup.exists():
        shutil.copy2(model_mat, backup)
    tmp_path = Path(tempfile.gettempdir()) / model_mat.name
    savemat(str(tmp_path), D, do_compression=True)
    shutil.move(str(tmp_path), str(model_mat))

    null_acc = float(np.mean(perm["null_trial_acc"]))
    return (f"done: {n_perm} shuffles x {n_win} windows (cols {null_w_win_idx[0]}-{null_w_win_idx[-1]}), "
            f"null acc {null_acc:.3f}")


if __name__ == "__main__":
    folders = sorted(d for d in PERM_ROOT.glob("*__withWeights__*")
                     if d.is_dir() and not d.name.endswith("__running") and (d / "perm_result.npz").exists())
    print(f"{len(folders)} finished permutation runs in {PERM_ROOT}\n")
    status = []
    for i, d in enumerate(folders, start=1):
        print(f"[{i}/{len(folders)}] {d.name}")
        try:
            msg = append_one(d)
        except Exception as e:
            msg = f"FAILED: {type(e).__name__}: {e}"
            traceback.print_exc()
        print("   ", msg)
        status.append((d.name, msg))

    print("\n=== summary ===")
    for name, msg in status:
        print(f"  {name.split('__')[2]:<25} {msg}")
