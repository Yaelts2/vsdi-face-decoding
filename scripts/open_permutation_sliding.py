# open_permutation_sliding.py
# Open ONE permutation run (made by run_permutation_test_slidingwin.py, windows 0-300 ms)
# and append the null weights to the model's .mat, for the overlap null in MATLAB.
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
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.functions_scripts import save_results as sr

# =========================
# CONFIG  (edit per dataset)
# =========================
PERM_ROOT = Path(r"C:\project\vsdi-face-decoding\results\permutation_test_sliding_window")
TEST_ROOT = PERM_ROOT / "slidingwindow__perm100__110209a24_frame1-100__withWeights__..."   # <- EDIT

# folder of the model .mat files (the .mat name = source_model_name in config.json)
MAT_DIR = Path(r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab"
               r"\new project\new project results\saveDataModels")

# =========================
# 1) Load the permutation run
# =========================
perm = sr.load_sliding_window_permutation(str(TEST_ROOT))
cfg = perm["config"]
null_w = np.asarray(perm["null_w_mean"], dtype=np.float32)        # (n_perm, n_win, n_roi)
null_w_centers = np.asarray(perm["null_w_centers"], dtype=int)    # (n_win,)
roi_mask = np.asarray(perm["roi_mask"], dtype=bool).ravel()       # (10000,)
n_perm, n_win, n_roi = null_w.shape
print("perm run:", TEST_ROOT.name)
print(f"null weights: {n_perm} shuffles x {n_win} windows x {n_roi} ROI pixels")
print(f"null trial accuracy: mean {float(perm['null_trial_acc'].mean()):.3f} (should be ~0.5)")

# null weights -> full image, NaN outside the ROI (same layout as Weights_acrossT)
NullW_img = np.full((roi_mask.size, n_win, n_perm), np.nan, dtype=np.float32)
NullW_img[roi_mask, :, :] = np.transpose(null_w, (2, 1, 0))

# =========================
# 2) Load the model .mat and match the windows
# =========================
model_mat = MAT_DIR / (cfg["source_model_name"] + ".mat")
D = {k: v for k, v in loadmat(str(model_mat)).items() if not k.startswith("__")}
print("model .mat:", model_mat.name)

mat_centers = np.asarray(D["centers"]).ravel().astype(int)
null_w_win_idx = np.array([int(np.where(mat_centers == c)[0][0]) for c in null_w_centers]) + 1   # 1-based for MATLAB
if not np.array_equal(np.isfinite(np.asarray(D["Weights_acrossT"])[:, 0]), roi_mask):
    raise ValueError("ROI of Weights_acrossT differs from the ROI of the permutation run")

# =========================
# 3) Append and save (backup once, write to tempdir, then move)
# =========================
D["NullWeights_acrossT"] = NullW_img
D["null_w_win_idx"] = null_w_win_idx.astype(np.float64).reshape(-1, 1)
D["null_w_centers"] = null_w_centers.astype(np.float64).reshape(-1, 1)
D["null_trial_acc_perm"] = np.asarray(perm["null_trial_acc"], dtype=np.float64)   # (n_perm, n_win)
D["perm_n"] = np.float64(n_perm)
D["perm_seed"] = np.float64(cfg["perm_seed"])
D["perm_dir"] = str(TEST_ROOT)

backup = model_mat.with_name(model_mat.stem + "__backup_before_perm.mat")
if not backup.exists():
    shutil.copy2(model_mat, backup)

tmp_path = Path(tempfile.gettempdir()) / model_mat.name
savemat(str(tmp_path), D, do_compression=True)
shutil.move(str(tmp_path), str(model_mat))
print(f"saved null weights into {model_mat.name}  (windows {null_w_win_idx[0]}-{null_w_win_idx[-1]} of Weights_acrossT)")
