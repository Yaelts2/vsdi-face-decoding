# open_permutation_sliding.py
"""
Open ONE sliding-window permutation run and append everything MATLAB needs to the model's .mat:
    accuracy null + significance (as before)  AND  the null weights of every shuffled model,
    in the same format as Weights_acrossT, for the shuffle-model overlap null.

Variables added to the model .mat
---------------------------------
    real_trial_curve     (n_win x n_folds)          real per-fold trial accuracy
    real_frame_curve     (n_win x 1)
    null_trial_folds     (n_perm x n_win x n_folds)
    null_trial_acc       (n_perm x n_win)
    sig01                (n_win x 1) logical         perm test, mean across folds, FDR, alpha=SIG_ALPHA
    NullWeights_acrossT  (10000 x n_save x n_perm) single, NaN outside the ROI
                         = w_mean_windows of each shuffled model -> same kind of map as Weights_acrossT(:, k)
    null_w_centers       (n_save x 1)  window centers of the saved null weights (same units as 'centers')
    null_w_win_idx       (n_save x 1)  1-based column index into Weights_acrossT / centers, i.e.
                                       NullWeights_acrossT(:, j, p)  <->  Weights_acrossT(:, null_w_win_idx(j))
    perm_n, perm_seed, perm_dir

File safety: a one-time backup <model>__backup_before_perm.mat is made, the new .mat is written to
tempdir and then moved onto the target.
"""
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # project root
from scripts.functions_scripts import save_results as sr
from scripts.functions_scripts import model_control as mc

# =========================
# CONFIG  (edit per dataset)
# =========================
PERM_ROOT = Path(r"C:\project\vsdi-face-decoding\results\permutation_test_sliding_window")
TEST_ROOT = PERM_ROOT / "slidingwindow__perm100__110209a24_frame1-100__withWeights__2026-10-20_..."   # <- EDIT

# folder of the model .mat files; the .mat name is taken from config.json (source_model_name)
MAT_DIR = Path(r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab"
               r"\new project\new project results\saveDataModels")
MODEL_MAT = None        # None = MAT_DIR / (source_model_name + '.mat'); or set a full path by hand

SIG_ALPHA = 0.05
MAKE_FIGURES = True

# =========================
# 1) Load the permutation run
# =========================
perm = sr.load_sliding_window_permutation(str(TEST_ROOT))
cfg = perm["config"]
print("perm run :", TEST_ROOT.name)
print("model    :", cfg["source_model_name"])

if perm["null_w_mean"] is None:
    raise RuntimeError("This permutation run has no null weights - rerun it with run_permutation_test_slidingwin.py")

centers = np.asarray(perm["centers"], dtype=int)
real_fold_trial_acc = np.asarray(perm["real_fold_trial_acc"], dtype=float)   # (n_win, n_folds)
null_trial_folds = np.asarray(perm["null_trial_folds"], dtype=float)         # (n_perm, n_win, n_folds)
null_w = np.asarray(perm["null_w_mean"], dtype=np.float32)                   # (n_perm, n_save, n_roi)
null_w_centers = np.asarray(perm["null_w_centers"], dtype=int)               # (n_save,)
roi_mask = np.asarray(perm["roi_mask"], dtype=bool).ravel()                  # (10000,)
n_perm, n_save, n_roi = null_w.shape
print(f"null weights: {n_perm} perms x {n_save} windows x {n_roi} ROI pixels")

sig01_full, p_raw, p_corr = mc.sig_vector_perm_mean_across_folds(real_fold_trial_acc, null_trial_folds,
                                                                 alpha=SIG_ALPHA)

# =========================
# 2) Null weights -> full-image (10000 x n_save x n_perm), NaN outside ROI (same as Weights_acrossT)
# =========================
if int(roi_mask.sum()) != n_roi:
    raise ValueError(f"ROI mask has {int(roi_mask.sum())} pixels but the null weights have {n_roi}")
NullW_img = np.full((roi_mask.size, n_save, n_perm), np.nan, dtype=np.float32)
NullW_img[roi_mask, :, :] = np.transpose(null_w, (2, 1, 0))                  # (n_roi, n_save, n_perm)

# =========================
# 3) Load the model .mat and check it matches
# =========================
model_mat = Path(MODEL_MAT) if MODEL_MAT else MAT_DIR / (cfg["source_model_name"] + ".mat")
if not model_mat.exists():
    raise FileNotFoundError(f"Model .mat not found: {model_mat}")
D = {k: v for k, v in loadmat(str(model_mat)).items() if not k.startswith("__")}
print("model .mat:", model_mat.name)

mat_centers = np.asarray(D["centers"]).ravel().astype(int)
if not np.array_equal(mat_centers, centers):
    raise ValueError("centers in the .mat differ from the permutation run - wrong model?")
null_w_win_idx = np.array([int(np.where(mat_centers == c)[0][0]) for c in null_w_centers]) + 1   # 1-based

W_real = np.asarray(D["Weights_acrossT"], dtype=float)                       # (10000, n_win)
roi_mat = np.isfinite(W_real[:, 0])
if not np.array_equal(roi_mat, roi_mask):
    raise ValueError("ROI of Weights_acrossT differs from the ROI used in the permutation run")
# real weights in the .mat vs mean of null weights: just for the log (null maps should be much weaker)
k0 = null_w_win_idx[n_save // 2] - 1
print(f"|w| at center {mat_centers[k0]}: real mean {np.nanmean(np.abs(W_real[:, k0])):.3e}, "
      f"null mean {np.nanmean(np.abs(NullW_img[:, n_save // 2, :])):.3e}")

# =========================
# 4) Append + save (backup once, write to tempdir, move)
# =========================
D["real_trial_curve"] = real_fold_trial_acc.astype(np.float64)                         # (n_win, n_folds)
D["real_frame_curve"] = np.asarray(perm["real_frame_curve"], dtype=np.float64).reshape(-1, 1)
D["null_trial_folds"] = null_trial_folds.astype(np.float64)                            # (n_perm, n_win, n_folds)
D["null_trial_acc"] = np.asarray(perm["null_trial_acc"], dtype=np.float64)              # (n_perm, n_win)
D["sig01"] = np.asarray(sig01_full, dtype=bool).reshape(-1, 1)                          # (n_win, 1)
D["NullWeights_acrossT"] = NullW_img                                                   # single in MATLAB
D["null_w_centers"] = null_w_centers.astype(np.float64).reshape(-1, 1)
D["null_w_win_idx"] = null_w_win_idx.astype(np.float64).reshape(-1, 1)
D["perm_n"] = np.float64(n_perm)
D["perm_seed"] = np.float64(cfg.get("perm_seed", np.nan))
D["perm_dir"] = str(TEST_ROOT)

backup = model_mat.with_name(model_mat.stem + "__backup_before_perm.mat")
if not backup.exists():
    shutil.copy2(model_mat, backup)
    print("backup   :", backup.name)

tmp_path = Path(tempfile.gettempdir()) / model_mat.name
savemat(str(tmp_path), D, do_compression=True)
shutil.move(str(tmp_path), str(model_mat))
print(f"appended permutation results + null weights to {model_mat.name} "
      f"({model_mat.stat().st_size / 1e6:.0f} MB)")

# =========================
# 5) Check figures
# =========================
if MAKE_FIGURES:
    zero_frame, frame_ms = 27, 10
    t_ms = (centers + 1 - zero_frame) * frame_ms

    fig, ax = plt.subplots(figsize=(9, 4))
    null_curves = null_trial_folds.mean(axis=2)
    ax.fill_between(t_ms, np.percentile(null_curves, 2.5, axis=0), np.percentile(null_curves, 97.5, axis=0),
                    color="gray", alpha=0.4, label="null 95%")
    ax.plot(t_ms, real_fold_trial_acc.mean(axis=1), color="k", lw=2, label="real")
    ax.plot(t_ms[sig01_full.astype(bool)], np.full(int(sig01_full.sum()), 1.02), "s", color="k", ms=3,
            label=f"sig (FDR {SIG_ALPHA})")
    ax.axvspan((null_w_centers[0] + 1 - zero_frame) * frame_ms, (null_w_centers[-1] + 1 - zero_frame) * frame_ms,
               color="tab:blue", alpha=0.08, label="windows with saved null weights")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("Time from stimulus onset (ms)")
    ax.set_ylabel("Trial accuracy")
    ax.set_title(cfg["source_model_name"], fontsize=8)
    ax.legend(fontsize=7, loc="lower right")

    # real map vs 3 null maps at the real peak (inside the saved range)
    real_acc = real_fold_trial_acc.mean(axis=1)
    j_peak = int(np.argmax(real_acc[null_w_win_idx - 1]))
    k_peak = null_w_win_idx[j_peak] - 1
    lim = float(np.nanpercentile(np.abs(W_real[:, k_peak]), 99))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    maps = [W_real[:, k_peak]] + [NullW_img[:, j_peak, p] for p in range(min(3, n_perm))]
    titles = [f"real, {t_ms[k_peak]} ms"] + [f"shuffle {p + 1}" for p in range(min(3, n_perm))]
    for a, m, tt in zip(axes, maps, titles):
        im = a.imshow(m.reshape(100, 100), cmap="PiYG_r", vmin=-lim, vmax=lim)
        a.set_title(tt, fontsize=9)
        a.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8, label="weight (same scale)")
    plt.show()
