import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from functions_scripts import preprocessing_functions as pre
from functions_scripts import save_results as sr
from functions_scripts import model_control as mc

ourCmap = pre.green_gray_magenta()


################ 
# ploting the permutation results for a fixed window run
################

## user must edit these parameters for each run!##
### which permutation test to load and plot results from
results_root = Path(r"C:\project\vsdi-face-decoding\results\permutation_test_sliding_window")
test_root = results_root / "slidingwindow__perm100__110209d24_frame1-100__2026-06-23_20-06-07" # <-- update this to your model folder you want to load and plot results from
print("test_root:", test_root)
prem_results = sr.load_sliding_window_permutation(str(test_root))
# sig01 from your permutation-mean-across-folds function:
centers=prem_results["centers"]
real_fold_trial_acc=prem_results["real_fold_trial_acc"]
real_frame_acc=prem_results["real_frame_curve"]
null_trial_folds=prem_results["null_trial_folds"]
sig01,p_raw,p_corr = mc.sig_vector_perm_mean_across_folds(real_fold_trial_acc, null_trial_folds, alpha=0.05)
sig01=sig01[0:45]
p_raw=p_raw[0:45]
frames = prem_results["centers"]+1
frames=frames[0:45]
'''
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(frames, p_raw[0:45], color='steelblue', linewidth=1.5)
ax.axhline(0.05, color='red', linestyle='--', label='alpha = 0.05')
ax.fill_between(frames, p_raw, where=(p_raw < 0.05), color='green', alpha=0.3, label='significant')
ax.set_xlabel('Frame')
ax.set_ylabel('p-value')
ax.legend()
plt.tight_layout()
plt.show()

mc.plot_sw_perm_simple(prem_results, frames=frames, sig01=sig01, frame0=27, ms_per_frame=10)
'''
a=1


# =========================
# APPEND PERMUTATION RESULTS INTO THE MODEL'S .mat
# =========================
from scipy.io import loadmat, savemat

# This permutation corresponds to which model? Set it by hand, because the
# permutation folder name has no session token. Point to the saved model .mat:
model_mat = Path(r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\saveDataModels\sliding_window__110209d24_frame1-100__SVM_10foldCV____2026-06-15_12-38-00.mat")

# full-length significance (NOT the [0:45] cropped one)
sig01_full, _, _ = mc.sig_vector_perm_mean_across_folds(real_fold_trial_acc, null_trial_folds, alpha=0.05)

null = np.asarray(null_trial_folds, dtype=np.float64)
# the plotting collapses folds with mean over the LAST axis, so we want (n_perm, n_win, n_folds).
# if your null is (n_perm, n_folds, n_win), uncomment:
# null = np.transpose(null, (0, 2, 1))

# load existing model export, drop scipy's __header__ etc., add perm vars, re-save
D = {k: v for k, v in loadmat(str(model_mat)).items() if not k.startswith("__")}
D["real_trial_curve"] = np.asarray(real_fold_trial_acc, dtype=np.float64)          # (n_folds, n_win)
D["real_frame_curve"] = np.asarray(real_frame_acc,      dtype=np.float64).ravel()  # (n_win,)
D["null_trial_folds"] = null                                                       # (n_perm, n_win, n_folds)
D["sig01"]            = np.asarray(sig01_full, dtype=bool).reshape(-1, 1)          # (n_win, 1)
savemat(str(model_mat), D)
print("appended permutation results to", model_mat.name)