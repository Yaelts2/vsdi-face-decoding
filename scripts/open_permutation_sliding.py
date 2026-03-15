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


### which permutation test to load and plot results from
results_root = Path(r"C:\project\vsdi-face-decoding\results\permutation_test_sliding_window")
test_root = results_root / "slidingwindow__perm1000__seed42__2026-02-26_17-34-13" # <-- update this to your model folder you want to load and plot results from
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

a=1