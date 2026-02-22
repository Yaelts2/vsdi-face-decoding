import numpy as np
from pathlib import Path
from functions_scripts import preprocessing_functions as pre
from functions_scripts import save_results as sr
from functions_scripts import model_control as mc

ourCmap = pre.green_gray_magenta()


################ 
# ploting the permutation results for a fixed window run
################


### which permutation test to load and plot results from
results_root = Path(r"C:\project\vsdi-face-decoding\results\permutation_test")
test_root = results_root / "fixed_window__frame32-40__SVM_10fold__2026-02-18_17-43-53__permutation__perm2__seed42__2026-02-22_10-51-27" # <-- update this to your model folder you want to load and plot results from
print("test_root:", test_root)
prem_results = sr.load_sliding_window_permutation(str(test_root))
mc.plot_sliding_window_permutation_trial_level(prem_results,
                                                chance=0.5,
                                                title="Sliding-window permutation test (trial-level)",
                                                figsize=(7, 4),
                                                ylim=(0.4, 1.0),
                                                show_null_curves=True,
                                                null_alpha=0.08)

a=1