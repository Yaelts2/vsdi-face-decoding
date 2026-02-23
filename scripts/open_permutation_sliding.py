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
results_root = Path(r"C:\project\vsdi-face-decoding\results\permutation_test_sliding_window")
test_root = results_root / "slidingwindow__perm100__seed42__2026-02-22_22-22-56" # <-- update this to your model folder you want to load and plot results from
print("test_root:", test_root)
prem_results = sr.load_sliding_window_permutation(str(test_root))
mc.plot_sliding_window_permutation_trial_level(prem_results)

a=1