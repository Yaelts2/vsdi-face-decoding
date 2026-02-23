import numpy as np
import matplotlib.pyplot as plt
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
test_root = results_root / "fixed_window__frame32-40__SVM_10fold__2026-02-18_17-43-53__permutation__perm100__seed42__2026-02-20_01-59-40" # <-- update this to your model folder you want to load and plot results from
print("test_root:", test_root)
prem_results = sr.load_permutation_run(str(test_root))
mc.plot_permutation_test_trial(prem_results, bins=30, figsize=(7, 4), title='permutation test for fixed window run (frame 32-40) distribution of trial-level accuracy')
plt.show()
a=1