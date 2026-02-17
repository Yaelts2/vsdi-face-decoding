import numpy as np
import matplotlib.pyplot as plt
from scripts.functions_scripts import preprocessing_functions as pre
from scripts.functions_scripts import ml_cv as cv
from scripts.functions_scripts import ml_plots as pl
from scripts.functions_scripts import feature_extraction as fe
from scripts.functions_scripts.save_results import save_experiment
from sklearn.model_selection import GroupKFold



# =========================
# CONFIG
# =========================

# Data files:
face_file= "condsXn1_110209a.npy"
nonface_file= "condsXn5_110209a.npy"
data_dir= "data/processed/condsXn/"

# Preprocessing:
Baseline_frames_zscore= (1, 24)

# Feature extraction:
window= (32, 40)  #(start, end)
ROI_mask_path= "data/processed/ROI_mask2.npy"


# Model / validation:
outer_cv= "GroupKFold(n_splits=10)"
inner_cv= "GroupKFold(n_splits=4)"
outer = GroupKFold(n_splits=10)
inner = GroupKFold(n_splits=4)
C_grid = np.logspace(-4, 2, 7) 
metric = "acc"  # "acc" or "auc"
rule = "one_se"                
tie_break = "smaller_C"     


# save model results path:
results_root = "C:\\project\\vsdi-face-decoding\\results"


# Expected core shapes:
# - X_trials: (pixels, frames, trials)
# - y_trials: (trials,)
# - X_frames: (samples, features)   # samples = trials x frames
# - y_frames : (samples,)           # samples = trials x frames
# - groups: (samples,)              # samples = trials x frames




# ================== preparing the data ==================
# 1) Build dataset (X,y)
# X shape must be: (trials, pixels, frames)
X_trials, y_trials,dataset_info = pre.build_X_y(face_file=face_file,
                                                nonface_file=nonface_file,
                                                data_dir=data_dir)
print(f"Dataset shape: {X_trials.shape}, Labels shape: {y_trials.shape}")

# ) Z score across all trials
X_z,mean,std=pre.zscore_dataset_pixelwise_trials(X_trials,Baseline_frames_zscore)
b = X_z[:, Baseline_frames_zscore[0]:Baseline_frames_zscore[1], :]
print(np.nanmean(b), np.nanstd(b))  # should be close to 0  and 1 (global check)
print("Data z-scored across all trials.")
#a=np.nanmean(X_z, axis=(2))
#binned, fig, axes, cid = pl.plot_superpixel_traces(a[:,25:125], xs=100, ys=100, nsubplots=15)
#plt.show()


#feature extraction: window + ROI
##ROI selection
ROI_mask=np.load(ROI_mask_path)  # boolean mask in full image space (10000,) or (100,100)
pl.mimg(ROI_mask, xsize=100, ysize=100, low=0, high=1)
print(f"ROI mask loaded: {ROI_mask.shape}")
X_roi=X_z[ROI_mask,:,:] # (roi_pixels x frames x trials)
print(f"ROI selected: {X_roi.shape}")

##window selection
X_win=fe.extract_window(X_roi, window[0],window[1])  # (10000 x window.size x trials)
print(f"Feature window extracted: {X_win.shape}")   

# flatten data
## X_avgWindow=fe.avgWindow(X_win) # (trials x features)
#print(f"Data averaged over window: {X_avgWindow.shape}")
## alternative: frames-as-samples
X_frames ,y_frames,groups=fe.frames_as_samples(X_win,y_trials, trial_axis=-1, frame_axis=1, pixel_axis=0)
print(f"Data : {X_frames.shape}")
print(f"Labels shape: {y_frames.shape}")
print(f"Groups shape: {groups.shape}")



############################# main_nested_cv###################################
assert X_frames.shape[0] == y_frames.shape[0] == groups.shape[0], "X, y, groups must align in n_samples"
pl.plot_groupkfold_splits(y_frames,groups,outer)


# =========================
# RUN NESTED CV
# =========================
nested = cv.run_nested_cv_selectC_then_eval(X_frames, y_frames,groups=groups,outer_splitter=outer,inner_splitter=inner,
                                            C_grid=C_grid,
                                            metric=metric,
                                            rule=rule,
                                            tie_break=tie_break,
                                            n_jobs_inner=1,
                                            verbose=True)

print("\n=== Nested CV summary ===")
print(f"Outer acc: {nested['outer_acc_mean']:.4f} ± {nested['outer_acc_std']:.4f}")
print(f"Outer auc: {nested['outer_auc_mean']:.4f} ± {nested['outer_auc_std']:.4f}")
print(f"Chosen Cs per outer fold: {nested['chosen_Cs']}")
print(f"Recommended final_C: {nested['final_C']}\n")
print("Trial accuracy:", nested["outer_acc_trial_mean"])
print("Frame accuracy:", nested["outer_acc_mean"])



# =========================
# SAVE RESULTS
# =========================

dataset_info.update({"face_file": face_file,
                    "nonface_file": nonface_file,
                    "data_dir": data_dir,
                    "window": window,
                    "zscore_baseline_frames": Baseline_frames_zscore,
                    "metric": metric,
                    "C_grid": list(C_grid),
                    "rule": rule,
                    "tie_break": tie_break,
                    "outer_cv": outer_cv,
                    "inner_cv": inner_cv})

run_dir = save_experiment(results_root=results_root,
                        experiment="fixed_window",
                        experiment_tag=f"frame{window[0]}-{window[1]}__SVM_10fold",
                        results=nested,
                        ROI_mask=ROI_mask_path,
                        dataset_info=dataset_info)

print(f"\nResults saved to: {run_dir}")
