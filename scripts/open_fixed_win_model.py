import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from functions_scripts import preprocessing_functions as pre
from functions_scripts import ml_evaluation as ev
from functions_scripts import ml_plots as pl
from functions_scripts import save_results as sr

ourCmap = pre.green_gray_magenta()




'''
########## OPTIONAL: Leave one group pair out splits for outer CV (instead of GroupKFold) ##########

##########
C_grid = np.logspace(-4, 2, 7)
metric = "acc"
rule = "one_se"
tie_break = "smaller_C"

# Outer: leave-one-(face group, nonface group)-out
outer = lambda y_frames, groups: cv.leave_one_group_pair_out_splits(y_frames, groups, seed=0)
inner = GroupKFold(n_splits=4)

nested = cv.run_nested_cv_selectC_then_eval(
    X_frames, y_frames,
    groups=groups,
    outer_splitter=outer,
    inner_splitter=inner,
    C_grid=C_grid,
    metric=metric,
    rule=rule,
    tie_break=tie_break,
)
##########
'''


################ 
# Analysis for fixed window experiment
################


### which model to load and plot results from
results_root = Path(r"C:\project\vsdi-face-decoding\results")
model_root = results_root / "fixed_window__frame32-40__SVM_10fold__2026-02-18_17-43-53" # <-- update this to your model folder you want to load and plot results from
print("model_root:", model_root)


### Load the experiment results and config
config_fixed_window, results_fixed_window, ROI_mask_path = sr.load_experiment(str(model_root))
trials_per_cond = config_fixed_window.get("n_trials_per_class",28)
print("Experiment config:", config_fixed_window)
### prepering data that was used for this experiment (e.g. for plotting weight maps, etc.)
#load the data
X_trials, y_trials = sr.load_data_from_config(config_fixed_window,)
print(X_trials.shape, y_trials.shape)
#load the ROI mask and apply it to the data
ROI_mask = np.load(ROI_mask_path)
print("ROI_mask shape:", ROI_mask.shape)
X_ROI = X_trials[ROI_mask,:,:]
print
#apply window that was used for this experiment
window = config_fixed_window["window"]
window = (int(window[0]), int(window[1]))
X_win = X_ROI[:, window[0]:window[1], :]
print("X_trials after windowing:", X_win.shape)

a=11

# =========================
# PLOTS 
# =========================

# 1) Accuracy bars across outer folds
pl.plot_frame_vs_trial_bars(results_fixed_window, chance=0.5, ylim=(0.4, 1.0))

# 2) Confusion matrix from outer OOF predictions
pl.plot_confusion_matrix(results=results_fixed_window,class_names=("Non-face", "Face"),
                        title="Nested CV: OOF confusion matrix",
                        figsize=(5.6, 4.9),
                        show_counts=True)

# 3) ROC from outer OOF scores (only where score exists)
try:
    pl.plot_roc_curve(results=results_fixed_window,title="Nested CV: OOF ROC curve",figsize=(5.6, 4.9))
except Exception as e:
    print(f"[ROC] Skipped: {e}")

# 4) Weight maps from outer CV (mean across folds)
W_outer = results_fixed_window["W_outer"]
ev.plot_all_fold_weight_maps(W_outer, ROI_mask, pixels=100,
                            n_cols=5,
                            cmap=ourCmap,
                            clip=(-0.0002, 0.0002))


# 5) Mean weight map across folds + extract top positive and negative pixels
stats = ev.plot_weight_stat_maps(W_outer, ROI_mask, pixels=100)


# 6) Extract top positive and negative weight pixels 
mean_weightmap= stats['mean']
frac=0.20
positive_mask, negative_mask = ev.extract_extreme_weight_masks(mean_weightmap, ROI_mask, pixels=100, frac=frac)
pos_img = positive_mask.reshape(100, 100).astype(float)
neg_img = negative_mask.reshape(100, 100).astype(float)
# plot the positive and negative weight masks
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ev.draw_weight_map(axes[0], pos_img, cmap="Reds", clip=(0, 1), title=f"Top {frac*100:.0f}% positive", show_colorbar=False)
ev.draw_weight_map(axes[1], neg_img, cmap="Blues", clip=(0, 1), title=f"Bottom {frac*100:.0f}% negative", show_colorbar=False)
plt.tight_layout()
plt.show()

# 7) Timecourses of positive vs negative weight pixels (using the masks from #6)
neg_mask= negative_mask[:,np.newaxis,np.newaxis]
X_negtive=np.where(neg_mask, X_trials[:,window[0]:window[1]],np.nan)
X_negtive=X_negtive.mean(axis=2)
pl.mimg(X_negtive-1, xsize=100,ysize=100, low=-0.0009, high=0.003,frames=range(window[0],window[1]))

pos_mask= positive_mask[:,np.newaxis,np.newaxis]
X_positive=np.where(pos_mask, X_trials[:,window[0]:window[1]],np.nan)
X_positive=X_positive.mean(axis=2)
pl.mimg(X_positive-1, xsize=100,ysize=100, low=-0.0009, high=0.003,frames=range(window[0],window[1]))


positive_tc_face, negative_tc_face = ev.average_activation_by_weight_sign(X_win[:,:,0:trials_per_cond],ROI_mask,
                                                                        positive_mask,
                                                                        negative_mask)

positive_tc_nonface, negative_tc_nonface = ev.average_activation_by_weight_sign(X_win[:,:,trials_per_cond:],ROI_mask,
                                                                                positive_mask,
                                                                                negative_mask)

ev.plot_pos_neg_timecourses(positive_tc_face, negative_tc_face,frame_times=np.arange(window[0],window[1]), title="Face trials: positive vs negative weight pixels")
ev.plot_pos_neg_timecourses(positive_tc_nonface, negative_tc_nonface,frame_times=np.arange(window[0],window[1]), title="Non-face trials: positive vs negative weight pixels")
print(positive_mask.sum(), negative_mask.sum())



