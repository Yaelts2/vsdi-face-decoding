import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from functions_scripts import preprocessing_functions as pre
from functions_scripts import Weights_Evaluation as ev
from functions_scripts import ml_plots as pl
from functions_scripts import save_results as sr
from functions_scripts import feature_extraction as fe

ourCmap = pre.green_gray_magenta()


################ 
# Analysis for fixed window experiment
################

## user must edit these parameters for each run!##
### which model to load and plot results from
results_root = Path(r"C:\project\vsdi-face-decoding\results")
model_root = results_root / "fixed_window__frame32-40__SVM_10fold____2026-06-25_19-14-36" # <-- update this to your model folder you want to load and plot results from
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
y_true = np.asarray(results_fixed_window["oof_y_true"], dtype=int)
scores = np.asarray(results_fixed_window["oof_scores"], dtype=float)
a=11
# --- rebuild trial IDs (groups); depends only on array shape ---
_, y_frames_chk, groups = fe.frames_as_samples(X_win, y_trials,
                                               trial_axis=-1, frame_axis=1, pixel_axis=0)
assert np.array_equal(np.asarray(y_frames_chk, dtype=int), y_true), "sample order mismatch!"

# --- one score per trial = mean over its frames ---
trial_ids = np.unique(groups)
trial_score = np.array([scores[groups == t].mean() for t in trial_ids], dtype=float)
trial_label = np.array([y_true[groups == t][0] for t in trial_ids], dtype=int)

print("trial accuracy (sign of mean score):", float(np.mean((trial_score > 0).astype(int) == trial_label)))
print("trial accuracy (majority vote, saved):", float(results_fixed_window["outer_acc_trial_mean"]))


# --- presentation strip plot ---
face_col, nonface_col = "#C2185B", "#2E7D32"   # magenta / green, as in your weight maps
correct = (trial_score > 0).astype(int) == trial_label
n_correct, n_total = int(correct.sum()), int(correct.size)

# print misclassified trials (for your notes)
for i in np.where(~correct)[0]:
    print(f"misclassified trial {int(trial_ids[i])}: label={int(trial_label[i])}, "
          f"score={float(trial_score[i]):.4f}")

lim = 1.15 * float(np.max(np.abs(trial_score)))
rng = np.random.default_rng(0)

fig, ax = plt.subplots(figsize=(8, 3.8))

# shaded decision regions + boundary
ax.axvspan(-lim, 0, color=nonface_col, alpha=0.08, zorder=0)
ax.axvspan(0, lim, color=face_col, alpha=0.08, zorder=0)
ax.axvline(0, color="0.3", linestyle="--", linewidth=1.5, zorder=1)
ax.text(-lim * 0.97, 1.5, "decoded as non-face", color=nonface_col,
        fontsize=12, fontweight="bold", va="center")
ax.text(lim * 0.97, 1.5, "decoded as face", color=face_col,
        fontsize=12, fontweight="bold", va="center", ha="right")

# one dot per trial: filled = correct, hollow ring = misclassified
for lab, col, row in [(1, face_col, 1), (0, nonface_col, 0)]:
    m = trial_label == lab
    s, ok = trial_score[m], correct[m]
    y = row + rng.uniform(-0.14, 0.14, s.size)
    ax.scatter(s[ok], y[ok], s=80, color=col, edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter(s[~ok], y[~ok], s=95, facecolor="white", edgecolor=col, linewidth=2.2, zorder=4)

ax.set_xlim(-lim, lim)
ax.set_ylim(-0.5, 1.7)
ax.set_yticks([0, 1])
ax.set_yticklabels(["Non-face\ntrials", "Face\ntrials"], fontsize=14)
ax.set_xlabel("SVM decision score (one dot = one trial)", fontsize=13)

# title + held-out subtitle
ax.set_title(f"{n_correct}/{n_total} trials decoded correctly", fontsize=16,
             fontweight="bold", pad=28)
ax.text(0.5, 1.04, "Held-out trials · 10-fold cross-validation (nested)",
        transform=ax.transAxes, ha="center", fontsize=11, color="0.35")

ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", length=0)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)

plt.tight_layout()
fig.savefig("fixed_window_trial_scores.png", dpi=300, bbox_inches="tight", transparent=True)
plt.show()
###############################

'''
# =========================
# PLOTS 
# =========================

# 1) Accuracy bars across outer folds
pl.plot_frame_vs_trial_bars(results_fixed_window, chance=0.5, ylim=(0.4, 1.0))

# 2) Confusion matrix from outer OOF predictions
pl.plot_confusion_matrix(results=results_fixed_window,class_names=("Non-face", "Face"),
                        title="Frame-level confusion matrix",
                        figsize=(5.6, 4.9),
                        show_counts=True,
                        colormap='Purples',)
pl.plot_confusion_matrix(y_true=results_fixed_window["oof_y_true_trial"],
                        y_pred=results_fixed_window["oof_pred_trial"],
                        title="Trial-level confusion matrix",
                        colormap='Blues',)

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



'''