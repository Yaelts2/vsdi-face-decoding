import numpy as np
import matplotlib.pyplot as plt
from functions_scripts import preprocessing_functions as pre
from functions_scripts import ml_cv as cv
from functions_scripts import ml_evaluation as ev
from functions_scripts import ml_plots as pl
from functions_scripts import feature_extraction as fe
from functions_scripts import movie_function as mf
from sklearn.model_selection import GroupKFold
from functions_scripts import sliding_win as sw



# ================== preparing the data ==================
# 1) Build dataset (X,y)
# X shape must be: (trials, pixels, frames)
X_trials, y_trials,metadata = pre.build_X_y(face_file="condsXn1_110209a.npy",
                nonface_file="condsXn5_110209a.npy",
                data_dir="data/processed/condsXn/")
print(f"Dataset shape: {X_trials.shape}, Labels shape: {y_trials.shape}")

# ) Z score across all trials
X_z,mean,std=pre.zscore_dataset_pixelwise_trials(X_trials,baseline_frames=(1, 24))
b = X_z[:, 2:26, :]
print(np.nanmean(b), np.nanstd(b))  # should be close to 0  and 1 (global check)
print("Data z-scored across all trials.")
a=np.nanmean(X_z, axis=(2))
#binned, fig, axes, cid = pl.plot_superpixel_traces(a[:,25:125], xs=100, ys=100, nsubplots=15)
#plt.show()
#6) feature extraction: window + ROI
#ROI selection
ROI_mask=np.load(r"C:\project\vsdi-face-decoding\data\processed\ROI_onlyV24_mask.npy")
fig,axes_flat =pl.mimg(ROI_mask, xsize=100, ysize=100, low=0, high=1)
plt.show()
print(f"ROI mask loaded: {ROI_mask.shape}")
X_roi=X_z[ROI_mask,:,:] # (roi_pixels x 5 x trials)
print(f"ROI selected: {X_roi.shape}")
# window selection
window=(60,70)  # frames to use for feature extraction 
X_win=fe.extract_window(X_roi, window[0],window[1])  # (10000 x 5 x trials)
print(f"Feature window extracted: {X_win.shape}")   

#7) flatten data
X_avgWindow=fe.avgWindow(X_win) # (trials x features)
print(f"Data averaged over window: {X_avgWindow.shape}")
# alternative: frames-as-samples
X_frames ,y_frames,groups=fe.frames_as_samples(X_win,y_trials, trial_axis=-1, frame_axis=1, pixel_axis=0)
print(f"Data : {X_frames.shape}")
print(f"Labels shape: {y_frames.shape}")
print(f"Groups shape: {groups.shape}")


'''
############################# main_nested_cv###################################
ourCmap = pre.green_gray_magenta()

X = np.asarray(X_frames)
y = np.asarray(y_frames).astype(int).ravel()
groups = np.asarray(groups).ravel()

assert X.shape[0] == y.shape[0] == groups.shape[0], "X, y, groups must align in n_samples"


# =========================
# NESTED CV CONFIG
# =========================
outer = GroupKFold(n_splits=10)
inner = GroupKFold(n_splits=4)
pl.plot_groupkfold_splits(y,groups,outer)

C_grid = np.logspace(-4, 2, 7)  # adjust if you want denser grid
metric = "acc"                  # "acc" or "auc"
rule = "one_se"                 # recommended (stability / anti-overfit)
tie_break = "smaller_C"         # recommended


# =========================
# RUN NESTED CV
# =========================
nested = cv.run_nested_cv_selectC_then_eval(X, y,groups=groups,outer_splitter=outer,inner_splitter=inner,
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


# =========================
# PLOTS 
# =========================

# 1) Accuracy bars across outer folds
pl.plot_accuracy_kfold_bars(nested,chance=0.5,title="Nested CV: accuracy per outer fold",
                            figsize=(6.5, 4.2),
                            ylim=(0.4, 1.0))
pl.plot_accuracy_kfold_bars([f["acc_trial"] for f in nested["outer_folds"]], chance=0.5,
                        title="Trial accuracy (majority vote) per outer fold")

pl.plot_frame_vs_trial_bars(nested, chance=0.5, ylim=(0.4, 1.0))

# 2) Confusion matrix from outer OOF predictions
pl.plot_confusion_matrix(results=nested,class_names=("Non-face", "Face"),
                        title="Nested CV: OOF confusion matrix",
                        figsize=(5.6, 4.9),
                        show_counts=True)

# 3) ROC from outer OOF scores (only where score exists)
try:
    pl.plot_roc_curve(results=nested,title="Nested CV: OOF ROC curve",figsize=(5.6, 4.9))
except Exception as e:
    print(f"[ROC] Skipped: {e}")



W_outer = nested["W_outer"]
ev.plot_all_fold_weight_maps(W_outer, ROI_mask, pixels=100,
                            n_cols=5,
                            cmap=ourCmap,
                            clip=(-0.0002, 0.0002))
stats = ev.plot_weight_stat_maps(W_outer, ROI_mask, pixels=100)


mean_weightmap= stats['mean']
frac=0.20
positive_mask, negative_mask = ev.extract_extreme_weight_masks(mean_weightmap, ROI_mask, pixels=100, frac=frac)
pos_img = positive_mask.reshape(100, 100).astype(float)
neg_img = negative_mask.reshape(100, 100).astype(float)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ev.draw_weight_map(axes[0], pos_img, cmap="Reds", clip=(0, 1), title=f"Top {frac*100:.0f}% positive", show_colorbar=False)
ev.draw_weight_map(axes[1], neg_img, cmap="Blues", clip=(0, 1), title=f"Bottom {frac*100:.0f}% negative", show_colorbar=False)
plt.tight_layout()
plt.show()


neg_mask= negative_mask[:,np.newaxis,np.newaxis]
X_negtive=np.where(neg_mask, X_trials[:,window[0]:window[1]],np.nan)
X_negtive=X_negtive.mean(axis=2)
pl.mimg(X_negtive-1, xsize=100,ysize=100, low=-0.0009, high=0.003,frames=range(window[0],window[1]))

pos_mask= positive_mask[:,np.newaxis,np.newaxis]
X_positive=np.where(pos_mask, X_trials[:,window[0]:window[1]],np.nan)
X_positive=X_positive.mean(axis=2)
pl.mimg(X_positive-1, xsize=100,ysize=100, low=-0.0009, high=0.003,frames=range(window[0],window[1]))


positive_tc_face, negative_tc_face = ev.average_activation_by_weight_sign(X_win[:,:,0:29],ROI_mask,
                                                                        positive_mask,
                                                                        negative_mask)

positive_tc_nonface, negative_tc_nonface = ev.average_activation_by_weight_sign(X_win[:,:,30:59],ROI_mask,
                                                                                positive_mask,
                                                                                negative_mask)

ev.plot_pos_neg_timecourses(positive_tc_face, negative_tc_face,frame_times=np.arange(window[0],window[1]), title="Face trials: positive vs negative weight pixels")
ev.plot_pos_neg_timecourses(positive_tc_nonface, negative_tc_nonface,frame_times=np.arange(window[0],window[1]), title="Non-face trials: positive vs negative weight pixels")
print(positive_mask.sum(), negative_mask.sum())
'''



# --- Run sliding window decoding ---
results = sw.sliding_window_decode_with_stats(X_roi,      # shape: (8518, 256, 56)
                                            y_trials,          # shape: (56,)
                                            make_estimator=lambda: cv.make_linear_svm(C=0.0001, max_iter=100000),
                                            window_size=5,
                                            start_frame=15,
                                            stop_frame=125,
                                            step=1,
                                            n_splits=5)

# --- Plot results ---
sw.plot_sliding_window_accuracy_with_sem(res=results,
                                        chance=0.5,
                                        title="Sliding Window Decoding (5-frame window)")

print("Peak frame accuracy:",
    results["frame_acc_mean"].max())

print("Peak trial accuracy:",
    results["trial_acc_mean"].max())

peak_idx = results["trial_acc_mean"].argmax()
print("Peak window center frame:",
    results["centers"][peak_idx])

W_img=ev.window_weights_to_pixel_time_matrix(results["w_mean_windows"], ROI_mask, pixels=100)

                                            
print(W_img.shape)
pl.mimg(W_img, xsize=100, ysize=100, low=-0.0002, high=0.0002, frames=results["centers"])




mf.show_weight_movie(W_pixel_time=W_img,
                    centers=results["centers"],
                    frame_acc=results["frame_acc_mean"],
                    trial_acc=results["trial_acc_mean"],
                    pixels=100,
                    fps=10,
                    clip_q=90,
                    title="Mean weight map (across folds) per window")




a=1

