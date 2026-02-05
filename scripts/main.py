# main.py 

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from preprocessing_functions import build_X_y, zscore_dataset
from feature_extraction import extract_window, mimg, avgWindow
from sklearn.model_selection import GroupKFold, StratifiedKFold
from model_functions import frames_as_samples, leave_one_pair_out_splits, make_linear_svm, run_cv, select_best_C




# ================== model pipeline ==================
# 4) Build dataset (X,y)
# X shape must be: (trials, pixels, frames)
X_trials, y_trials,metadata = build_X_y(face_file="condsXn1_270109b.npy",
                nonface_file="condsXn5_270109b.npy",
                data_dir="data/processed/condsXn/")
print(f"Dataset shape: {X_trials.shape}, Labels shape: {y_trials.shape}")

# 5) Z score across all trials
X_z,mean,std=zscore_dataset(X_trials,
                            baseline_frames=(2, 26),
                            eps=1e-8)
print("Data z-scored across all trials.")
print(X_z.shape)
#6) feature extraction: window + ROI
X_win=extract_window(X_z, 3, 26)  # (10000 x 5 x trials)
print(f"Feature window extracted: {X_win.shape}")   
#ROI selection
ROI_mask=np.load(r"C:\project\vsdi-face-decoding\data\processed\ROI_mask2.npy")
fig,axes_flat =mimg(ROI_mask, xsize=100, ysize=100, low=0, high=1)
plt.show()
print(f"ROI mask loaded: {ROI_mask.shape}")
X_roi=X_win[ROI_mask,:,:] # (roi_pixels x 5 x trials)
print(f"ROI selected: {X_roi.shape}")
#7) flatten data
X_avgWindow=avgWindow(X_roi) # (trials x features)
print(f"Data averaged over window: {X_avgWindow.shape}")
# alternative: frames-as-samples
X_frames ,y_frames,groups=frames_as_samples(X_roi,y_trials, trial_axis=-1, frame_axis=1, pixel_axis=0)
print(f"Data : {X_frames.shape}")
print(f"Labels shape: {y_frames.shape}")
print(f"Groups shape: {groups.shape}")

'''
##### option 1 : input is avg over frames in window  #####
#8) split data
#X_train, X_test, y_train, y_test = split_data(X_avgWindow, y_trials, test_size=0.2)
#print(f"Train/test split: {X_train.shape}, {X_test.shape}")
#9) train / evaluate model
splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
results_avgWindow=run_cv(X=X_avgWindow, y=y_trials, groups=None,
                splitter=splitter,
                plot_splits=True,
                split_title="StratifiedKFold (avg-window)",
                expect_full_coverage=True,
                n_jobs=1,          # start with 1; later try 2 or -1
                verbose=True)
print("\n=== CV Summary (avg-window) ===")
print(f"Mean accuracy: {results_avgWindow['acc_mean']:.4f} ± {results_avgWindow['acc_std']:.4f}")
print(f"Mean AUC:      {results_avgWindow['auc_mean']:.4f} ± {results_avgWindow['auc_std']:.4f}")
print("\nConfusion matrix (OOF predictions):")
print(results_avgWindow["confusion_matrix"])    
plot_accuracy_kfold_bars([f["acc"] for f in results_avgWindow["folds"]], chance=0.5, title="StratifiedKFold CV Accuracy (avg-window)")
plot_confusion_matrix(y_true=y_trials,y_pred=results_avgWindow["oof_pred"],class_names=("Non-face", "Face"),title="StratifiedKFold (OOF) confusion matrix")
plot_roc_curve(y_true=y_trials,y_scores=results_avgWindow["oof_scores"],title="StratifiedKFold ROC (OOF scores)")
figs, summary = plot_weight_stability(results_avgWindow, ROI_mask, pixels=100, prefix="Avg-window")
print(summary)
'''

'''
##### option 2 frames as samples with GroupKFold CV 
# 1) Build CV splitter
cv = GroupKFold(n_splits=5)
# 2) Run CV
best_C, table = select_best_C(X=X_frames, y=y_frames, splitter=cv, C_grid=np.logspace(-4, 2, 7), groups=groups, metric="acc",# "acc" or "auc"
                tie_break="smaller_C",        # "smaller_C" or "larger_C"
                expect_full_coverage=False,   # inner CV is often not full-coverage (e.g., repeated CV)
                n_jobs=1,
                verbose=True)
print(f"\nBest C selected: {best_C}")
print("C selection results:")
print(table)

results = run_cv(X=X_frames, y=y_frames, groups=groups,
        splitter=cv,
        make_estimator=lambda: make_linear_svm(C=best_C),
        expect_full_coverage=True,
        n_jobs=1,
        verbose=True,
        plot_splits=False,
        split_title="CV splits")

# 3) Print summary
print("\n=== CV Summary ===")
print(f"Mean accuracy: {results['acc_mean']:.4f} ± {results['acc_std']:.4f}")
print(f"Mean AUC:      {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")
print("\nConfusion matrix (OOF predictions):")
print(results["confusion_matrix"])
accs = [f["acc"] for f in results["folds"]]
plot_accuracy_kfold_bars(accs, chance=0.5, title="GroupKFold CV Accuracy (frames-as-samples)")
plot_confusion_matrix(y_true=y_frames,y_pred=results["oof_pred"],class_names=("Non-face", "Face"),title="GroupKFold (OOF) confusion matrix")
plot_roc_curve(y_true=y_frames,y_scores=results["oof_scores"],title="GroupKFold ROC (OOF scores)")
figs, summary = plot_weight_stability(results, ROI_mask, pixels=100, prefix="groupKfolds-window")
print(summary)
'''





'''
#### option 3 frames as samples with Leave-One-Pair-Out CV
# 1) Build L-O-P-O splitter
lopo = leave_one_pair_out_splits(y_trials=y_trials, n_frames=10)
# 2) Run CV
results_lopo = run_cv(X=X_frames, y=y_frames, groups=groups,          
                    splitter=lopo,                
                    plot_splits=True,              # will print "no plotting available"
                    split_title="Leave-one-pair-out",
                    expect_full_coverage=False,    # may not cover all samples with LOPO
                    n_jobs=2,                      
                    verbose=False)

print("\n=== Leave-One-Pair-Out Summary ===")
print(f"Mean accuracy: {results_lopo['acc_mean']:.4f} ± {results_lopo['acc_std']:.4f}")
print(f"Mean AUC:      {results_lopo['auc_mean']:.4f} ± {results_lopo['auc_std']:.4f}")
accs_lopo = [f["acc"] for f in results_lopo["folds"]]
plot_lopo_metric_hist(accs_lopo, title="LOPO accuracy distribution", xlabel="Accuracy")
figs, summary = plot_weight_stability(results_lopo, ROI_mask, pixels=100, prefix="LOPO-window")
print(summary)
'''






# main_nested_cv.py

import ml_cv as cv
import ml_plots as pl
import ml_evaluation as ev
from preprocessing_functions import green_gray_magenta
ourCmap = green_gray_magenta()

X = np.asarray(X_frames)
y = np.asarray(y_frames).astype(int).ravel()
groups = np.asarray(groups).ravel()

assert X.shape[0] == y.shape[0] == groups.shape[0], "X, y, groups must align in n_samples"


# =========================
# NESTED CV CONFIG
# =========================
outer = GroupKFold(n_splits=5)
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



'''
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
'''


# =========================
# PLOTS 
# =========================

# 1) Accuracy bars across outer folds
pl.plot_accuracy_kfold_bars(nested,chance=0.5,title="Nested CV: accuracy per outer fold",
                            figsize=(6.5, 4.2),
                            ylim=(0.4, 1.0))

# 2) Confusion matrix from outer OOF predictions
pl.plot_confusion_matrix(results=nested,class_names=("Non-face", "Face"),
                        title="Nested CV: OOF confusion matrix",
                        figsize=(5.6, 4.9),
                        show_counts=True)

# 3) ROC from outer OOF scores (only where score exists)
try:
    pl.plot_roc_curve(results=nested,title="Nested CV: OOF ROC curve",
                    figsize=(5.6, 4.9))
except Exception as e:
    print(f"[ROC] Skipped: {e}")

''' not sure i need this , i think to use only the weights across folds
# =========================
# FINAL MODEL (for weight map)
# =========================
final = cv.fit_final_model(X, y, nested["final_C"])
w = final["w"]  # weights in original feature units (Pipeline-safe)

print("Final model fitted.")
print(f"C_final = {final['C']}")
print(f"w shape = {None if w is None else w.shape}")
print(w.shape, ROI_mask.sum())
final = cv.fit_final_model(X, y, nested["final_C"])
w_finel = final["w"]

print("w length:", w_finel.size)
print("ROI pixels:", ROI_mask.sum())
'''

W_outer = nested["W_outer"]
ev.plot_all_fold_weight_maps(W_outer, ROI_mask, pixels=100,
                            n_cols=5,
                            cmap=ourCmap,
                            clip=(-0.0002, 0.0002))
stats = ev.plot_weight_stat_maps(W_outer, ROI_mask, pixels=100)


mean_weightmap= stats['mean']
frac=0.30
positive_mask, negative_mask = ev.extract_extreme_weight_masks(mean_weightmap, ROI_mask, pixels=100, frac=frac)
pos_img = positive_mask.reshape(100, 100).astype(float)
neg_img = negative_mask.reshape(100, 100).astype(float)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ev.draw_weight_map(axes[0], pos_img, cmap="Reds", clip=(0, 1), title=f"Top {frac*100:.0f}% positive", show_colorbar=False)
ev.draw_weight_map(axes[1], neg_img, cmap="Blues", clip=(0, 1), title=f"Bottom {frac*100:.0f}% negative", show_colorbar=False)
plt.tight_layout()
plt.show()


neg_mask= negative_mask[:,np.newaxis,np.newaxis]
X_negtive=np.where(neg_mask, X_win,np.nan)
X_negtive=X_negtive.mean(axis=2)
mimg(X_negtive, xsize=100,ysize=100)

pos_mask= positive_mask[:,np.newaxis,np.newaxis]
X_positive=np.where(pos_mask, X_win,np.nan)
X_positive=X_positive.mean(axis=2)
mimg(X_positive, xsize=100,ysize=100)


positive_tc_face, negative_tc_face = ev.average_activation_by_weight_sign(X_roi[:,:,0:29],ROI_mask,
                                                                positive_mask,
                                                                negative_mask)

positive_tc_nonface, negative_tc_nonface = ev.average_activation_by_weight_sign(X_roi[:,:,30:59],ROI_mask,
                                                                positive_mask,
                                                                negative_mask)

ev.plot_pos_neg_timecourses(positive_tc_face, negative_tc_face,frame_times=np.arange(34,44))
ev.plot_pos_neg_timecourses(positive_tc_nonface, negative_tc_nonface,frame_times=np.arange(34,44))
print(positive_mask.sum(), negative_mask.sum())