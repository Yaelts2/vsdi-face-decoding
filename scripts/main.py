# main.py 

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from preprocessing import (split_conds_files,
                        frame_zero_normalize_all_conds,
                        normalize_to_clean_blank,
                        build_X_y,
                        zscore_dataset)
from feature_extraction import extract_window, creat_ROI, mimg
from model_functions import flatten_data, split_data, svm_kfold,svm_kfold_nested
from model_evaluation import plot_accuracy_kfold_bars, plot_confusion_matrix, plot_roc_curve, plot_spatial_weights, permutation_test_linear_svm_fast

'''
# ================== optional preprocessing ==================
# 1) Split raw MATLAB -> per-condition .npy
split_conds_files(rew_dir=RAW_DIR,
                out_dir=SPLIT_DIR,
                overwrite=OVERWRITE,
                dtype=DTYPE)

# 2) Frame-zero normalization (trial-wise)
frame_zero_normalize_all_conds(in_dir=SPLIT_DIR,
                            out_dir=FZ_DIR,
                            zero_frames=ZERO_FRAMES,
                            overwrite=OVERWRITE,
                            dtype=DTYPE,
                            eps=EPS)

# 3) Normalize to blank (session-wise; blank=cond3)
normalize_to_clean_blank(blank_cond=BLANK_COND,
                        in_dir=FZ_DIR,
                        out_dir=BLANK_DIR,
                        overwrite=OVERWRITE,
                        dtype=DTYPE,
                        eps=EPS)
'''
###
'''
# define ROI once and save mask
x = np.load(r"C:\project\vsdi-face-decoding\data\processed\condsXn\condsXn5_270109b.npy")
x_avg = x.mean(axis=2)
x_avg_frames =  x_avg[:, 25:120]
fig,axes_flat =mimg(x_avg_frames-1, xsize=100, ysize=100, low=-0.0009, high=0.003,frames=range(25,120))
'''
'''
ROI_mask,roi_idx=creat_ROI(x_avg_frames, pixels=100)
# save ROI mask for future use
np.save(r"C:\project\vsdi-face-decoding\data\processed\ROI_mask.npy", ROI_mask)
'''
# ================== model pipeline ==================
# 4) Build dataset (X,y)
# X shape must be: (trials, pixels, frames)
X, y,metadata = build_X_y(face_file="condsXn1_270109b.npy",
                nonface_file="condsXn5_270109b.npy",
                data_dir="data/processed/condsXn/")
print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")

# 5) Z score across all trials
X_z,mean,std=zscore_dataset(X,
                            baseline_frames=(5, 25),
                            eps=1e-8)
print("Data z-scored across all trials.")
#6) feature extraction: window + ROI
X=extract_window(X_z, 34, 44)  # (10000 x 5 x trials)
print(f"Feature window extracted: {X.shape}")   
#ROI selection
ROI_mask=np.load(r"C:\project\vsdi-face-decoding\data\processed\ROI_mask.npy")
fig,axes_flat =mimg(ROI_mask, xsize=100, ysize=100, low=0, high=1)
plt.show()
print(f"ROI mask loaded: {ROI_mask.shape}")
X=X[ROI_mask,:,:] # (roi_pixels x 5 x trials)
print(f"ROI selected: {X.shape}")
#7) flatten data
X_flat=flatten_data(X) # (trials x features)
print(f"Data flattened: {X_flat.shape}")
#8) split data
X_train, X_test, y_train, y_test = split_data(X_flat, y, test_size=0.2)
print(f"Train/test split: {X_train.shape}, {X_test.shape}")

'''
#9) train / evaluate model
results,best_model =svm_kfold(
    X_train, y_train,
    n_splits=5,
    C_grid=(0.01, 0.1),
    kernel="linear",
    seed=0
)
'''
results,best_model=svm_kfold_nested(
    X_train, y_train,
    n_splits=5,
    C_grid=(0.01, 0.1),
    kernel="linear",
    seed=0)
print("Cross-validation results:")
for k, v in results.items():
    print(f"{k}: {v}")  
# Final evaluation on held-out test set
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred) 
print(f"Test set accuracy: {test_accuracy:.4f}")


#10) plot results
real_acc = np.mean(results["cv_fold_accuracy"])
plot_accuracy_kfold_bars(results["cv_fold_accuracy"], chance=0.5)

y_pred = best_model.predict(X_test)
plot_confusion_matrix(y_test, y_pred, class_names=("Non-face", "Face"))


# plot ROC curve
y_scores = best_model.decision_function(X_test)
fpr, tpr, auc_val = plot_roc_curve(y_true=y_test,y_scores=y_scores, title="Face vs Non-face ROC")
print("AUC:", auc_val)

# plot spatial weights
plot_spatial_weights(best_model, ROI_mask, n_frames=10, aggregate="mean")


# Quick Sanity Check: Shuffle the labels
shuffle_result = permutation_test_linear_svm_fast(X=X_train,
                                                y=y_train,
                                                real_acc=real_acc,
                                                n_perm=100,
                                                n_splits=5,
                                                C=0.01,
                                                seed=42,
                                                print_every=10,
                                                bins=30)
