from pathlib import Path
import sys
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
print("Project root:", project_root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import matplotlib.pyplot as plt
from functions_scripts import preprocessing_functions as pre
from functions_scripts import ml_evaluation as ev
from functions_scripts import ml_plots as pl
from functions_scripts import save_results as sr
from functions_scripts import sliding_win as sw
from functions_scripts import movie_function as mf

ourCmap = pre.green_gray_magenta()
###############
# Analysis for sliding window experiment #
###############


### which model to load and plot results from

results_root = Path(r"C:\project\vsdi-face-decoding\results")
model_root = results_root / "sliding_window__frame0-125__SVM_5foldCV__2026-02-24_17-34-36" # <-- update this to your model folder you want to load and plot results from
print("model_root:", model_root)


### Load the experiment results and config
config_sliding_window, results_sliding_window, ROI_mask_path = sr.load_experiment(str(model_root))
trials_per_cond = config_sliding_window.get("n_trials_per_class",28)
print("Experiment config:", config_sliding_window)
### prepering data that was used for this experiment (e.g. for plotting weight maps, etc.)
#load the data
X_trials, y_trials = sr.load_data_from_config(config_sliding_window)
print(X_trials.shape, y_trials.shape)
#load the ROI mask and apply it to the data
ROI_mask = np.load(ROI_mask_path)
print("ROI_mask shape:", ROI_mask.shape)
X_ROI = X_trials[ROI_mask,:,:]
print("X_ROI shape:", X_ROI.shape)


# =========================
# PLOTS 
# =========================

W_img=ev.window_weights_to_pixel_time_matrix(results_sliding_window["w_mean_windows"], ROI_mask, pixels=100)
print(W_img.shape) #(pixels, frames) where frames correspond to window centers
pl.mimg(W_img, xsize=100, ysize=100, low=-0.0002, high=0.0002, frames=results_sliding_window["centers"])
frame_ids = np.arange(0, 43)         
weights=W_img[:,0:43]
X_avg, labels_ms, bin_frames = pl.avg_consecutive_frames_with_ms_labels(X=weights, frame_ids=frame_ids, avg_n=2, dt_ms=10, zero_frame=27
)
fig,axes_flat =pl.mimg(X_avg, xsize=100, ysize=100, low=-0.002, high=0.002,frames=labels_ms.astype(int), width=14,colormap=ourCmap)
plt.show()


#mimg of positive and negative weight masks across windows
pos_masks, neg_masks = ev.extract_extreme_weight_masks_sliding(W_img, ROI_mask,  frac=0.20)
pl.mimg(pos_masks, xsize=100, ysize=100, low='auto', high=None, frames=results_sliding_window["centers"])
pl.mimg(neg_masks, xsize=100, ysize=100, low=0, high=1, frames=results_sliding_window["centers"])


sw.plot_sliding_window_accuracy_with_std(
    results_sliding_window,
    zero_frame=27,          # frame 27 = 0 ms
    frame_duration_ms=10,   # 10 ms per frame
)


# weights movie
mf.show_weight_movie(W_pixel_time=W_img,
                    centers=results_sliding_window["centers"],
                    frame_acc=results_sliding_window["frame_acc_mean"],
                    trial_acc=results_sliding_window["trial_acc_mean"],
                    pixels=100,
                    fps=10,
                    clip_q=90,
                    title="Mean weight map (across folds) per window")

a=1


#################
# Analysis for sliding window experiment only V 2 ,4  #
##################


'''
### which model to load and plot results from
results_root = Path(r"C:\project\vsdi-face-decoding\results")
model_root = results_root / "sliding_window__V24_frame15-125__SVM_5foldCV__2026-02-17_16-09-38" # <-- update this to your model folder you want to load and plot results from
print("model_root:", model_root)


### Load the experiment results and config
config_sliding_window, results_sliding_window, ROI_mask_path = sr.load_experiment(str(model_root))
trials_per_cond = config_sliding_window.get("n_trials_per_class",28)
print("Experiment config:", config_sliding_window)
### prepering data that was used for this experiment (e.g. for plotting weight maps, etc.)
#load the data
X_trials, y_trials = sr.load_data_from_config(config_sliding_window)
print(X_trials.shape, y_trials.shape)
#load the ROI mask and apply it to the data
ROI_mask = np.load(ROI_mask_path)
print("ROI_mask shape:", ROI_mask.shape)
X_ROI = X_trials[ROI_mask,:,:]
print("X_ROI shape:", X_ROI.shape)




# =========================
# PLOTS 
# =========================

W_img=ev.window_weights_to_pixel_time_matrix(results_sliding_window["w_mean_windows"], ROI_mask, pixels=100)
print(W_img.shape) #(pixels, frames) where frames correspond to window centers
pl.mimg(W_img, xsize=100, ysize=100, low=-0.0002, high=0.0002, frames=results_sliding_window["centers"])


#mimg of positive and negative weight masks across windows
pos_masks, neg_masks = ev.extract_extreme_weight_masks_sliding(W_img, ROI_mask,  frac=0.20)
pl.mimg(pos_masks, xsize=100, ysize=100, low='auto', high=None, frames=results_sliding_window["centers"])
pl.mimg(neg_masks, xsize=100, ysize=100, low=0, high=1, frames=results_sliding_window["centers"])


sw.plot_sliding_window_accuracy_with_sem(res=results_sliding_window,
                                        chance=0.5,
                                        title="Sliding Window Decoding (5-frame window)")


# weights movie
mf.show_weight_movie(W_pixel_time=W_img,
                    centers=results_sliding_window["centers"],
                    frame_acc=results_sliding_window["frame_acc_mean"],
                    trial_acc=results_sliding_window["trial_acc_mean"],
                    pixels=100,
                    fps=10,
                    clip_q=90,
                    title="Mean weight map (across folds) per window")
'''



a=1
