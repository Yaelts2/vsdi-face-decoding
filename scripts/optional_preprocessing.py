import numpy as np
import matplotlib.pyplot as plt
from functions_scripts import preprocessing_functions as pre
from functions_scripts import ml_plots as pl
from functions_scripts import feature_extraction as fe

RAW_DIR = r"C:\project\vsdi-face-decoding\data\raw_mat"
SPLIT_DIR = r"C:\project\vsdi-face-decoding\data\processed\all_conds_split"
FZ_DIR = r"C:\project\vsdi-face-decoding\data\processed\condsX"
BLANK_COND = 3
BLANK_DIR = r"C:\project\vsdi-face-decoding\data\processed\condsXn"
OVERWRITE = True
EPS = 1e-8
DTYPE = np.float32
ZERO_FRAMES = (5, 25)  # frames to use for frame-zero normalization

'''
pre.split_conds_files(raw_dir=RAW_DIR,
                out_dir=SPLIT_DIR,
                overwrite=OVERWRITE,
                dtype=DTYPE)

# 2) Frame-zero normalization (trial-wise)
pre.frame_zero_normalize_all_conds(in_dir=SPLIT_DIR,
                            out_dir=FZ_DIR,
                            zero_frames=ZERO_FRAMES,
                            overwrite=OVERWRITE,
                            dtype=DTYPE,
                            eps=EPS)

# 3) Normalize to blank (session-wise; blank=cond3)
pre.normalize_to_clean_blank(blank_cond=BLANK_COND,
                        in_dir=FZ_DIR,
                        out_dir=BLANK_DIR,
                        overwrite=OVERWRITE,
                        dtype=DTYPE,
                        eps=EPS)

'''

# Example usage of mimg function
x = np.load(r"C:\project\vsdi-face-decoding\data\processed\condsXn\condsXn1_110209a.npy")
x_avg = x.mean(axis=2)
x_avg_frames =  x_avg[:, 32:39]
fig,axes_flat =pl.mimg(x_avg_frames-1, xsize=100, ysize=100, low=-0.0005, high=0.002,frames=range(25,120))
plt.show()

# define ROI once and save mask
x_avg_frames=np.mean(x_avg_frames,axis=1)
ROI_mask,roi_idx=fe.creat_ROI(x_avg_frames, pixels=100)
# save ROI mask for future use
np.save(r"C:\project\vsdi-face-decoding\data\processed\ROI_onlyV24_mask.npy", ROI_mask)
