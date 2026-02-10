import numpy as np
import matplotlib.pyplot as plt
from preprocessing_functions import (split_conds_files,
                        frame_zero_normalize_all_conds, 
                        normalize_to_clean_blank,
                        )
from feature_extraction import creat_ROI, mimg

RAW_DIR = r"C:\project\vsdi-face-decoding\data\raw_mat"
SPLIT_DIR = r"C:\project\vsdi-face-decoding\data\processed\all_conds_split"
FZ_DIR = r"C:\project\vsdi-face-decoding\data\processed\condsX"
BLANK_COND = 3
BLANK_DIR = r"C:\project\vsdi-face-decoding\data\processed\condsXn"
OVERWRITE = True
EPS = 1e-8
DTYPE = np.float32
ZERO_FRAMES = (5, 25)  # frames to use for frame-zero normalization


split_conds_files(raw_dir=RAW_DIR,
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



# Example usage of mimg function
x = np.load(r"C:\project\vsdi-face-decoding\data\processed\condsXn\condsXn1_110209a.npy")
x_avg = x.mean(axis=2)
x_avg_frames =  x_avg[:, 35:45]
fig,axes_flat =mimg(x_avg_frames-1, xsize=100, ysize=100, low=-0.0009, high=0.003,frames=range(25,120))
plt.show()
'''
# define ROI once and save mask
x_avg_frames=np.mean(x_avg_frames,axis=1)
ROI_mask,roi_idx=creat_ROI(x_avg_frames, pixels=100)
# save ROI mask for future use
np.save(r"C:\project\vsdi-face-decoding\data\processed\ROI_mask2.npy", ROI_mask)
'''