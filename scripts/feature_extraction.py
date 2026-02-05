import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector



def load_day_data(date_str, data_dir, ext=".npy"):
    import os, glob, numpy as np
    pattern = os.path.join(data_dir, f"*{date_str}*{ext}")
    files_used = sorted(glob.glob(pattern))
    #exclude condition 3 - the blank
    files_used = [f for f in files_used if "Z3" not in os.path.basename(f)]
    if not files_used:
        raise FileNotFoundError("No valid files found for this day (after excluding cond 3).")
    arrays = []
    for fp in files_used:
        arr = np.load(fp)
        arrays.append(arr)
    #same number of frames
    n_frames = arrays[0].shape[1]
    for arr in arrays:
        if arr.shape[1] != n_frames:
            raise ValueError("Frame mismatch between files.")
    day_data = np.concatenate(arrays, axis=2)
    return day_data, files_used



def mimg(x, xsize=100, ysize=100, low='auto', high=None, frames=None, width=0):
    # if looking in raw data (not Zscored), data needs to substract 1  (-1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    m, n = x.shape
    if width <= 0:
        width = int(np.ceil(np.sqrt(n)))
    height = int(np.ceil(n / width))
    # Handle Clipping
    if isinstance(low, str) and low.lower() == 'auto':
        v_mins, v_maxs = [None] * n, [None] * n
    elif isinstance(low, str) and low.lower() == 'all':
        v_mins, v_maxs = [np.min(x)] * n, [np.max(x)] * n
    else:
        v_mins = np.full(n, low) if np.isscalar(low) else low
        v_maxs = np.full(n, high) if np.isscalar(high) else high
    fig, axes = plt.subplots(height, width, figsize=(width*3, height*3), squeeze=False)
    axes_flat = axes.flatten()
    for i in range(n):
        ax = axes_flat[i]
        # 1. Reshape using 'C' order because your data is "ordered by rows"
        # 2. We reshape as (xsize, ysize) or (ysize, xsize) depending on the source
        img_data = x[:, i].reshape((ysize, xsize), order='C')
        # Use 'jet' or 'nipy_spectral' 
        im = ax.imshow(img_data, cmap='jet', vmin=v_mins[i], vmax=v_maxs[i], origin='upper')
        ax.axis('off')
        # Add frame numbers if provided
        if frames is not None:
            ax.set_title(f"frame {frames[i]}", fontsize=6)
    # Hide extra subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')
    plt.tight_layout()
    return fig,axes_flat
'''
x = np.load(r"C:\project\vsdi-face-decoding\data\processed\condsXn\condsXn1_270109b.npy")
x_avg = x.mean(axis=2)
x_avg_frames =  x_avg[:, 25:120]
fig,axes_flat =mimg(x_avg_frames-1, xsize=100, ysize=100, low=-0.0009, high=0.003,frames=range(25,120))
plt.show()
x = np.load(r"C:\project\vsdi-face-decoding\data\processed\condsXn\condsXn5_270109b.npy")
x_avg = x.mean(axis=2)
x_avg_frames =  x_avg[:, 25:120]
fig,axes_flat =mimg(x_avg_frames-1, xsize=100, ysize=100, low=-0.0009, high=0.003,frames=range(25,120))
plt.show()
'''


def extract_window(data, start=35, end=45):
    if start < 0 or end > data.shape[1]:
        raise ValueError(f"Window [{start}:{end}] out of bounds for n_frames={data.shape[1]}")
    return data[:, start:end, :]



def creat_ROI(map_flat, pixels=100):
    """
    map_flat : (pixels*pixels,) vector
    returns:
        mask_flat : (pixels*pixels,) bool
        roi_idx   : indices where mask == True
    """
    fig, axes_flat = mimg(map_flat.reshape(-1, 1) - 1,
                        xsize=pixels, ysize=pixels,
                        low=-0.0009, high=0.003)
    ax = axes_flat[0]
    ax.set_title("ROI: left click = add point, right click = finish")
    # show + make sure GUI is ready BEFORE ginput
    plt.show(block=False)
    plt.pause(0.05)
    plt.sca(ax)  # ensure clicks go to this axes
    pts = plt.ginput(n=-1, timeout=0)  # right click to finish
    plt.close(fig)
    if len(pts) < 3:
        raise ValueError("Polygon requires at least 3 points")
    poly = Path(pts)

    # pixel-center grid
    X, Y = np.meshgrid(np.arange(pixels) + 0.5,
                    np.arange(pixels) + 0.5)
    points = np.column_stack([X.ravel(), Y.ravel()])
    mask_flat = poly.contains_points(points)
    roi_idx = np.where(mask_flat)[0]
    #display mask
    fig2, ax2 = plt.subplots()
    ax2.imshow(mask_flat.reshape(pixels, pixels), cmap='gray',
            interpolation='nearest', origin='upper', aspect='equal')
    ax2.set_title("ROI mask")
    plt.show()
    return mask_flat, roi_idx


#example usage
'''
x = np.load(r"C:\project\vsdi-face-decoding\data\processed\condsXn\condsXn1_030209e.npy")
x_avg = x.mean(axis=2)
print(x_avg.shape)
x_avg_frames =  x_avg[:, 55:65]
print(x_avg_frames.shape)
mimg(x_avg_frames-1, xsize=100, ysize=100, low=-0.0009, high=0.003)

roi,x=choose_polygon(x_avg[:, 50], pixels=100)
'''


def avgWindow(data_3d):
    """
    Reshapes 3D brain data (Pixels, Frames, Trials) into 2D (Trials, avg across frames).
    """
    # 1. Ensure 'Trials' is the first dimension
    # If input is (Pixels, Frames, Trials), move Trials to the front
    if data_3d.shape[-1] < data_3d.shape[0]: 
        data_3d = np.moveaxis(data_3d, -1, 0) 
    n_trials, n_pixels, n_frames = data_3d.shape
    # 2. Flatten Space and Time into one long vector per trial
    # Result shape: (60, 5000)
    X_avgWindow = np.nanmean(data_3d, axis=2) # average over frames
    return X_avgWindow