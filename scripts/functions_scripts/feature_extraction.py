import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from functions_scripts import ml_plots as pl


def frames_as_samples(data_3d,y_trials, trial_axis=-1, frame_axis=1, pixel_axis=0):
    """
    Convert (Pixels, Frames, Trials) into frame-level samples:
    X:      (Trials*Frames, Pixels)
    groups: (Trials*Frames,)   trial id repeated for each frame

    Default assumes input is (Pixels, Frames, Trials).
    """
    data_3d = np.asarray(data_3d)

    # Reorder to (Trials, Frames, Pixels)
    x = np.moveaxis(data_3d, (trial_axis, frame_axis, pixel_axis), (0, 1, 2))
    n_trials, n_frames, n_pixels = x.shape

    # X: each frame is a sample, features are pixels
    X = x.reshape(n_trials * n_frames, n_pixels)

    # groups: keep frames from the same trial together
    groups = np.repeat(np.arange(n_trials), n_frames)
    y_frames = np.repeat(y_trials, n_frames) #n_frames in the window
    return X,y_frames, groups


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
    fig, axes_flat = pl.mimg(map_flat.reshape(-1, 1) - 1,
                        xsize=pixels, ysize=pixels,
                        low=-0.0005, high=0.002)
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
pl.mimg(x_avg_frames-1, xsize=100, ysize=100, low=-0.0009, high=0.003)

roi,x= fe.creat_ROI(x_avg[:, 50], pixels=100)
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