from pathlib import Path
import numpy as np
from scipy.io import loadmat
import json
import matplotlib.colors as mcolors


def split_conds_files(raw_dir="data/raw_mat",
                    out_dir="data/processed/all_conds_npy",
                    overwrite=False,
                    dtype=np.float32):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    if out_dir.exists():
        print("Split cond files already exist in data/processed/all_conds_npy")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("*.mat"))
    if len(files) == 0:
        print(f"No .mat files found in: {raw_dir.resolve()}")
        return

    for file in files:
        print("Processing:", file.name)
        data = loadmat(file)

        session = file.stem.replace("conds_", "")

        cond_keys = sorted(
            [k for k in data.keys() if k.startswith("cond") and k[4:].isdigit()],
            key=lambda x: int(x[4:])
        )

        for ck in cond_keys:
            cond_num = int(ck[4:])
            mat = np.asarray(data[ck]).astype(dtype, copy=False)

            out_file = out_dir / f"conds{cond_num}_{session}.npy"
            if out_file.exists() and not overwrite:
                continue

            np.save(out_file, mat)

    print(f"Done. Saved .npy files to: {out_dir.resolve()}")



def frame_zero_normalize_all_conds(in_dir="data/processed/all_conds_npy",
                                out_dir="data/processed/condsX",
                                zero_frames=(24, 25, 26),   # MATLAB 25:27 -> Python
                                overwrite=False,
                                dtype=np.float32,
                                eps=1e-12):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    if out_dir.exists() and not overwrite:
        print("condsX files already exist in data/processed/condsX")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.npy"))
    if len(files) == 0:
        print(f"No .npy files found in: {in_dir.resolve()}")
        return
    zf = np.array(zero_frames, dtype=int)
    for file in files:
        mat = np.load(file).astype(dtype, copy=False)   # (pixels, frames, trials)
        if mat.ndim != 3:
            raise ValueError(f"{file.name}: expected 3D array, got {mat.shape}")
        # baseline: (pixels, trials)
        baseline = mat[:, zf, :].mean(axis=1)
        # safe baseline (avoid /0) - simpler than np.where
        baseline = np.maximum(np.abs(baseline), eps) * np.sign(baseline + eps)
        # normalize (broadcast)
        norm_mat = (mat / baseline[:, None, :]).astype(dtype, copy=False)
        # filename: conds1_030209a.npy -> condsX1_030209a.npy
        out_name = file.name.replace("conds", "condsX", 1)
        np.save(out_dir / out_name, norm_mat)
    print(f"Done. Saved frame-zero normalized .npy files to: {out_dir.resolve()}")


def normalize_to_clean_blank(blank_cond=3,
                            in_dir="data/processed/condsX",
                            out_dir="data/processed/condsXn",
                            overwrite=False,
                            dtype=np.float32,
                            eps=1e-12
                            ):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    if out_dir.exists() and not overwrite:
        print("condsXn files already exist in data/processed/condsXn")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("condsX*_*.npy"))
    if len(files) == 0:
        print(f"No condsX files found in: {in_dir.resolve()}")
        return
    sessions = {}
    for f in files:
        stem = f.stem.replace("condsX", "", 1)   # "1_030209a"
        cond_str, session = stem.split("_", 1)
        cond_num = int(cond_str)
        sessions.setdefault(session, {})[cond_num] = f
    #process each session separately
    for session, cond_files in sessions.items():
        if blank_cond not in cond_files:
            raise ValueError(f"Session {session} has no blank condition (cond{blank_cond})")
        # load blank and compute mean across trials
        blank_mat = np.load(cond_files[blank_cond]).astype(dtype, copy=False)
        blank_mean = np.nanmean(blank_mat, axis=2)   # (pixels, frames)
        blank_mean = np.where(np.abs(blank_mean) < eps, eps, blank_mean) # avoid /0
        for cond_num, file in cond_files.items():
            mat = np.load(file).astype(dtype, copy=False)
            out_name = file.name.replace("condsX", "condsXn", 1)
            out_path = out_dir / out_name
            if cond_num == blank_cond:
                # blank condition saved unchanged
                np.save(out_path, mat)
            else:
                # divide each trial by blankMean
                new_mat = (mat / blank_mean[:, :, None]).astype(dtype, copy=False)
                np.save(out_path, new_mat)
    print(f"Done. Saved blank-normalized files to: {out_dir.resolve()}")

'''
def zscore_all_files(in_dir="data/processed/condsXn",
                    out_dir="data/processed/condsXnZ",
                    zero_frames=(24, 25, 26),   # MATLAB 25:27 -> Python
                    overwrite=False,
                    dtype=np.float32,
                    eps=1e-6
                    ):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    if out_dir.exists() and not overwrite:
        print("Z-scored files already exist in data/processed/condsXnZ")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("condsXn*_*.npy"))
    if len(files) == 0:
        print(f"No input files found in: {in_dir.resolve()}")
        return
    zf = np.array(zero_frames, dtype=int)
    for file in files:
        mat = np.load(file).astype(dtype, copy=False)  # (pixels, frames, trials)
        # baseline stats per pixel per trial: (pixels, trials)
        base = mat[:, zf, :]  # (pixels, len(zf), trials)
        mu = np.nanmean(base, axis=1)
        sigma = np.nanstd(base, axis=1)
        sigma = np.where(sigma < eps, eps, sigma) # avoid /0
        # z-score (broadcast across frames)
        zmat = (mat - mu[:, None, :]) / sigma[:, None, :]
        zmat = zmat.astype(dtype, copy=False)
        out_name = file.name.replace("condsXn", "condsXnZ", 1)
        np.save(out_dir / out_name, zmat)
    print(f"Done. Saved Z-scored files to: {out_dir.resolve()}")
'''


    
def build_X_y(face_file, nonface_file, data_dir):
    """
    Load one face condition and one non-face condition
    and return X (all trials) and y (labels).

    X shape: (pixels, frames, trials)
    y shape: (trials,)
    """
    data_dir = Path(data_dir) 
    # load data
    X_face = np.load(data_dir / face_file)
    X_non  = np.load(data_dir / nonface_file)
    print("Loaded shapes:")
    print(" face    :", X_face.shape)
    print(" nonface :", X_non.shape)
    # make trial counts equal
    n = min(X_face.shape[2], X_non.shape[2])
    X_face = X_face[:, :, :n]
    X_non  = X_non[:, :, :n]
    # stack trials
    X = np.concatenate([X_face, X_non], axis=2)
    # labels based on order
    FACE_LABEL = 1
    NONFACE_LABEL = 0
    y = np.array([FACE_LABEL]*n + [NONFACE_LABEL]*n)
    print("Final:")
    print(" X shape:", X.shape)
    print(" y:", y)
    return X, y, {"n_trials_per_class": n,
                    "face_file": face_file,
                    "nonface_file": nonface_file}

# example usage
'''
X, y, metadata = build_X_y(
    face_file="condsXn1_270109b.npy",
    nonface_file="condsXn5_270109b.npy",
    data_dir="data/processed/condsXn/"
)
print("Metadata:", metadata)
print("X shape:", X.shape)
print("y shape:", y.shape)

'''



def zscore_dataset_pixelwise_trials(X, baseline_frames=(0, 24), eps: float = 1e-8, ddof: int = 0):
    """
    Updated Pixel-wise Z-score:
    1. Subtract trial-specific baseline mean from each trial.
    2. Calculate STD of those 'mean-centered' baseline segments pooled across trials.
    3. Divide by that pooled STD.
    """
    X = np.asarray(X, dtype=float)
    pixels, frames, trials = X.shape

    start, end = baseline_frames
    
    # 1. Get the Baseline window
    # Shape: (pixels, baseline_frames, trials)
    baseline = X[:, start:end, :] 
    
    # 2. Calculate Mean per pixel PER TRIAL
    # Shape: (pixels, 1, trials)
    mean_per_trial = baseline.mean(axis=1, keepdims=True) 
    
    # 3. Subtract the mean from the whole data (Broadcasting)
    # X_centered shape: (pixels, frames, trials)
    X_centered = X - mean_per_trial
    
    # 4. Calculate STD across both 'frames' and 'trials' of the CENTERED baseline
    # We use the centered baseline so the STD reflects variance after mean removal
    baseline_centered = X_centered[:, start:end, :]
    std_pooled = baseline_centered.std(axis=(1, 2), keepdims=True, ddof=ddof)
    
    # 5. Final Division (Z-score)
    # Shape: (pixels, frames, trials)
    X_z = X_centered / np.maximum(std_pooled, eps)

    return X_z, mean_per_trial, std_pooled




#=====colormap=========

def green_gray_magenta():
    """
    Custom diverging colormap:
    Negative -> green
    Zero     -> white
    Positive -> magenta
    With gray transitions to reduce visual bias.
    """
    colors = [
        (0.0, "mediumblue"),  
        (0.1, "darkgray"),
        (0.3, "lightgray"),
        (0.5, "white"),
        (0.7, "lightgray"),
        (0.9, "darkgray"),
        (1.0, "magenta"),
    ]

    return mcolors.LinearSegmentedColormap.from_list(
        "green_gray_magenta",
        colors
    )
