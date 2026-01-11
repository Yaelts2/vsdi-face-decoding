from pathlib import Path
import numpy as np
from scipy.io import loadmat

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
    """
    Uses condsX{blank_cond} as the blank.
    Computes blankMean = mean(cond3, axis=2).
    Divides all other conditions by blankMean (per trial).
    """
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


def zscore_all_files(in_dir="data/processed/condsXn",
                    out_dir="data/processed/condsXnZ",
                    zero_frames=(24, 25, 26),   # MATLAB 25:27 -> Python
                    overwrite=False,
                    dtype=np.float32,
                    eps=1e-6
                    ):
    """
    Z-score each trial using baseline frames (zero_frames).
    Input:  condsXn{cond}_{session}.npy  (pixels, frames, trials)
    Output: condsXnZ{cond}_{session}.npy (same shape)
    """
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

    # sanity check: same number of frames
    n_frames = arrays[0].shape[1]
    for arr in arrays:
        if arr.shape[1] != n_frames:
            raise ValueError("Frame mismatch between files.")

    day_data = np.concatenate(arrays, axis=2)
    return day_data, files_used


day_data, files = load_day_data("270109", r"data/processed/condsXnZ")
print(day_data.shape)   # (10000, n_frames, total_trials)
print(len(files))
