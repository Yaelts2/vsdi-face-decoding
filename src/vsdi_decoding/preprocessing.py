from pathlib import Path
import numpy as np
from scipy.io import loadmat
import json

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
# should be in configuration file
#when using in main need to import the dir paths from config
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Data" / "condsXnZ"
OUT_DIR = SCRIPT_DIR / "Data" / "labeled_face_vs_nonface"

print(">>> labeling.py started")
print("Script location:", SCRIPT_DIR)
print("Input folder:", DATA_DIR)
print("Input folder exists?", DATA_DIR.exists())

def build_face_vs_nonface_dataset(face_conds=(1, 2),
                                nonface_conds=(4, 5),   # blank (3) ignored
                                dtype=np.float32
                                ):
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {DATA_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("condsXnZ*_*.npy"))
    print("Found input files:", len(files))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched condsXnZ*_*.npy in {DATA_DIR}")

    # -------------------------
    # PASS 1: figure out total samples + feature size
    # -------------------------
    total_trials = 0
    n_features = None

    keep_files = []  # (file_path, label, session)
    for i, f in enumerate(files, 1):
        stem = f.stem.replace("condsXnZ", "")
        cond_str, session = stem.split("_", 1)
        cond = int(cond_str)

        if cond in face_conds:
            label = 1
        elif cond in nonface_conds:
            label = 0
        else:
            continue

        mat = np.load(f, mmap_mode="r")  # cheap header read + OS paging
        if mat.ndim != 3:
            raise ValueError(f"{f.name}: expected 3D array, got {mat.shape}")

        pixels, frames, trials = mat.shape
        feat = pixels * frames
        if n_features is None:
            n_features = feat
        elif n_features != feat:
            raise ValueError(f"Feature size mismatch in {f.name}: {feat} vs {n_features}")

        total_trials += trials
        keep_files.append((f, label, session))

        if i % 5 == 0:
            print(f"[Pass1] scanned {i}/{len(files)} files... kept so far: {len(keep_files)}")

    if total_trials == 0:
        raise ValueError("No trials collected. Check condition numbers / files.")

    print("Total samples (trials):", total_trials)
    print("Features per sample:", n_features)

    # -------------------------
    # PASS 2: write directly to disk (memmap) to avoid giant RAM concat
    # -------------------------
    X_path = OUT_DIR / "X.npy"
    y_path = OUT_DIR / "y.npy"
    s_path = OUT_DIR / "sessions.npy"

    # Pre-allocate on disk
    X_mm = np.lib.format.open_memmap(
        X_path, mode="w+", dtype=dtype, shape=(total_trials, n_features)
    )
    y = np.empty((total_trials,), dtype=np.int32)
    sessions = np.empty((total_trials,), dtype=object)

    idx = 0
    for j, (f, label, session) in enumerate(keep_files, 1):
        mat = np.load(f).astype(dtype, copy=False)  # load one file at a time
        pixels, frames, trials = mat.shape

        # reshape to (trials, pixels*frames)
        X_block = mat.reshape(pixels * frames, trials).T

        X_mm[idx:idx + trials, :] = X_block
        y[idx:idx + trials] = label
        sessions[idx:idx + trials] = session
        idx += trials

        print(f"[Pass2] wrote {j}/{len(keep_files)} files | samples written: {idx}/{total_trials}")

    # flush memmap to disk
    del X_mm

    np.save(y_path, y)
    np.save(s_path, sessions)

    with open(OUT_DIR / "label_map.json", "w") as f:
        json.dump(
            {"0": "non-face (scrambled)", "1": "face (real monkey faces)"},
            f,
            indent=2
        )

    print("✅ Dataset built successfully")
    print("Saved to:", OUT_DIR)
    print("X:", (total_trials, n_features))
    print("y distribution:", np.unique(y, return_counts=True))


if __name__ == "__main__":
    build_face_vs_nonface_dataset()
'''
    
def building_dataset()