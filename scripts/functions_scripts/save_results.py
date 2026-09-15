from pathlib import Path
import sys
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
print("Project root:", project_root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.functions_scripts import preprocessing_functions as pre
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from scripts.functions_scripts import ml_plots as pl


def _sem(x, axis=0):
    x = np.asarray(x, dtype=float)
    n = np.sum(~np.isnan(x), axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=1)
    return sd / np.sqrt(n)

def save_experiment(results_root: str | Path,experiment: str, experiment_tag: str,
                        results: dict,
                        ROI_mask_path: str | Path,
                        dataset_info: dict | None = None) -> Path:
    """
    Save everything you typically need to re-plot and re-analyze a fixed-window run
    WITHOUT retraining.

    Parameters
    results_root : str | Path
        Root folder, e.g. "results"
    experiment : str
        Short name, e.g. "fixed_window__[34:44]"
    experiment_tag : str
        how the model was trained (e.g. "SVM_10foldCV" or "SVM_looCV")
    results : dict
        Output of cv.run_nested_cv_selectC_then_eval(...)
    ROI_mask : np.ndarray
        Boolean mask in full image space (10000,) or (100,100) 
    dataset_info : dict | None
        Any metadata you want to remember (filenames, subject, area, etc.)
    
    Returns
    run_dir : Path ---        Folder where results were saved.
    """
    #results folder root
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # timestamp for the model run
    run_dir = results_root / f"{experiment}__{experiment_tag}____{ts}"    
    run_dir.mkdir(parents=True, exist_ok=False)
    
    #config.json 
    config = {"experiment": experiment, "tag": experiment_tag}
    if dataset_info:
        config.update(dataset_info)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Save the full nested dict as an object (simple + flexible).
    np.savez_compressed(run_dir / "result.npz",
                        results=np.array(results, dtype=object))

    # Save ROI path (not the full mask array)
    roi_path_dict = {"roi_mask_path": str(ROI_mask_path)}
    np.savez_compressed(run_dir / "ROI_mask_path.npz", **roi_path_dict)

    return run_dir


def load_experiment(run_dir):
    run_dir = Path(run_dir)

    with open(run_dir / "config.json", "r") as f:
        config = json.load(f)

    d = np.load(run_dir / "result.npz", allow_pickle=True)
    results = d["results"].item()

    d_roi = np.load(run_dir / "ROI_mask_path.npz", allow_pickle=True)
    ROI_mask_path = str(d_roi["roi_mask_path"])
    
    return config, results, ROI_mask_path


def load_permutation_run(run_dir: str):
    run_dir = Path(run_dir)

    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    d = np.load(run_dir / "perm_result.npz", allow_pickle=False)
    keys = set(d.files)

    # new schema (trial + frame)
    if "shuffled_scores_trials" in keys:
        return {
            "config": config,
            "shuffled_scores_trials": d["shuffled_scores_trials"].ravel(),
            "real_trial_acc": float(d["real_trial_acc"]),
            "real_trial_std": float(d["real_trial_std"]),
            "shuffled_trial_mean": float(d["shuffled_trial_mean"]),
            "shuffled_trial_std": float(d["shuffled_trial_std"]),
            "shuffled_trial_min": float(d["shuffled_trial_min"]),
            "shuffled_trial_max": float(d["shuffled_trial_max"]),
            "p_value_trial_two_tailed": float(d["p_value_trial_two_tailed"]),
            "pass_alpha_0p05_trial": bool(d["pass_alpha_0p05_trial"]),
            # optional extras if present
            "shuffled_scores_frames": d["shuffled_scores_frames"].ravel() if "shuffled_scores_frames" in keys else None,
            "real_frame_acc": float(d["real_frame_acc"]) if "real_frame_acc" in keys else None,
            "real_frame_std": float(d["real_frame_std"]) if "real_frame_std" in keys else None,
        }

    # old schema (legacy)
    return {
        "config": config,
        "shuffled_scores_trials": d["shuffled_scores"].ravel(),
        "real_trial_acc": float(d["real_score"]),
        "real_trial_std": float(d["real_std"]),
        "shuffled_trial_mean": float(d["shuffled_mean"]),
        "shuffled_trial_std": float(d["shuffled_std"]),
        "shuffled_trial_min": float(d["shuffled_min"]),
        "shuffled_trial_max": float(d["shuffled_max"]),
        "p_value_trial_two_tailed": float(d["p_value"]),
        "pass_alpha_0p05_trial": bool(d["pass_alpha_0p05"]),
        # legacy has no frame-level fields
        "shuffled_scores_frames": None,
        "real_frame_acc": None,
        "real_frame_std": None,
    }


def load_sliding_window_permutation(run_dir):
    """
    Load saved sliding-window permutation results.

    Expects:
        run_dir/
            config.json
            perm_result.npz

    Returns:
        dict with:
            centers
            real_trial_curve
            real_frame_curve
            null_trial_acc
            null_frame_acc
            null_trial_mean
            null_trial_std
            null_frame_mean
            null_frame_std
            config
    """

    run_dir = Path(run_dir)

    #Load config
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    #Load npz results
    npz_path = run_dir / "perm_result.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing perm_result.npz in {run_dir}")
    data = np.load(npz_path, allow_pickle=True)
    out = {"centers": data["centers"],
        # real curves
        "real_trial_curve": data.get("real_trial_curve"),
        "real_fold_trial_acc" : data.get("real_fold_trial_acc"),
        "real_frame_curve": data.get("real_frame_curve"),
        # full null curves
        "null_trial_acc": data.get("null_trial_acc"),
        "null_trial_folds": data.get("null_trial_folds"),
        "null_frame_acc": data.get("null_frame_acc"),
        # null mean ± std
        "null_trial_mean": data.get("null_trial_mean"),
        "null_trial_std": data.get("null_trial_std"),
        "null_frame_mean": data.get("null_frame_mean"),
        "null_frame_std": data.get("null_frame_std"),
        "config": config }

    return out




def load_data_from_config(config):
    """
    Load and prepare dataset exactly like training:
    1) build X_trials, y_trials
    2) z-score pixelwise across trials

    Handles TWO cases, distinguished by config["baseline_source"]:

    A) Regular / stim-aligned configs (baseline_source absent, i.e. every
       config saved before the RT-aligned baseline fix, and any future
       stim-aligned run): baseline frames are taken directly from
       config["zscore_baseline_frames"] and applied to X_trials itself,
       via pre.zscore_dataset_pixelwise_trials -- exactly as before.

    B) RT-aligned configs (baseline_source == "stim_aligned_per_trial",
       saved by the corrected run_sliding_window_RT.py): X_trials has no
       true pre-stimulus period of its own (it's cut around the reaction
       time), so the per-trial baseline mean instead comes from that same
       trial's stimulus-aligned data (config["baseline_face_file_stim"] /
       "baseline_nonface_file_stim" / "baseline_data_dir_stim" /
       "zscore_baseline_frames_stim"). This mirrors
       zscore_RT_with_stim_baseline in run_sliding_window_RT.py exactly,
       so a model's saved config always reproduces the same data it was
       actually trained on.

    Returns:
        X_trials_z, y_trials
    """
    # --- paths / files (the data actually being loaded/decoded) ---
    data_dir = config["data_dir"]
    face_file = config["face_file"]
    nonface_file = config["nonface_file"]

    # 1) build dataset
    X_trials, y_trials, dataset_info = pre.build_X_y(
        face_file=face_file,
        nonface_file=nonface_file,
        data_dir=data_dir
    )

    baseline_source = config.get("baseline_source")

    if baseline_source == "stim_aligned_per_trial":
        # --- RT-aligned case: baseline comes from the matching stim-aligned trial ---
        face_file_stim = config["baseline_face_file_stim"]
        nonface_file_stim = config["baseline_nonface_file_stim"]
        data_dir_stim = config["baseline_data_dir_stim"]
        baseline_frames_stim = tuple(config["zscore_baseline_frames_stim"])

        X_stim_trials, y_stim_trials, _ = pre.build_X_y(
            face_file=face_file_stim,
            nonface_file=nonface_file_stim,
            data_dir=data_dir_stim
        )

        if X_stim_trials.shape[-1] != X_trials.shape[-1]:
            raise ValueError(
                f"Trial count mismatch reloading RT-aligned data: RT-aligned "
                f"({face_file}/{nonface_file}) has {X_trials.shape[-1]} trials, "
                f"stim-aligned baseline source ({face_file_stim}/{nonface_file_stim}) "
                f"has {X_stim_trials.shape[-1]} trials. Cannot apply per-trial "
                f"baseline with mismatched trial counts."
            )
        if not np.array_equal(y_stim_trials, y_trials):
            print("WARNING: label vectors from RT-aligned and stim-aligned baseline "
                  "data differ even though trial counts match -- positional trial "
                  "matching may be WRONG here. Double-check before trusting this reload.")

        b0, b1 = baseline_frames_stim
        baseline = X_stim_trials[:, b0:b1, :]          # (pixels, baseline_frames, trials)
        mean = np.nanmean(baseline, axis=1)            # (pixels, trials)
        std = np.nanstd(baseline, axis=1)               # (pixels, trials)
        n_zero_std = np.sum(std == 0)
        if n_zero_std > 0:
            print(f"WARNING: {n_zero_std} (pixel, trial) baseline std values are exactly 0 "
                  f"-- these will become NaN after division.")
        std_safe = np.where(std == 0, np.nan, std)

        X_z = (X_trials - mean[:, None, :]) / std_safe[:, None, :]

    else:
        # --- regular / stim-aligned case: unchanged from before ---
        baseline = config.get("zscore_baseline_frames")
        X_z, mean, std = pre.zscore_dataset_pixelwise_trials(X_trials, baseline)
        '''
        x_avg_frames =  X_z[:, 25:75,0:27]
        x_avg_frames = np.nanmean(x_avg_frames,axis=2)
        frame_ids = list(range(25, 75))     # 25..80 (56 frames)
        binned, fig, axes, cid =pl.plot_superpixel_traces(x_avg_frames, xs=100, ys=100, nsubplots=5,overlay=False, frames=frame_ids )
        plt.show()
        '''

    return X_z, y_trials