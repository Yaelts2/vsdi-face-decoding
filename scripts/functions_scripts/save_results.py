from __future__ import annotations
from functions_scripts import preprocessing_functions as pre
from pathlib import Path
from datetime import datetime
import json
import numpy as np



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
    run_dir = results_root / f"{experiment}__{experiment_tag}__{ts}"    
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



def load_data_from_config(config):
    """
    Load and prepare dataset exactly like training:
    1) build X_trials, y_trials
    2) z-score pixelwise across trials (using baseline frames from config)
    Returns:
        X_trials_z, y_trials
    """
    # --- paths / files ---
    data_dir = config["data_dir"]
    face_file = config["face_file"]
    nonface_file = config["nonface_file"]

    # --- baseline frames for z-score ---
    # Accept either "Baseline_frames_zscore" or "baseline_frames_zscore"
    baseline = config.get("zscore_baseline_frames")

    # 1) build dataset
    X_trials, y_trials, dataset_info = pre.build_X_y(
        face_file=face_file,
        nonface_file=nonface_file,
        data_dir=data_dir
    )

    # 2) z-score across all trials (pixelwise)
    X_z, mean, std = pre.zscore_dataset_pixelwise_trials(X_trials, baseline)

    return X_z, y_trials

