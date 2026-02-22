from pathlib import Path
import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
print("Project root:", project_root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
import json
import numpy as np
from sklearn.model_selection import GroupKFold
from scripts.functions_scripts import preprocessing_functions as pre
from scripts.functions_scripts import feature_extraction as fe
from scripts.functions_scripts import model_control as mc
from scripts.functions_scripts import save_results as sr


# ------------------------------------------------------------
# USER: set which saved model/run to test
# ------------------------------------------------------------

RESULTS_ROOT = Path(r"C:\project\vsdi-face-decoding\results")

# Point to the specific run folder you want to test (the folder that contains config.json/result.npz/ROI_mask.npz)
# Example:
RUN_DIR = RESULTS_ROOT / "fixed_window__frame32-40__SVM_10fold__2026-02-18_17-43-53"

N_PERMUTATIONS = 2
PERM_SEED = 42

# Save destination root:
PERM_ROOT = RESULTS_ROOT / "permutation_test"



# 1) Load saved model run (config + real nested results + ROI)
config, real_nested, ROI_mask_path = sr.load_experiment(RUN_DIR)

print("Loaded run:")
print("  RUN_DIR:", RUN_DIR)
print("  experiment:", config.get("experiment"))
print("  tag:", config.get("tag"))
print("  window:", config.get("window"))
print("  metric:", config.get("metric"))


# 2) Rebuild the SAME dataset features (X_frames, y_frames, groups) using saved config
#    (Permutation needs X + labels; we do NOT save X in results to keep files small.)
face_file = config["face_file"]
nonface_file = config["nonface_file"]
data_dir = config["data_dir"]

baseline = tuple(config["zscore_baseline_frames"])
window = tuple(config["window"])

metric = config.get("metric", "acc")
rule = config.get("rule", "one_se")
tie_break = config.get("tie_break", "smaller_C")
C_grid = np.asarray(config.get("C_grid", np.logspace(-4, 2, 7)), dtype=float)

# CV splitters (match your training defaults)
outer = GroupKFold(n_splits=10)
inner = GroupKFold(n_splits=4)

# Build trials dataset
X_trials, y_trials, _dataset_info2 = pre.build_X_y(face_file=face_file,
                                                nonface_file=nonface_file,
                                                data_dir=data_dir)

# Z-score pixelwise across trials (same as training)
X_z, _mean, _std = pre.zscore_dataset_pixelwise_trials(X_trials, baseline_frames=baseline)

# Apply ROI + window
ROI_mask = np.load(ROI_mask_path)
X_roi = X_z[ROI_mask, :, :]                        # (roi_pixels, frames, trials)
X_win = fe.extract_window(X_roi, window[0], window[1])  # (roi_pixels, win_frames, trials)

# Frames as samples
X_frames, y_frames, groups = fe.frames_as_samples(X_win, y_trials,
                                                trial_axis=-1, frame_axis=1, pixel_axis=0)

# Trial labels for permutation (aligned with np.unique(groups) order)
unique_groups = np.unique(groups)
if unique_groups.size != y_trials.size:
    raise ValueError(
        f"groups has {unique_groups.size} unique trials, but y_trials has {y_trials.size}. "
        "They must match for trial-level permutation."    )

print("Rebuilt features:")
print("  X_frames:", X_frames.shape)
print("  y_frames:", y_frames.shape)
print("  groups:", groups.shape)
print()

# 3) Run permutation test (TRIAL-level shuffle expanded to frames)
perm_result = mc.run_permutation_nested_cv(X_frames, y_trials, groups,
                                        n_permutations=N_PERMUTATIONS,
                                        outer_splitter=outer,
                                        inner_splitter=inner,
                                        C_grid=C_grid,
                                        metric=metric,
                                        rule=rule,
                                        tie_break=tie_break,
                                        random_seed=PERM_SEED,
                                        verbose=True)

# 4) Compute significance vs the REAL run results you loaded (TRIAL level)
# NOTE: trial-level significance is defined for metric="acc"
stats_trial = mc.permutation_significance_test(real_nested,perm_result)

print("\n=== Permutation significance (TRIAL level) ===")
print(f"Real trial acc: {stats_trial['real_score']:.4f}")
print(f"Shuffled mean ± std: {stats_trial['shuffled_mean']:.4f} ± {stats_trial['shuffled_std']:.4f}")
print(f"Shuffled range: [{stats_trial['shuffled_min']:.4f}, {stats_trial['shuffled_max']:.4f}]")
print(f"P-value (two-tailed): {stats_trial['p_value_two_tailed']:.4f}")
print("PASS (alpha=0.05):", stats_trial["pass_alpha_0p05"])
print()

# 5) Save permutation run outputs 
PERM_ROOT.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

model_name = RUN_DIR.name  # folder name of the original model
save_dir = PERM_ROOT / f"{model_name}__permutation__perm{N_PERMUTATIONS}__seed{PERM_SEED}__{ts}"
save_dir.mkdir(parents=True, exist_ok=False)

# config for permutation run
perm_config = {"source_run_dir": str(RUN_DIR),
            "source_model_name": model_name,
            "n_permutations": int(N_PERMUTATIONS),
            "perm_seed": int(PERM_SEED),
            "metric": "acc",
            "level": "trial",
            "rule": rule,
            "tie_break": tie_break,
            "window": list(window),
            "baseline_frames": list(baseline),
            "C_grid": [float(x) for x in C_grid]}

with open(save_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(perm_config, f, indent=2)

np.savez_compressed(
    save_dir / "perm_result.npz",
    # distributions
    shuffled_scores_trials=np.asarray(perm_result["shuffled_scores_trials"], dtype=float),
    shuffled_scores_frames=np.asarray(perm_result["shuffled_scores_frames"], dtype=float),
    # real model summaries (trial + frame)
    real_trial_acc=float(real_nested["outer_acc_trial_mean"]),
    real_trial_std=float(real_nested["outer_acc_trial_std"]),
    real_frame_acc=float(real_nested["outer_acc_mean"]),
    real_frame_std=float(real_nested["outer_acc_std"]),
    # shuffled summaries (trial)
    shuffled_trial_mean=float(stats_trial["shuffled_mean"]),
    shuffled_trial_std=float(stats_trial["shuffled_std"]),
    shuffled_trial_min=float(stats_trial["shuffled_min"]),
    shuffled_trial_max=float(stats_trial["shuffled_max"]),
    # significance (trial)
    p_value_trial_two_tailed=float(stats_trial["p_value_two_tailed"]),
    pass_alpha_0p05_trial=bool(stats_trial["pass_alpha_0p05"]),
)

print("Saved permutation results to:")
print(" ", save_dir)