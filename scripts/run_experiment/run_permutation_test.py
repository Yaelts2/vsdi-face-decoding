# run_permutation_test.py

from pathlib import Path
from datetime import datetime
import json
import numpy as np
from sklearn.model_selection import GroupKFold
from functions_scripts import preprocessing_functions as pre
from functions_scripts import feature_extraction as fe
from functions_scripts import model_control as mc
from functions_scripts import save_results as sr

# ------------------------------------------------------------
# USER: set which saved model/run to test
# ------------------------------------------------------------

RESULTS_ROOT = Path(r"C:\project\vsdi-face-decoding\results")

# Point to the specific run folder you want to test (the folder that contains config.json/result.npz/ROI_mask.npz)
# Example:
RUN_DIR = RESULTS_ROOT / "fixed_window" / "frame32-40__SVM_10fold__2026-02-16_12-03-51"

N_PERMUTATIONS = 100
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
X_roi = X_z[ROI_mask_path, :, :]                        # (roi_pixels, frames, trials)
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

# 3) Run permutation test (trial-level shuffle expanded to frames)
perm_result = mc.run_permutation_nested_cv(X_frames,y_frames=y_frames,
                                        y_trial=y_trials,
                                        groups=groups,
                                        n_permutations=N_PERMUTATIONS,
                                        outer_splitter=outer,
                                        inner_splitter=inner,
                                        C_grid=C_grid,
                                        metric=metric,
                                        rule=rule,
                                        tie_break=tie_break,
                                        random_seed=PERM_SEED,
                                        verbose=True)

# 4) Compute significance vs the REAL run results you loaded
stats = mc.permutation_significance_test(real_nested, perm_result, metric=metric, chance_level=0.5)

print("\n=== Permutation significance ===")
print(f"Real {metric}: {stats['real_score']:.4f}")
print(f"Shuffled mean ± std: {stats['shuffled_mean']:.4f} ± {stats['shuffled_std']:.4f}")
print(f"Shuffled range: [{stats['shuffled_min']:.4f}, {stats['shuffled_max']:.4f}]")
print(f"P-value (two-tailed): {stats['p_value_two_tailed']:.4f}")
print("PASS (alpha=0.05):", stats["pass_alpha_0p05"])
print()

# 5) Save permutation run outputs (everything needed later for plotting)
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
            "metric": metric,
            "rule": rule,
            "tie_break": tie_break,
            "window": list(window),
            "baseline_frames": list(baseline),
            "C_grid": [float(x) for x in C_grid]}

with open(save_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(perm_config, f, indent=2)

# Save permutation scores + stats + (optionally) real score
np.savez_compressed(save_dir / "perm_result.npz",
                    shuffled_scores=np.asarray(perm_result["shuffled_scores"], dtype=float),
                    # real model summary
                    real_score=float(stats["real_score"]),
                    real_std=float(real_nested["outer_acc_std"]),
                    # shuffled summary
                    shuffled_mean=float(stats["shuffled_mean"]),
                    shuffled_std=float(stats["shuffled_std"]),
                    shuffled_min=float(stats["shuffled_min"]),
                    shuffled_max=float(stats["shuffled_max"]),
                    # significance
                    p_value=float(stats["p_value_two_tailed"]),
                    pass_alpha_0p05=bool(stats["pass_alpha_0p05"]))


print("Saved permutation results to:")
print(" ", save_dir)
