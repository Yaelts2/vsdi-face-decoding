"""
errorTrials_Functions.py

All functions for Aim 3, Part 2 -- error trial analysis:
1. Data loading & condition selection
2. Normalization (matching correct-trial training stats)
3. Decoding (final model + fold-averaged)
4. Session-level pipeline wrappers (training + error-trial processing)
5. Statistical comparisons (correct vs. error, single session)
6. Plotting (single session)
7. Multi-session grand average (accuracy AND score)
8. Group-level significance testing (per-timepoint + windowed + effect size)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats
from scipy.stats import wilcoxon

from scripts.functions_scripts import preprocessing_functions as pre
from scripts.functions_scripts import ml_cv as cv
from scripts.functions_scripts import sliding_win as sw
from scripts.functions_scripts import feature_extraction as fe
from scripts.functions_scripts.save_results import save_experiment, load_data_from_config


# =====================================================================
# 1. DATA LOADING & CONDITION SELECTION
# =====================================================================

def load_error_trials(mat_path):
    """
    Load error-trial VSD data + metadata saved from MATLAB.

    Returns:
        vsd_matrix : (pixels, frames, n_trials) array
        meta : dict with keys cortexTrialId, condId, errType, physicalLabel,
            behavioralLabel -- physicalLabel/behavioralLabel are read DIRECTLY
            from the .mat file's own errorTrialsStruct fields (not re-derived
            here), since the MATLAB side already computes them correctly
            per-trial accounting for errType. Do NOT recompute behavioralLabel
            as 1-physicalLabel downstream -- that assumes every error is a
            clean flip, which may not hold for all errType values (e.g.
            fixation breaks, no-response trials).
    """
    mat_path = Path(mat_path)
    d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    vsd_matrix = np.asarray(d["vsdMatrix"])
    s = d["errorTrialsStruct"]

    meta = {"cortexTrialId": np.atleast_1d(s.cortexTrialId).astype(int),
            "condId": np.atleast_1d(s.condId).astype(int),
            "errType": list(np.atleast_1d(s.errType)),
            "physicalLabel": np.atleast_1d(s.physicalLabel).astype(float),
            "behavioralLabel": np.atleast_1d(s.behavioralLabel).astype(float)}

    n_trials_vsd = vsd_matrix.shape[2]
    n_trials_meta = meta["cortexTrialId"].shape[0]
    if n_trials_vsd != n_trials_meta:
        raise ValueError(f"Mismatch: vsdMatrix has {n_trials_vsd} trials, "
                          f"metadata has {n_trials_meta} trials.")

    print(f"Loaded {n_trials_vsd} error trials (all conditions).")
    return vsd_matrix, meta


def select_conditions(vsd_matrix, meta, face_cond, nonface_cond):
    """
    Filter error trials down to only the face/nonface conditions this
    specific trained model used. physicalLabel/behavioralLabel are the file's
    OWN values (loaded in load_error_trials), just masked down to the kept
    trials here -- NOT recomputed. A sanity check compares the file's
    physicalLabel against the condId-based convention (1 = face_cond) to
    catch any label-convention mismatch between this .mat file and the rest
    of the pipeline.
    """
    cond_id = meta["condId"]
    keep = np.isin(cond_id, [face_cond, nonface_cond])

    vsd_filtered = vsd_matrix[:, :, keep]
    meta_filtered = {k: (np.asarray(v)[keep] if k != "errType" else [v[i] for i in np.where(keep)[0]])
                      for k, v in meta.items()}

    condid_based_physical = (meta_filtered["condId"] == face_cond).astype(float)
    mismatch = meta_filtered["physicalLabel"] != condid_based_physical
    if mismatch.any():
        print(f"WARNING: {mismatch.sum()} trial(s) where the .mat file's physicalLabel "
              f"disagrees with the condId==face_cond convention. Using the file's own "
              f"physicalLabel (it's the source of truth), but this is worth investigating "
              f"-- check errType for these trials.")

    if np.isnan(meta_filtered["physicalLabel"]).any() or np.isnan(meta_filtered["behavioralLabel"]).any():
        n_nan_phys = np.isnan(meta_filtered["physicalLabel"]).sum()
        n_nan_behav = np.isnan(meta_filtered["behavioralLabel"]).sum()
        print(f"WARNING: NaN found in physicalLabel (n={n_nan_phys}) or behavioralLabel "
              f"(n={n_nan_behav}) after filtering -- likely an errType (e.g. no-response) "
              f"where behavioralLabel isn't well-defined. These trials should probably be "
              f"excluded before downstream analysis; not dropped automatically here.")
    else:
        meta_filtered["physicalLabel"] = meta_filtered["physicalLabel"].astype(int)
        meta_filtered["behavioralLabel"] = meta_filtered["behavioralLabel"].astype(int)

    print(f"Selected {keep.sum()} trials from conditions {face_cond} (face) / {nonface_cond} (nonface).")
    print(f"  face={np.sum(meta_filtered['physicalLabel']==1)}, "
          f"nonface={np.sum(meta_filtered['physicalLabel']==0)}")

    return vsd_filtered, meta_filtered


# =====================================================================
# 2. NORMALIZATION
# =====================================================================

def normalize_error_trials(vsd_matrix, baseline_frames, std_pooled, eps=1e-8):
    """
    Normalize error-trial VSD data using the SAME std_pooled from
    correct-trial training. Per-trial baseline mean is computed fresh
    on the error trials themselves; std_pooled is reused, never recomputed.

    ONLY VALID when `baseline_frames` indexes a true pre-stimulus window of
    `vsd_matrix` itself -- i.e. stimulus-aligned data. Do NOT use this on
    RT-aligned data (see normalize_with_external_baseline below for that case).
    """
    X = np.asarray(vsd_matrix, dtype=float)
    std_pooled = np.asarray(std_pooled, dtype=float)

    if X.shape[0] != std_pooled.shape[0]:
        raise ValueError(
            f"Pixel dimension mismatch: vsd_matrix has {X.shape[0]} pixels, "
            f"std_pooled has {std_pooled.shape[0]} pixels. "
            "Did you accidentally ROI-mask before normalizing?"
        )

    start, end = baseline_frames
    baseline = X[:, start:end, :]
    mean_per_trial = baseline.mean(axis=1, keepdims=True)

    X_centered = X - mean_per_trial
    X_z = X_centered / np.maximum(std_pooled, eps)
    return X_z


def normalize_with_external_baseline(vsd_matrix, baseline_mean_per_trial, std_pooled, eps=1e-8):
    """
    Like normalize_error_trials, but the per-trial baseline MEAN is supplied
    externally instead of being computed from vsd_matrix's own frames.

    This exists for RT-aligned data, which has no true pre-stimulus period of
    its own (it's cut 30 frames before / 15 after the reaction time, so
    "early" frames in the RT cut can already contain real evoked signal).
    The baseline mean for each RT-aligned trial must instead come from that
    SAME trial's true pre-stimulus window in the stimulus-aligned data --
    computed elsewhere (see load_stim_aligned_error_baseline) and passed in
    here. std_pooled is still reused from correct-trial training, unchanged.

    Parameters
    ----------
    vsd_matrix : (pixels, frames, trials) array
        The data to be normalized (e.g. RT-aligned error trials).
    baseline_mean_per_trial : (pixels, trials) array
        Per-pixel, per-trial baseline mean, computed from each trial's OWN
        stimulus-aligned pre-stimulus window. Must be in the SAME trial
        order as vsd_matrix.
    std_pooled : (pixels,) array
        Same pooled std used everywhere else in this pipeline (from the
        trained model's correct-trial stats). Not recomputed here.
    """
    X = np.asarray(vsd_matrix, dtype=float)
    std_pooled = np.asarray(std_pooled, dtype=float)
    baseline_mean_per_trial = np.asarray(baseline_mean_per_trial, dtype=float)

    if X.shape[0] != std_pooled.shape[0]:
        raise ValueError(
            f"Pixel dimension mismatch: vsd_matrix has {X.shape[0]} pixels, "
            f"std_pooled has {std_pooled.shape[0]} pixels."
        )
    if baseline_mean_per_trial.shape[0] != X.shape[0]:
        raise ValueError(
            f"Pixel dimension mismatch: vsd_matrix has {X.shape[0]} pixels, "
            f"baseline_mean_per_trial has {baseline_mean_per_trial.shape[0]} pixels."
        )
    if baseline_mean_per_trial.shape[-1] != X.shape[2]:
        raise ValueError(
            f"Trial count mismatch: vsd_matrix has {X.shape[2]} trials, "
            f"baseline_mean_per_trial has {baseline_mean_per_trial.shape[-1]} trials. "
            "These must be in matching trial order -- check the upstream "
            "condition-count sanity check that should have caught this earlier."
        )

    # (pixels, trials) -> (pixels, 1, trials) so it broadcasts over the frame axis
    mean_per_trial = baseline_mean_per_trial[:, np.newaxis, :]

    X_centered = X - mean_per_trial
    X_z = X_centered / np.maximum(std_pooled, eps)[:, np.newaxis, np.newaxis] \
        if std_pooled.ndim == 1 else X_centered / np.maximum(std_pooled, eps)
    return X_z


def load_stim_aligned_error_baseline(error_mat_path, face_cond, nonface_cond, baseline_frames):
    """
    Compute per-trial baseline mean for error trials from the STIMULUS-
    ALIGNED .mat file, split by condition and concatenated face-then-nonface
    -- matching the exact trial order used by load_rt_aligned_paired for the
    corresponding RT-aligned .npy files.

    IMPORTANT ASSUMPTION: within each condition, the trial order in the .mat
    file (as loaded by load_error_trials, i.e. struct/cortexTrialId order) is
    assumed to match the trial order in the corresponding
    RTaligned_<session_prefix><cond>_error.npy file. There is currently no
    shared trial ID between the two sources to verify this directly -- only
    trial COUNTS per condition can be checked (done by the caller). This is
    the same positional-matching convention already used for the correct
    trials elsewhere in this pipeline.

    Parameters
    ----------
    error_mat_path : str or Path
        Path to this session's errorTrialsStruct .mat file (stimulus-aligned).
    face_cond, nonface_cond : int
        Condition numbers for this model's face/nonface pairing.
    baseline_frames : tuple (start, end)
        Frame indices into the STIM-ALIGNED data's frame axis (true
        pre-stimulus window).

    Returns
    -------
    baseline_mean : (pixels, n_face + n_nonface) array
        Per-trial baseline mean, face trials first then nonface -- same
        concatenation order as load_rt_aligned_paired.
    n_face, n_nonface : int
        Trial counts per condition, for the caller to verify against the
        RT-aligned file's own trial counts before trusting the pairing.
    """
    vsd_matrix, meta = load_error_trials(error_mat_path)
    cond_id = meta["condId"]

    face_mask = cond_id == face_cond
    nonface_mask = cond_id == nonface_cond

    n_face = int(face_mask.sum())
    n_nonface = int(nonface_mask.sum())

    b0, b1 = baseline_frames
    face_baseline = vsd_matrix[:, b0:b1, face_mask]
    nonface_baseline = vsd_matrix[:, b0:b1, nonface_mask]

    face_mean = face_baseline.mean(axis=1)       # (pixels, n_face)
    nonface_mean = nonface_baseline.mean(axis=1)  # (pixels, n_nonface)

    baseline_mean = np.concatenate([face_mean, nonface_mean], axis=1)  # (pixels, n_face+n_nonface)

    print(f"Stim-aligned baseline computed for RT error trials: "
          f"face(cond {face_cond})={n_face}, nonface(cond {nonface_cond})={n_nonface} "
          f"trials, from {error_mat_path}")

    return baseline_mean, n_face, n_nonface


# =====================================================================
# 3. DECODING
# =====================================================================

def decode_error_trials(X_error_roi, results):
    """
    Apply the final (100%-trained) sliding-window decoders to error trials.
    """
    final_models = results["final_models"]
    centers = results["centers"]
    window_size = results["params"]["window_size"]

    n_trials = X_error_roi.shape[2]
    n_windows = len(final_models)

    trial_pred = np.zeros((n_windows, n_trials), dtype=int)
    trial_score = np.zeros((n_windows, n_trials), dtype=float)

    for w_idx, center in enumerate(centers):
        start = center - window_size // 2
        end = start + window_size
        X_win = X_error_roi[:, start:end, :]
        clf = final_models[w_idx]

        for trial_idx in range(n_trials):
            X_frames_trial = X_win[:, :, trial_idx].T
            scores = clf.decision_function(X_frames_trial)
            preds = (scores > 0).astype(int)
            trial_pred[w_idx, trial_idx] = np.bincount(preds).argmax()
            trial_score[w_idx, trial_idx] = scores.mean()

    return {"centers": centers, "trial_pred": trial_pred, "trial_score": trial_score}


def decode_error_trials_fold_averaged(X_error_roi, results, target_center):
    """
    Apply ALL fold-trained models (not just final_models) for one window
    to error trials, average scores across folds per trial.
    """
    centers = results["centers"]
    win_idx = np.where(centers == target_center)[0]
    if len(win_idx) == 0:
        raise ValueError(f"No window with center={target_center}")
    win_idx = win_idx[0]

    window_size = results["params"]["window_size"]
    start = target_center - window_size // 2
    end = start + window_size

    fold_models = results["fold_models_all"][win_idx]
    n_trials = X_error_roi.shape[2]
    n_folds = len(fold_models)

    fold_trial_scores = np.zeros((n_folds, n_trials))
    for f_idx, clf in enumerate(fold_models):
        for trial_idx in range(n_trials):
            X_frames_trial = X_error_roi[:, start:end, trial_idx].T
            fold_trial_scores[f_idx, trial_idx] = clf.decision_function(X_frames_trial).mean()

    fold_averaged_score = fold_trial_scores.mean(axis=0)
    return fold_averaged_score, fold_trial_scores


# =====================================================================
# 4. SESSION-LEVEL PIPELINE WRAPPERS
# =====================================================================

def process_error_trials_for_session(error_mat_path, results, ROI_mask_path, y_trials,
                                      face_cond, nonface_cond,
                                      Baseline_frames_zscore=(1, 24),
                                      match_correct_to_error=False, subsample_seed=42,
                                      fallback_target=15):
    """
    Full error-trial pipeline for one session: load, filter to matched
    conditions, normalize, ROI mask, compute BOTH accuracy and
    label-consistent score comparisons across windows.

    Parameters
    ----------
    match_correct_to_error : bool
        If True, randomly subsample the correct trials ONCE (before computing
        any window) down to the same count as this session's error trials,
        so both groups contribute equal noise to the comparison (matches
        Ayzenshtat et al. 2012's approach for exactly this reason). If False
        (default), all correct trials are used, as before.

        If this session has MORE error trials than correct trials (so
        matching correct DOWN to the error count isn't possible), both
        groups are instead subsampled down to `fallback_target` trials each,
        so the noise-matching principle still holds for these sessions too.
    subsample_seed : int
        Seed for the subsampling, used only if match_correct_to_error=True.
    fallback_target : int
        Trial count used for BOTH groups when error trials outnumber correct
        trials for this session (see above). Default 15.

    Returns (acc_results, score_results, meta, X_error_roi)
    """
    vsd_matrix, meta = load_error_trials(error_mat_path)
    vsd_matrix, meta = select_conditions(vsd_matrix, meta, face_cond, nonface_cond)

    X_error_z = normalize_error_trials(vsd_matrix, Baseline_frames_zscore, results["zscore_std_pooled"])

    ROI_mask = np.load(ROI_mask_path).astype(bool)
    X_error_roi = X_error_z[ROI_mask, :, :]

    physical = meta["physicalLabel"]
    n_error = len(physical)

    if match_correct_to_error:
        if n_error <= len(y_trials):
            correct_subset_idx = subsample_correct_indices(len(y_trials), n_error, seed=subsample_seed)
            print(f"Matching correct trials to error count: using {n_error} of {len(y_trials)} correct trials.")
        else:
            n_target = min(fallback_target, len(y_trials), n_error)
            if n_target < fallback_target:
                print(f"WARNING: requested fallback_target={fallback_target}, but this session only "
                      f"has {len(y_trials)} correct / {n_error} error trials -- using {n_target} "
                      f"of each instead.")
            correct_subset_idx = subsample_correct_indices(len(y_trials), n_target, seed=subsample_seed)
            error_subset_idx = subsample_correct_indices(n_error, n_target, seed=subsample_seed + 1)
            print(f"Session has more error trials ({n_error}) than correct trials ({len(y_trials)}) -- "
                  f"subsampling BOTH groups down to {n_target} trials each instead of the usual "
                  f"correct-to-error matching.")
            # subset the error trials (and their corresponding VSDI data) here, since
            # compare_*_across_windows always use ALL trials in X_error_roi/physical otherwise
            physical = physical[error_subset_idx]
            X_error_roi = X_error_roi[:, :, error_subset_idx]
    else:
        correct_subset_idx = None

    acc_results = compare_accuracy_across_windows(X_error_roi, results, physical, y_trials,
                                                    correct_subset_idx=correct_subset_idx)
    score_results = compare_fold_averaged_across_windows(X_error_roi, results, physical, y_trials,
                                                           correct_subset_idx=correct_subset_idx)

    return acc_results, score_results, meta, X_error_roi


def load_rt_aligned_paired(session_prefix, face_cond, nonface_cond, kind, base_dir):
    """
    Load and combine RT-aligned trial data for one face/nonface condition
    pair, from the per-condition .npy files. Unlike the stimulus-locked
    errorTrialsStruct .mat files, these have NO embedded metadata --
    physicalLabel is determined structurally by which condition file a
    trial came from (condition number alone tells you the true stimulus
    category), not read from a struct field.

    File naming on disk: RTaligned_<session_prefix><cond>_<kind>.npy
    e.g. RTaligned_110209a1_error.npy for session_prefix="110209a", cond=1.

    Parameters
    ----------
    session_prefix : str
        Session-day code WITHOUT the condition-pair suffix, e.g. "110209a"
        (not "110209a15" -- that suffix is split into face_cond/nonface_cond).
    face_cond, nonface_cond : int
        The two condition numbers making up this model's face/nonface pair
        (e.g. 1 and 5 for the "15" pairing, 2 and 4 for the "24" pairing).
    kind : {'error', 'corrct'}
        Matches the actual filename spelling on disk (note: 'corrct', not
        'correct').
    base_dir : str or Path
        Root of the RT-aligned data tree, e.g.
        r"C:\\project\\vsdi-face-decoding\\data\\processed\\RTaligend"

    Returns
    -------
    combined : (pixels, frames, n_trials) array -- face trials first, then nonface
    physical : (n_trials,) int array -- 1 for face trials, 0 for nonface trials
    """
    base_dir = Path(base_dir)
    face_path = base_dir / session_prefix / f"RTaligned_{session_prefix}{face_cond}_{kind}.npy"
    nonface_path = base_dir / session_prefix / f"RTaligned_{session_prefix}{nonface_cond}_{kind}.npy"

    face_arr = np.load(face_path)
    nonface_arr = np.load(nonface_path)

    # Defensive fix: a session with only ONE trial for a condition sometimes
    # gets saved as 2D (pixels, frames) instead of 3D (pixels, frames, 1) --
    # the trailing single-trial axis gets squeezed out somewhere upstream.
    # Restore it here rather than crashing on shape[2].
    if face_arr.ndim == 2:
        print(f"WARNING: face(cond {face_cond}) array is 2D {face_arr.shape} -- "
              f"assuming a single squeezed trial, reshaping to add trial axis.")
        face_arr = face_arr[:, :, np.newaxis]
    elif face_arr.ndim != 3:
        raise ValueError(f"Unexpected shape for {face_path}: {face_arr.shape} (ndim={face_arr.ndim}). "
                          f"Expected 3D (pixels, frames, trials).")

    if nonface_arr.ndim == 2:
        print(f"WARNING: nonface(cond {nonface_cond}) array is 2D {nonface_arr.shape} -- "
              f"assuming a single squeezed trial, reshaping to add trial axis.")
        nonface_arr = nonface_arr[:, :, np.newaxis]
    elif nonface_arr.ndim != 3:
        raise ValueError(f"Unexpected shape for {nonface_path}: {nonface_arr.shape} (ndim={nonface_arr.ndim}). "
                          f"Expected 3D (pixels, frames, trials).")

    print(f"Loaded RT-aligned {kind} trials: face(cond {face_cond})={face_arr.shape[2]} "
          f"(raw shape {face_arr.shape}), nonface(cond {nonface_cond})={nonface_arr.shape[2]} "
          f"(raw shape {nonface_arr.shape})")

    combined = np.concatenate([face_arr, nonface_arr], axis=2)
    physical = np.concatenate([
        np.ones(face_arr.shape[2], dtype=int),
        np.zeros(nonface_arr.shape[2], dtype=int),
    ])
    return combined, physical


def process_rt_aligned_trials_for_session(session_prefix, face_cond, nonface_cond, base_dir,
                                           results, ROI_mask_path, y_trials,
                                           error_mat_path,
                                           Baseline_frames_zscore=(1, 24),
                                           match_correct_to_error=False, subsample_seed=42,
                                           fallback_target=15):
    """
    RT-aligned equivalent of process_error_trials_for_session. Error trials
    come from the per-condition RT-aligned .npy files (combined via
    load_rt_aligned_paired) instead of an errorTrialsStruct .mat file --
    everything downstream (ROI masking, accuracy/score comparison across
    windows, optional trial-count matching) is identical, so all existing
    plotting/significance-testing code works unchanged on the output.

    BASELINE FIX (see load_stim_aligned_error_baseline /
    normalize_with_external_baseline): RT-aligned trials are cut 30 frames
    before / 15 after the reaction time, so they have no true pre-stimulus
    period of their own -- frames 1-24 of the RT cut can already contain real
    evoked signal, and how much varies trial-to-trial with RT length. So the
    per-trial baseline MEAN used here comes from each error trial's OWN
    stimulus-aligned data (error_mat_path), not from the RT-aligned cut
    itself. std_pooled is still reused from the trained model, unchanged.

    This relies on POSITIONAL trial matching within each condition between
    the .mat file and the RT .npy files (see load_stim_aligned_error_baseline
    docstring) -- trial COUNTS per condition are checked below and will raise
    a loud error if they disagree, but order itself can't be independently
    verified without a shared trial ID.

    Correct-trial performance still comes from the trained model's own
    cross-validated results (results["oof_trial_score"] + y_trials), exactly
    as in the stimulus-locked pipeline -- these already reflect RT-aligned
    correct-trial performance IF the model itself was trained on this same
    RT-aligned data (as the "RTaligned_" model folder naming implies).

    Returns (acc_results, score_results, meta, X_error_roi) -- same shape as
    process_error_trials_for_session.
    """
    vsd_matrix, physical = load_rt_aligned_paired(session_prefix, face_cond, nonface_cond,
                                                    kind="error", base_dir=base_dir)

    baseline_mean, n_face_stim, n_nonface_stim = load_stim_aligned_error_baseline(
        error_mat_path, face_cond, nonface_cond, Baseline_frames_zscore)

    n_face_rt = int((physical == 1).sum())
    n_nonface_rt = int((physical == 0).sum())
    if n_face_stim != n_face_rt or n_nonface_stim != n_nonface_rt:
        raise ValueError(
            f"Trial count mismatch between stim-aligned error .mat ({error_mat_path}) and "
            f"RT-aligned error .npy files for session_prefix={session_prefix}: "
            f"stim-aligned face={n_face_stim}/nonface={n_nonface_stim}, "
            f"RT-aligned face={n_face_rt}/nonface={n_nonface_rt}. "
            f"Positional trial matching is unsafe with mismatched counts -- stopping."
        )

    X_error_z = normalize_with_external_baseline(vsd_matrix, baseline_mean, results["zscore_std_pooled"])

    ROI_mask = np.load(ROI_mask_path).astype(bool)
    X_error_roi = X_error_z[ROI_mask, :, :]

    n_error = len(physical)

    if match_correct_to_error:
        if n_error <= len(y_trials):
            correct_subset_idx = subsample_correct_indices(len(y_trials), n_error, seed=subsample_seed)
            print(f"Matching correct trials to error count: using {n_error} of {len(y_trials)} correct trials.")
        else:
            n_target = min(fallback_target, len(y_trials), n_error)
            if n_target < fallback_target:
                print(f"WARNING: requested fallback_target={fallback_target}, but this session only "
                      f"has {len(y_trials)} correct / {n_error} error trials -- using {n_target} "
                      f"of each instead.")
            correct_subset_idx = subsample_correct_indices(len(y_trials), n_target, seed=subsample_seed)
            error_subset_idx = subsample_correct_indices(n_error, n_target, seed=subsample_seed + 1)
            print(f"Session has more error trials ({n_error}) than correct trials ({len(y_trials)}) -- "
                  f"subsampling BOTH groups down to {n_target} trials each instead of the usual "
                  f"correct-to-error matching.")
            physical = physical[error_subset_idx]
            X_error_roi = X_error_roi[:, :, error_subset_idx]
    else:
        correct_subset_idx = None

    acc_results = compare_accuracy_across_windows(X_error_roi, results, physical, y_trials,
                                                    correct_subset_idx=correct_subset_idx)
    score_results = compare_fold_averaged_across_windows(X_error_roi, results, physical, y_trials,
                                                           correct_subset_idx=correct_subset_idx)

    meta = {"physicalLabel": physical}
    return acc_results, score_results, meta, X_error_roi


# =====================================================================
# 5. STATISTICAL COMPARISONS (single session)
# =====================================================================

def compare_score_magnitude(correct_scores, error_scores, label="Score"):
    """
    Compare label-consistent decoder scores between correct and error trials.
    Inputs must already be sign-flipped (positive = matches trial's own true label).
    """
    correct_scores = np.asarray(correct_scores)
    error_scores = np.asarray(error_scores)

    n_correct, n_error = len(correct_scores), len(error_scores)
    mean_correct, mean_error = correct_scores.mean(), error_scores.mean()
    sd_correct, sd_error = correct_scores.std(ddof=1), error_scores.std(ddof=1)

    t_stat, p_welch = stats.ttest_ind(correct_scores, error_scores, equal_var=False)
    u_stat, p_mw = stats.mannwhitneyu(correct_scores, error_scores, alternative="two-sided")

    pooled_sd = np.sqrt(((n_correct - 1) * sd_correct ** 2 + (n_error - 1) * sd_error ** 2) / (n_correct + n_error - 2))
    cohens_d = (mean_correct - mean_error) / pooled_sd

    print(f"--- {label} ---")
    print(f"Correct trials: n={n_correct}, mean={mean_correct:.3f}, SD={sd_correct:.3f}")
    print(f"Error trials:   n={n_error}, mean={mean_error:.3f}, SD={sd_error:.3f}")
    print(f"Welch's t-test: t={t_stat:.3f}, p={p_welch:.4f}")
    print(f"Mann-Whitney U: U={u_stat:.1f}, p={p_mw:.4f}")
    print(f"Cohen's d: {cohens_d:.3f}")

    return {
        "n_correct": n_correct, "n_error": n_error,
        "mean_correct": mean_correct, "mean_error": mean_error,
        "sd_correct": sd_correct, "sd_error": sd_error,
        "t_stat": t_stat, "p_welch": p_welch,
        "u_stat": u_stat, "p_mw": p_mw, "cohens_d": cohens_d,
    }


def compare_fold_averaged_across_windows(X_error_roi, results, physical, y_trials, correct_subset_idx=None):
    """
    For every window: mean sign-flipped (label-consistent) score for
    correct trials (single held-out fold) vs. error trials (fold-averaged).

    Also returns the raw per-trial sign-flipped score matrices
    (correct_trial_values, error_trial_values -- shape (n_windows, n_trials)).

    Parameters
    ----------
    correct_subset_idx : 1D int array or None
        If given, restricts the correct trials used to this subset of indices
        into y_trials/results["oof_trial_score"] -- e.g. to match the correct
        trial count to the error trial count (Ayzenshtat et al. 2012 style).
        Chosen ONCE per session (not re-sampled per window) via
        subsample_correct_indices, so the same trials are used at every window.
    """
    centers = results["centers"]
    n_windows = len(centers)
    window_size = results["params"]["window_size"]

    if correct_subset_idx is not None:
        y_trials_used = y_trials[correct_subset_idx]
    else:
        y_trials_used = y_trials
    correct_sign_flip = np.where(y_trials_used == 1, 1, -1)
    error_sign_flip = np.where(physical == 1, 1, -1)

    n_correct, n_error = len(y_trials_used), len(physical)

    correct_mean_per_window = np.zeros(n_windows)
    error_mean_per_window = np.zeros(n_windows)
    correct_sem_per_window = np.zeros(n_windows)
    error_sem_per_window = np.zeros(n_windows)

    correct_trial_values = np.zeros((n_windows, n_correct))
    error_trial_values = np.zeros((n_windows, n_error))

    for w_idx, center in enumerate(centers):
        correct_raw_full = results["oof_trial_score"][w_idx, :]
        correct_raw = correct_raw_full[correct_subset_idx] if correct_subset_idx is not None else correct_raw_full
        correct_consistent = correct_raw * correct_sign_flip
        correct_mean_per_window[w_idx] = correct_consistent.mean()
        correct_sem_per_window[w_idx] = correct_consistent.std(ddof=1) / np.sqrt(len(correct_consistent))
        correct_trial_values[w_idx, :] = correct_consistent

        fold_models = results["fold_models_all"][w_idx]
        start = center - window_size // 2
        end = start + window_size
        n_trials_err = X_error_roi.shape[2]
        fold_scores = np.zeros((len(fold_models), n_trials_err))
        for f_idx, clf in enumerate(fold_models):
            for trial_idx in range(n_trials_err):
                X_frames_trial = X_error_roi[:, start:end, trial_idx].T
                fold_scores[f_idx, trial_idx] = clf.decision_function(X_frames_trial).mean()
        error_raw = fold_scores.mean(axis=0)
        error_consistent = error_raw * error_sign_flip
        error_mean_per_window[w_idx] = error_consistent.mean()
        error_sem_per_window[w_idx] = error_consistent.std(ddof=1) / np.sqrt(len(error_consistent))
        error_trial_values[w_idx, :] = error_consistent

    print(f"Correct trials: n={n_correct}")
    print(f"Error trials:   n={n_error}")

    return {
        "centers": centers,
        "correct_mean": correct_mean_per_window, "correct_sem": correct_sem_per_window,
        "error_mean": error_mean_per_window, "error_sem": error_sem_per_window,
        "n_correct": n_correct, "n_error": n_error,
        "correct_trial_values": correct_trial_values,
        "error_trial_values": error_trial_values,
    }


def compare_accuracy_across_windows(X_error_roi, results, physical, y_trials, correct_subset_idx=None):
    """
    Same structure as compare_fold_averaged_across_windows, but computes
    trial-level ACCURACY (fraction correct) instead of raw decision scores.

    Also returns the raw per-trial correctness matrices (correct_trial_values,
    error_trial_values -- shape (n_windows, n_trials)).

    Parameters
    ----------
    correct_subset_idx : 1D int array or None
        If given, restricts the correct trials used to this subset of indices
        into y_trials/results["oof_trial_score"] -- e.g. to match the correct
        trial count to the error trial count (Ayzenshtat et al. 2012 style).
        Chosen ONCE per session (not re-sampled per window) via
        subsample_correct_indices, so the same trials are used at every window.
    """
    centers = results["centers"]
    n_windows = len(centers)
    window_size = results["params"]["window_size"]

    if correct_subset_idx is not None:
        y_trials_used = y_trials[correct_subset_idx]
    else:
        y_trials_used = y_trials

    n_correct, n_error = len(y_trials_used), len(physical)

    correct_acc_per_window = np.zeros(n_windows)
    error_acc_per_window = np.zeros(n_windows)
    correct_sem_per_window = np.zeros(n_windows)
    error_sem_per_window = np.zeros(n_windows)

    correct_trial_values = np.zeros((n_windows, n_correct))
    error_trial_values = np.zeros((n_windows, n_error))

    for w_idx, center in enumerate(centers):
        correct_raw_full = results["oof_trial_score"][w_idx, :]
        correct_raw = correct_raw_full[correct_subset_idx] if correct_subset_idx is not None else correct_raw_full
        correct_pred = (correct_raw > 0).astype(int)
        correct_is_right = (correct_pred == y_trials_used).astype(float)
        correct_acc_per_window[w_idx] = correct_is_right.mean()
        correct_sem_per_window[w_idx] = correct_is_right.std(ddof=1) / np.sqrt(len(correct_is_right))
        correct_trial_values[w_idx, :] = correct_is_right

        fold_models = results["fold_models_all"][w_idx]
        start = center - window_size // 2
        end = start + window_size
        n_trials_err = X_error_roi.shape[2]

        fold_correctness = np.zeros((len(fold_models), n_trials_err))
        for f_idx, clf in enumerate(fold_models):
            for trial_idx in range(n_trials_err):
                X_frames_trial = X_error_roi[:, start:end, trial_idx].T
                score = clf.decision_function(X_frames_trial).mean()
                pred = int(score > 0)
                fold_correctness[f_idx, trial_idx] = float(pred == physical[trial_idx])

        per_trial_acc = fold_correctness.mean(axis=0)
        error_acc_per_window[w_idx] = per_trial_acc.mean()
        error_sem_per_window[w_idx] = per_trial_acc.std(ddof=1) / np.sqrt(len(per_trial_acc))
        error_trial_values[w_idx, :] = per_trial_acc

    print(f"Correct trials: n={n_correct}")
    print(f"Error trials:   n={n_error}")

    return {
        "centers": centers,
        "correct_acc": correct_acc_per_window, "error_acc": error_acc_per_window,
        "correct_sem": correct_sem_per_window, "error_sem": error_sem_per_window,
        "n_correct": n_correct, "n_error": n_error,
        "correct_trial_values": correct_trial_values,
        "error_trial_values": error_trial_values,
    }


def subsample_correct_indices(n_correct_total, n_target, seed=42):
    """
    Randomly choose n_target indices out of n_correct_total correct trials,
    WITHOUT replacement -- used to match the correct-trial count to the
    error-trial count per session (Ayzenshtat et al. 2012 style), so both
    groups contribute equal noise to the comparison instead of correct
    trials being averaged over a much larger, less noisy sample.

    Chosen ONCE per session with a fixed seed (not resampled per window),
    so the same subset of trials is used consistently across the whole
    sliding-window analysis for that session.

    Parameters
    ----------
    n_correct_total : int
        Total number of correct trials available for this session.
    n_target : int
        Number of correct trials to keep (typically the error trial count).
    seed : int
        For reproducibility -- use a DIFFERENT seed per session if you want
        independent subsamples, or the same seed if you want the same random
        draw pattern applied consistently (e.g. seed + session index).

    Returns
    -------
    1D int array of selected indices, sorted.
    """
    if n_target > n_correct_total:
        raise ValueError(f"Cannot subsample {n_target} from only {n_correct_total} correct trials.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_correct_total, size=n_target, replace=False)
    return np.sort(idx)


# =====================================================================
# 6. PLOTTING (single session)
# =====================================================================

def plot_error_trial_decoding(decode_results, meta):
    centers = decode_results["centers"]
    trial_score = decode_results["trial_score"]
    physical = meta["physicalLabel"]

    face_trials = physical == 1
    nonface_trials = physical == 0

    avg_face = trial_score[:, face_trials].mean(axis=1)
    sem_face = trial_score[:, face_trials].std(axis=1, ddof=1) / np.sqrt(face_trials.sum())
    avg_nonface = trial_score[:, nonface_trials].mean(axis=1)
    sem_nonface = trial_score[:, nonface_trials].std(axis=1, ddof=1) / np.sqrt(nonface_trials.sum())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centers, avg_face, color="tab:red",
            label=f"Physically FACE, reported NONFACE (n={face_trials.sum()})")
    ax.fill_between(centers, avg_face - sem_face, avg_face + sem_face, color="tab:red", alpha=0.2)
    ax.plot(centers, avg_nonface, color="tab:blue",
            label=f"Physically NONFACE, reported FACE (n={nonface_trials.sum()})")
    ax.fill_between(centers, avg_nonface - sem_nonface, avg_nonface + sem_nonface, color="tab:blue", alpha=0.2)
    ax.axhline(0, color="black", linestyle="--", linewidth=1, label="Decision boundary")
    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel("Decoder decision score\n(+ = face-like, - = nonface-like)")
    ax.set_title("Decoder output on error trials, by physical stimulus category")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_match_fractions(decode_results, physical, behavioral):
    centers = decode_results["centers"]
    match_physical = (decode_results["trial_pred"] == physical[None, :]).mean(axis=1)
    match_behavioral = (decode_results["trial_pred"] == behavioral[None, :]).mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centers, match_physical, color="tab:green", label="Matches physical stimulus")
    ax.plot(centers, match_behavioral, color="tab:purple", label="Matches monkey's percept")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance (50%)")
    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel("Fraction of error trials")
    ax.set_title("Decoder output on error trials: physical stimulus vs. percept")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_match_fractions_by_category(decode_results, physical, behavioral):
    centers = decode_results["centers"]
    trial_pred = decode_results["trial_pred"]

    face_trials = physical == 1
    nonface_trials = physical == 0
    n_face, n_nonface = face_trials.sum(), nonface_trials.sum()

    match_behavioral_face = (trial_pred[:, face_trials] == behavioral[face_trials][None, :]).mean(axis=1)
    match_behavioral_nonface = (trial_pred[:, nonface_trials] == behavioral[nonface_trials][None, :]).mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centers, match_behavioral_face, color="tab:red", label=f"Physically FACE trials (n={n_face})")
    ax.plot(centers, match_behavioral_nonface, color="tab:blue", label=f"Physically NONFACE trials (n={n_nonface})")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance (50%)")
    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel("Fraction matching monkey's percept")
    ax.set_title("Decoder tracks percept, by physical stimulus category")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_trial_heatmap(decode_results, physical, meta):
    centers = decode_results["centers"]
    trial_pred = decode_results["trial_pred"]

    sort_idx = np.argsort(physical)
    sorted_pred = trial_pred[:, sort_idx]
    sorted_physical = physical[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sorted_pred.T, aspect="auto", cmap="coolwarm",
                    extent=[centers[0], centers[-1], 0, sorted_pred.shape[1]],
                    origin="lower", vmin=0, vmax=1)
    n_boundary = np.sum(sorted_physical == 0)
    ax.axhline(n_boundary, color="black", linewidth=1.5)
    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel("Trial (sorted: nonface-physical below, face-physical above)")
    ax.set_title("Decoder guess per trial per window (blue=nonface, red=face)")
    plt.colorbar(im, ax=ax, label="Decoder guess (0=nonface, 1=face)")
    plt.tight_layout()
    return fig, ax


def plot_individual_traces(decode_results, physical):
    centers = decode_results["centers"]
    trial_score = decode_results["trial_score"]

    face_trials = physical == 1
    nonface_trials = physical == 0
    avg_face = trial_score[:, face_trials].mean(axis=1)
    avg_nonface = trial_score[:, nonface_trials].mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(trial_score.shape[1]):
        color = "tab:red" if physical[i] == 1 else "tab:blue"
        ax.plot(centers, trial_score[:, i], color=color, alpha=0.25, linewidth=1)
    ax.plot(centers, avg_face, color="darkred", linewidth=2.5, label=f"Face-physical avg (n={face_trials.sum()})")
    ax.plot(centers, avg_nonface, color="darkblue", linewidth=2.5, label=f"Nonface-physical avg (n={nonface_trials.sum()})")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel("Decoder decision score")
    ax.set_title("Individual error trial trajectories with group averages")
    ax.legend(loc="best")
    plt.tight_layout()
    return fig, ax


def plot_trial_dotplot(decode_results, physical, window_frames=(20, 50)):
    centers = decode_results["centers"]
    trial_score = decode_results["trial_score"]
    win_mask = (centers >= window_frames[0]) & (centers <= window_frames[1])
    avg_score = trial_score[win_mask, :].mean(axis=0)
    sign_flip = np.where(physical == 1, 1, -1)
    physical_consistent_score = avg_score * sign_flip

    face_trials = physical == 1
    nonface_trials = physical == 0

    fig, ax = plt.subplots(figsize=(7, 6))
    jitter_face = np.random.uniform(-0.05, 0.05, size=face_trials.sum())
    jitter_nonface = np.random.uniform(-0.05, 0.05, size=nonface_trials.sum())
    ax.scatter(np.zeros(face_trials.sum()) + jitter_face, physical_consistent_score[face_trials],
               color="tab:red", alpha=0.7, label=f"Face-physical (n={face_trials.sum()})")
    ax.scatter(np.ones(nonface_trials.sum()) + jitter_nonface, physical_consistent_score[nonface_trials],
               color="tab:blue", alpha=0.7, label=f"Nonface-physical (n={nonface_trials.sum()})")

    for x, mask, color in [(0, face_trials, "darkred"), (1, nonface_trials, "darkblue")]:
        m = physical_consistent_score[mask].mean()
        sem = physical_consistent_score[mask].std(ddof=1) / np.sqrt(mask.sum())
        ax.errorbar(x, m, yerr=sem, fmt="_", color=color, markersize=30,
                    markeredgewidth=3, capsize=8, linewidth=3, zorder=5)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Face-physical", "Nonface-physical"])
    ax.set_ylabel("Score (+ = matches physical stimulus, - = matches percept)")
    ax.set_title(f"Per-trial decoder score, avg over frames {window_frames[0]}-{window_frames[1]}")
    ax.legend(loc="best")
    plt.tight_layout()
    return fig, ax, physical_consistent_score


def plot_score_dotplot(score_matrix, labels, centers, window_frames, group_names=("Category 0", "Category 1")):
    win_mask = (centers >= window_frames[0]) & (centers <= window_frames[1])
    avg_score = score_matrix[win_mask, :].mean(axis=0)
    sign_flip = np.where(labels == 1, 1, -1)
    label_consistent_score = avg_score * sign_flip

    group0 = labels == 0
    group1 = labels == 1

    fig, ax = plt.subplots(figsize=(7, 6))
    jitter0 = np.random.uniform(-0.05, 0.05, size=group0.sum())
    jitter1 = np.random.uniform(-0.05, 0.05, size=group1.sum())
    ax.scatter(np.zeros(group0.sum()) + jitter0, label_consistent_score[group0],
               color="tab:blue", alpha=0.7, label=f"{group_names[0]} (n={group0.sum()})")
    ax.scatter(np.ones(group1.sum()) + jitter1, label_consistent_score[group1],
               color="tab:red", alpha=0.7, label=f"{group_names[1]} (n={group1.sum()})")

    for x, mask, color in [(0, group0, "darkblue"), (1, group1, "darkred")]:
        m = label_consistent_score[mask].mean()
        sem = label_consistent_score[mask].std(ddof=1) / np.sqrt(mask.sum())
        ax.errorbar(x, m, yerr=sem, fmt="_", color=color, markersize=30,
                    markeredgewidth=3, capsize=8, linewidth=3, zorder=5)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(group_names)
    ax.set_ylabel("Score (+ = matches own label)")
    ax.set_title(f"Per-trial decoder score, avg over frames {window_frames[0]}-{window_frames[1]}")
    ax.legend(loc="best")
    plt.tight_layout()
    return fig, ax, label_consistent_score


def plot_fold_averaged_comparison(comparison_results, session_tag=None):
    centers = comparison_results["centers"]
    n_correct = comparison_results["n_correct"]
    n_error = comparison_results["n_error"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centers, comparison_results["correct_mean"], color="tab:green", label=f"Correct trials (n={n_correct})")
    ax.fill_between(centers,
                     comparison_results["correct_mean"] - comparison_results["correct_sem"],
                     comparison_results["correct_mean"] + comparison_results["correct_sem"],
                     color="tab:green", alpha=0.2)
    ax.plot(centers, comparison_results["error_mean"], color="tab:orange", label=f"Error trials (n={n_error})")
    ax.fill_between(centers,
                     comparison_results["error_mean"] - comparison_results["error_sem"],
                     comparison_results["error_mean"] + comparison_results["error_sem"],
                     color="tab:orange", alpha=0.2)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel("Mean score (+ = matches own true label)")
    title = "Correct vs. error trials: label-consistent decoder confidence across windows"
    if session_tag:
        title = f"[{session_tag}] {title}"
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    return fig, ax


def plot_accuracy_comparison(comparison_results, session_tag=None):
    centers = comparison_results["centers"]
    n_correct = comparison_results["n_correct"]
    n_error = comparison_results["n_error"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centers, comparison_results["correct_acc"], color="tab:green", label=f"Correct trials (n={n_correct})")
    ax.fill_between(centers,
                     comparison_results["correct_acc"] - comparison_results["correct_sem"],
                     comparison_results["correct_acc"] + comparison_results["correct_sem"],
                     color="tab:green", alpha=0.2)
    ax.plot(centers, comparison_results["error_acc"], color="tab:orange", label=f"Error trials (n={n_error})")
    ax.fill_between(centers,
                     comparison_results["error_acc"] - comparison_results["error_sem"],
                     comparison_results["error_acc"] + comparison_results["error_sem"],
                     color="tab:orange", alpha=0.2)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance (50%)")
    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel("Accuracy (matches trial's own true label)")
    ax.set_ylim(0, 1)
    title = "Correct vs. error trials: decoding accuracy across windows"
    if session_tag:
        title = f"[{session_tag}] {title}"
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    return fig, ax


# =====================================================================
# 7. MULTI-SESSION GRAND AVERAGE
# =====================================================================

def grand_average_correct_and_error(correct_acc_list, error_acc_list, session_names, centers_ref):
    """
    Grand-average BOTH correct-trial and error-trial accuracy curves
    across sessions. Each session is one independent replicate.
    """
    for arr in correct_acc_list + error_acc_list:
        if len(arr) != len(centers_ref):
            raise ValueError("Mismatched window count across sessions -- check "
                              "start_frame/stop_frame/window_size consistency.")

    correct_stack = np.vstack(correct_acc_list)
    error_stack = np.vstack(error_acc_list)

    return {
        "centers": centers_ref,
        "grand_correct_mean": correct_stack.mean(axis=0),
        "grand_correct_sem": correct_stack.std(axis=0, ddof=1) / np.sqrt(correct_stack.shape[0]),
        "grand_error_mean": error_stack.mean(axis=0),
        "grand_error_sem": error_stack.std(axis=0, ddof=1) / np.sqrt(error_stack.shape[0]),
        "correct_stack": correct_stack, "error_stack": error_stack,
        "session_names": session_names, "n_sessions": correct_stack.shape[0],
    }


def plot_grand_average_correct_and_error(grand_results, chance_level=0.5, show_individual=True):
    centers = grand_results["centers"]
    n_sessions = grand_results["n_sessions"]

    fig, ax = plt.subplots(figsize=(9, 5))

    if show_individual:
        for i, name in enumerate(grand_results["session_names"]):
            ax.plot(centers, grand_results["correct_stack"][i, :], color="tab:green",
                    alpha=0.25, linewidth=1)
            ax.plot(centers, grand_results["error_stack"][i, :], color="tab:orange",
                    alpha=0.25, linewidth=1)

    ax.plot(centers, grand_results["grand_correct_mean"], color="darkgreen", linewidth=2.5,
            label=f"Correct trials, grand avg (n={n_sessions} sessions)")
    ax.fill_between(centers,
                     grand_results["grand_correct_mean"] - grand_results["grand_correct_sem"],
                     grand_results["grand_correct_mean"] + grand_results["grand_correct_sem"],
                     color="darkgreen", alpha=0.2)

    ax.plot(centers, grand_results["grand_error_mean"], color="darkorange", linewidth=2.5,
            label=f"Error trials, grand avg (n={n_sessions} sessions)")
    ax.fill_between(centers,
                     grand_results["grand_error_mean"] - grand_results["grand_error_sem"],
                     grand_results["grand_error_mean"] + grand_results["grand_error_sem"],
                     color="darkorange", alpha=0.2)

    ax.axhline(chance_level, color="black", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Grand average across {n_sessions} sessions: correct vs. error trial decoding accuracy")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    return fig, ax


def grand_average_metric(correct_list, error_list, session_names, centers_ref, metric_name="metric"):
    """Generic version of grand_average_correct_and_error -- works for accuracy OR score."""
    for arr in correct_list + error_list:
        if len(arr) != len(centers_ref):
            raise ValueError("Mismatched window count across sessions.")
    correct_stack = np.vstack(correct_list)
    error_stack = np.vstack(error_list)
    return {
        "centers": centers_ref, "metric_name": metric_name,
        "grand_correct_mean": correct_stack.mean(axis=0),
        "grand_correct_sem": correct_stack.std(axis=0, ddof=1) / np.sqrt(correct_stack.shape[0]),
        "grand_error_mean": error_stack.mean(axis=0),
        "grand_error_sem": error_stack.std(axis=0, ddof=1) / np.sqrt(error_stack.shape[0]),
        "correct_stack": correct_stack, "error_stack": error_stack,
        "session_names": session_names, "n_sessions": correct_stack.shape[0],
    }


def plot_grand_average_metric(grand_results, ylabel="Value", title=None,
                               ref_line=None, ref_label=None, show_individual=True, ylim=None):
    """Generic plot -- pass ref_line=0.5 for accuracy, ref_line=0 for score."""
    centers = grand_results["centers"]
    n_sessions = grand_results["n_sessions"]
    fig, ax = plt.subplots(figsize=(9, 5))

    if show_individual:
        for i in range(n_sessions):
            ax.plot(centers, grand_results["correct_stack"][i, :], color="tab:green", alpha=0.25, linewidth=1)
            ax.plot(centers, grand_results["error_stack"][i, :], color="tab:orange", alpha=0.25, linewidth=1)

    ax.plot(centers, grand_results["grand_correct_mean"], color="darkgreen", linewidth=2.5,
            label=f"Correct trials, grand avg (n={n_sessions} sessions)")
    ax.fill_between(centers,
                     grand_results["grand_correct_mean"] - grand_results["grand_correct_sem"],
                     grand_results["grand_correct_mean"] + grand_results["grand_correct_sem"],
                     color="darkgreen", alpha=0.2)
    ax.plot(centers, grand_results["grand_error_mean"], color="darkorange", linewidth=2.5,
            label=f"Error trials, grand avg (n={n_sessions} sessions)")
    ax.fill_between(centers,
                     grand_results["grand_error_mean"] - grand_results["grand_error_sem"],
                     grand_results["grand_error_mean"] + grand_results["grand_error_sem"],
                     color="darkorange", alpha=0.2)

    if ref_line is not None:
        ax.axhline(ref_line, color="black", linestyle="--", linewidth=1, label=ref_label or f"Reference ({ref_line})")

    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title or f"Grand average across {n_sessions} sessions")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    return fig, ax


def plot_area_comparison(grand_results_by_area, ylabel="Value", title=None,
                          ref_line=None, ref_label=None, ylim=None, colors=None):
    """
    Overlay grand-average correct/error curves from multiple areas on one figure.
    Solid line = correct trials, dashed line = error trials, one color per area.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    if colors is None:
        default_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = {area: default_cycle[i % len(default_cycle)]
                  for i, area in enumerate(grand_results_by_area.keys())}

    for area, gr in grand_results_by_area.items():
        centers = gr["centers"]
        n_sessions = gr["n_sessions"]
        color = colors.get(area)

        ax.plot(centers, gr["grand_correct_mean"], color=color, linestyle='-', linewidth=2.5,
                label=f"{area} correct (n={n_sessions})")
        ax.fill_between(centers,
                         gr["grand_correct_mean"] - gr["grand_correct_sem"],
                         gr["grand_correct_mean"] + gr["grand_correct_sem"],
                         color=color, alpha=0.15)

        ax.plot(centers, gr["grand_error_mean"], color=color, linestyle='--', linewidth=2.5,
                label=f"{area} error (n={n_sessions})")
        ax.fill_between(centers,
                         gr["grand_error_mean"] - gr["grand_error_sem"],
                         gr["grand_error_mean"] + gr["grand_error_sem"],
                         color=color, alpha=0.15)

    if ref_line is not None:
        ax.axhline(ref_line, color="black", linestyle=":", linewidth=1, label=ref_label or f"Reference ({ref_line})")

    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title or "Area comparison")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    return fig, ax


# =====================================================================
# 8. GROUP-LEVEL SIGNIFICANCE TESTING
# =====================================================================

def run_group_significance_tests(correct_list, error_list, centers, alpha=0.05, null_value=0.5):
    """
    Group-level significance testing across sessions, per timepoint. Runs
    three paired Wilcoxon signed-rank tests at each window (correct vs error,
    correct vs null_value, error vs null_value) -- matches your session-paired
    design (same 16 sessions contribute both a correct and an error value).

    NO multiple-comparisons correction is applied: the raw p-value at each
    timepoint is compared directly against `alpha`. (An FDR-corrected version
    was tried first and produced essentially the same significant stretches
    as this simpler uncorrected version, so FDR was dropped in favor of this.)

    NOTE: without correction, testing many timepoints (e.g. 95) means you
    should expect roughly alpha*n_timepoints "significant" hits by chance
    alone even with NO real effect (e.g. ~5 out of 95 at alpha=0.05) --
    isolated, non-contiguous significant frames deserve real caution; a
    sustained, contiguous run of significant frames is much stronger evidence
    than any single isolated dot.

    Parameters
    ----------
    correct_list, error_list : list of 1D arrays
        One curve per session (accuracy OR score), same length (n_windows),
        aligned to `centers`.
    centers : 1D array
        Window center timepoints, same length as each curve.
    alpha : float
        Significance threshold, applied directly to the raw p-value.
    null_value : float
        Reference value for the one-sample tests (0.5 for accuracy, 0.0 for score).

    Returns
    -------
    dict with, for each comparison ('correct_vs_error', 'correct_vs_chance', 'error_vs_chance'):
        'p_raw', 'sig_mask' (bool array, same length as centers).
    """
    correct_arr = np.vstack(correct_list)
    error_arr = np.vstack(error_list)
    n_windows = correct_arr.shape[1]
    assert n_windows == len(centers), "curves and centers length mismatch"

    def _per_window_wilcoxon(a, b=None):
        p_raw = np.full(n_windows, np.nan)
        for w in range(n_windows):
            try:
                if b is None:
                    stat, p = wilcoxon(a[:, w])
                else:
                    stat, p = wilcoxon(a[:, w], b[:, w])
                p_raw[w] = p
            except ValueError:
                p_raw[w] = np.nan
        return p_raw

    results = {}

    p_raw = _per_window_wilcoxon(correct_arr, error_arr)
    results['correct_vs_error'] = {'p_raw': p_raw, 'sig_mask': p_raw < alpha}

    p_raw = _per_window_wilcoxon(correct_arr - null_value)
    results['correct_vs_chance'] = {'p_raw': p_raw, 'sig_mask': p_raw < alpha}

    p_raw = _per_window_wilcoxon(error_arr - null_value)
    results['error_vs_chance'] = {'p_raw': p_raw, 'sig_mask': p_raw < alpha}

    results['centers'] = centers
    return results


def run_group_significance_tests_ranksum(correct_list, error_list, centers, alpha=0.005):
    """
    Per-timepoint Wilcoxon RANK-SUM test (scipy.stats.mannwhitneyu) -- the
    UNPAIRED version, matching the exact test used in Ayzenshtat et al. (2012)
    for their correct-vs-error comparisons (e.g. Fig 6B, Fig 7E). NO multiple-
    comparisons correction is applied, matching their approach directly (they
    tested a small number of pre-chosen windows, not a full per-frame sweep,
    so FDR wasn't part of their method).

    IMPORTANT CAVEAT: rank-sum assumes the two groups are INDEPENDENT samples.
    Your correct and error values at a given frame come from the SAME 16
    sessions -- they are naturally paired, and rank-sum discards that pairing
    information. This is included for direct comparability with the paper's
    reported method, not as a replacement for the paired signed-rank test in
    run_group_significance_tests, which better matches your actual design.

    Parameters
    ----------
    correct_list, error_list : list of 1D arrays
        One curve per session (accuracy OR score), aligned to `centers`.
    centers : 1D array
        Window center timepoints.
    alpha : float
        Significance threshold, applied directly to the RAW p-value (no FDR
        correction) -- default 0.005, matching the paper's reported threshold.

    Returns
    -------
    dict with 'centers', 'p_raw' (1D array, one per timepoint), 'sig_mask'
    (p_raw < alpha, no correction).
    """
    correct_arr = np.vstack(correct_list)
    error_arr = np.vstack(error_list)
    n_windows = correct_arr.shape[1]
    assert n_windows == len(centers), "curves and centers length mismatch"

    p_raw = np.full(n_windows, np.nan)
    for w in range(n_windows):
        try:
            stat, p = stats.mannwhitneyu(correct_arr[:, w], error_arr[:, w], alternative="two-sided")
            p_raw[w] = p
        except ValueError:
            # e.g. all values identical in both groups
            p_raw[w] = np.nan

    sig_mask = p_raw < alpha

    return {"centers": centers, "p_raw": p_raw, "sig_mask": sig_mask, "alpha": alpha}


def add_uncorrected_significance_markers(ax, test_results, y_offset=0.02, color="purple", label=None):
    """
    Overlay significance markers below the main curves on `ax`, for any
    single-comparison UNCORRECTED test result -- works with the output of
    either run_group_significance_tests_ranksum or
    run_group_significance_tests_paired_uncorrected (both return the same
    'centers'/'sig_mask'/'alpha' shape). Must be called BEFORE plt.show().
    """
    centers = np.asarray(test_results['centers'])
    ylim = ax.get_ylim()
    y0 = ylim[0]
    mask = np.asarray(test_results['sig_mask'])

    y = y0 - y_offset * (ylim[1] - ylim[0])
    alpha_val = test_results.get('alpha', 0.05)
    lbl = label or f"correct_vs_error (p<{alpha_val}, uncorrected)"
    ax.scatter(centers[mask], np.full(mask.sum(), y), marker='o', color=color, s=40, label=lbl)

    new_ylim = (y0 - 2 * y_offset * (ylim[1] - ylim[0]), ylim[1])
    ax.set_ylim(new_ylim)
    ax.legend(loc='upper right', fontsize=8)


# kept as an alias for backward compatibility with earlier snippets
add_ranksum_significance_markers = add_uncorrected_significance_markers



def add_significance_markers(ax, sig_results, comparisons=('correct_vs_error',), y_offset=0.02, colors=None, alpha=0.05):
    """
    Overlay significance markers below the main curves on `ax`.
    Must be called BEFORE plt.show() -- add all markers to the figure first,
    then call plt.show() exactly once at the very end of the script.

    Parameters
    ----------
    alpha : float
        The threshold used to build sig_results (just for the legend label --
        pass the same alpha you used in run_group_significance_tests so the
        label is accurate; sig_results itself already encodes the actual mask).
    """
    centers = np.asarray(sig_results['centers'])
    ylim = ax.get_ylim()
    y0 = ylim[0]
    if colors is None:
        colors = {'correct_vs_error': 'k', 'correct_vs_chance': 'C0', 'error_vs_chance': 'C1'}

    for i, comp in enumerate(comparisons):
        mask = np.asarray(sig_results[comp]['sig_mask'])
        y = y0 - (i + 1) * y_offset * (ylim[1] - ylim[0])
        ax.scatter(centers[mask], np.full(mask.sum(), y), marker='o',
                   color=colors.get(comp, 'k'), s=40, label=f'{comp} (p<{alpha}, uncorrected)')

    new_ylim = (y0 - (len(comparisons) + 1) * y_offset * (ylim[1] - ylim[0]), ylim[1])
    ax.set_ylim(new_ylim)
    ax.legend(loc='upper right', fontsize=8)


def test_window_average(correct_list, error_list, centers, window, null_value=None, label="metric",
                         plot=True, session_names=None, area_label=None, ylabel="Value"):
    """
    Collapse each session's curve to ONE mean value within `window`, then run a
    single paired Wilcoxon test. If plot=True (default), also builds the
    per-session paired dot plot + diff bar plot via plot_window_average_comparison.
    """
    centers = np.asarray(centers)
    mask = (centers >= window[0]) & (centers <= window[1])
    if mask.sum() == 0:
        raise ValueError(f"No frames in window {window}.")

    correct_arr = np.vstack(correct_list)
    error_arr = np.vstack(error_list)
    correct_win_mean = correct_arr[:, mask].mean(axis=1)
    error_win_mean = error_arr[:, mask].mean(axis=1)

    diff = correct_win_mean - error_win_mean
    stat, p = wilcoxon(correct_win_mean, error_win_mean)
    cohens_d = diff.mean() / diff.std(ddof=1)

    print(f"--- Window test ({label}): frames {window[0]}-{window[1]} ({mask.sum()} frames) ---")
    print(f"Correct: {correct_win_mean.mean():.4f} +/- {correct_win_mean.std(ddof=1):.4f}")
    print(f"Error:   {error_win_mean.mean():.4f} +/- {error_win_mean.std(ddof=1):.4f}")
    print(f"Diff (correct-error): {diff.mean():.4f}, Cohen's d={cohens_d:.3f}, Wilcoxon p={p:.4f}")

    result = {"window": window, "correct_win_mean": correct_win_mean, "error_win_mean": error_win_mean,
              "diff": diff, "p_value": p, "cohens_d": cohens_d}

    if null_value is not None:
        _, p_c = wilcoxon(correct_win_mean - null_value)
        _, p_e = wilcoxon(error_win_mean - null_value)
        print(f"Correct vs {null_value}: p={p_c:.4f} | Error vs {null_value}: p={p_e:.4f}")
        result["correct_vs_null_p"] = p_c
        result["error_vs_null_p"] = p_e

    if plot:
        fig, (ax_left, ax_right) = plot_window_average_comparison(
            result, session_names=session_names, area_label=area_label, ylabel=ylabel)
        result["fig"] = fig
        result["ax"] = (ax_left, ax_right)

    return result


def test_window_average_permutation(correct_list, error_list, centers, window,
                                     n_permutations=10000, seed=42, label="metric"):
    """
    Sign-flip permutation test on the window-averaged correct-vs-error difference.
    Alternative to the Wilcoxon version in test_window_average -- avoids the loss
    of power Wilcoxon suffers from tied values (common with small per-session
    trial counts), and doesn't rely on any asymptotic approximation since the
    null distribution is built directly from your actual per-session differences.
    """
    rng = np.random.default_rng(seed)
    centers = np.asarray(centers)
    mask = (centers >= window[0]) & (centers <= window[1])
    if mask.sum() == 0:
        raise ValueError(f"No frames in window {window}.")

    correct_arr = np.vstack(correct_list)
    error_arr = np.vstack(error_list)
    correct_win_mean = correct_arr[:, mask].mean(axis=1)
    error_win_mean = error_arr[:, mask].mean(axis=1)

    diffs = correct_win_mean - error_win_mean
    n_sessions = len(diffs)
    real_mean_diff = diffs.mean()

    signs = rng.choice([-1, 1], size=(n_permutations, n_sessions))
    null_means = (signs * diffs[None, :]).mean(axis=1)

    p_value = (np.abs(null_means) >= np.abs(real_mean_diff)).mean()

    cohens_d = diffs.mean() / diffs.std(ddof=1)
    n_positive = (diffs > 0).sum()

    print(f"--- Permutation window test ({label}): frames {window[0]}-{window[1]} "
          f"({mask.sum()} frames), {n_permutations} permutations ---")
    print(f"Real mean diff (correct-error): {real_mean_diff:.4f}, Cohen's d={cohens_d:.3f}")
    print(f"Sessions with correct > error: {n_positive}/{n_sessions}")
    print(f"Permutation p-value: {p_value:.4f}")

    return {
        "window": window, "diffs": diffs, "real_mean_diff": real_mean_diff,
        "cohens_d": cohens_d, "n_positive": n_positive, "n_sessions": n_sessions,
        "p_value": p_value, "null_means": null_means,
    }


def test_window_cohens_d_pooled(correct_list, error_list, centers, window,
                                 n_permutations=10000, seed=42, label="metric"):
    """
    Pooled (classic, unpaired) Cohen's d for a fixed window:
        d = (mean(correct) - mean(error)) / std(correct and error pooled)
    Paired with a sign-flip permutation test for a p-value.
    """
    rng = np.random.default_rng(seed)
    centers = np.asarray(centers)
    mask = (centers >= window[0]) & (centers <= window[1])
    if mask.sum() == 0:
        raise ValueError(f"No frames in window {window}.")

    correct_arr = np.vstack(correct_list)
    error_arr = np.vstack(error_list)
    correct_win_mean = correct_arr[:, mask].mean(axis=1)
    error_win_mean = error_arr[:, mask].mean(axis=1)
    n_sessions = len(correct_win_mean)

    pooled = np.concatenate([correct_win_mean, error_win_mean])
    pooled_std = pooled.std(ddof=1)
    real_d = (correct_win_mean.mean() - error_win_mean.mean()) / pooled_std

    null_d = np.zeros(n_permutations)
    paired = np.stack([correct_win_mean, error_win_mean], axis=1)
    for p in range(n_permutations):
        swap = rng.integers(0, 2, size=n_sessions).astype(bool)
        perm_correct = np.where(swap, paired[:, 1], paired[:, 0])
        perm_error = np.where(swap, paired[:, 0], paired[:, 1])
        d = (perm_correct.mean() - perm_error.mean()) / pooled_std
        null_d[p] = d

    p_value = (np.abs(null_d) >= np.abs(real_d)).mean()

    print(f"--- Pooled Cohen's d test ({label}): frames {window[0]}-{window[1]} "
          f"({mask.sum()} frames) ---")
    print(f"Correct mean: {correct_win_mean.mean():.4f}, Error mean: {error_win_mean.mean():.4f}")
    print(f"Pooled SD (all {2*n_sessions} values): {pooled_std:.4f}")
    print(f"Pooled Cohen's d: {real_d:.3f}")
    print(f"Permutation p-value: {p_value:.4f}")

    return {
        "window": window, "correct_win_mean": correct_win_mean, "error_win_mean": error_win_mean,
        "pooled_std": pooled_std, "pooled_d": real_d, "p_value": p_value, "null_d": null_d,
    }


def plot_window_average_comparison(window_result, session_names=None, area_label=None, ylabel="Value"):
    """
    Visualize the per-session values behind a test_window_average result:
    left panel -- paired dot plot, right panel -- per-session diff (sorted).
    """
    correct_vals = window_result["correct_win_mean"]
    error_vals = window_result["error_win_mean"]
    diff = window_result["diff"]
    window = window_result["window"]
    n_sessions = len(correct_vals)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    for i in range(n_sessions):
        color = "tab:green" if diff[i] > 0 else "tab:red"
        ax_left.plot([0, 1], [correct_vals[i], error_vals[i]], color=color, alpha=0.5, linewidth=1)
    ax_left.scatter(np.zeros(n_sessions), correct_vals, color="darkgreen", zorder=5, label="Correct")
    ax_left.scatter(np.ones(n_sessions), error_vals, color="darkorange", zorder=5, label="Error")
    ax_left.set_xticks([0, 1])
    ax_left.set_xticklabels(["Correct", "Error"])
    ax_left.set_ylabel(ylabel)
    ax_left.set_title(f"Per-session window means, frames {window[0]}-{window[1]}")
    ax_left.legend(loc="best")

    sort_idx = np.argsort(diff)
    diff_sorted = diff[sort_idx]
    colors = ["tab:green" if d > 0 else "tab:red" for d in diff_sorted]
    x_pos = np.arange(n_sessions)
    ax_right.bar(x_pos, diff_sorted, color=colors, alpha=0.7)
    ax_right.axhline(0, color="black", linewidth=1)
    if session_names is not None:
        labels_sorted = [session_names[i] for i in sort_idx]
        ax_right.set_xticks(x_pos)
        ax_right.set_xticklabels(labels_sorted, rotation=90, fontsize=7)
    else:
        ax_right.set_xlabel("Session (sorted by diff)")
    ax_right.set_ylabel("Diff (correct - error)")
    n_pos = (diff > 0).sum()
    ax_right.set_title(f"Per-session diff ({n_pos}/{n_sessions} positive)")

    title = f"Window comparison, frames {window[0]}-{window[1]}"
    if area_label:
        title += f" -- {area_label}"
    fig.suptitle(title)
    plt.tight_layout()
    return fig, (ax_left, ax_right)


def sliding_window_effect_size(correct_list, error_list, centers, window_width=16, method="paired"):
    """
    Sliding-window Cohen's d -- MAGNITUDE ONLY, no significance test. Slides a
    window_width-frame window across the whole time course; at each position,
    collapses each session to its window mean and computes an effect size.

    method='paired' (default): d = mean(diff) / std(diff) -- divides by
        session-to-session spread of the differences.
    method='pooled': d = (mean(correct)-mean(error)) / std(correct and error
        pooled) -- the "classic" unpaired Cohen's d.
    """
    if method not in ("paired", "pooled"):
        raise ValueError("method must be 'paired' or 'pooled'")

    correct_arr = np.vstack(correct_list)
    error_arr = np.vstack(error_list)
    n_windows_total = correct_arr.shape[1]
    half = window_width // 2

    valid_positions = np.arange(half, n_windows_total - half)
    window_centers = centers[valid_positions]
    n_positions = len(valid_positions)

    d_values = np.full(n_positions, np.nan)

    for k, pos in enumerate(valid_positions):
        lo, hi = pos - half, pos - half + window_width
        correct_win = correct_arr[:, lo:hi].mean(axis=1)
        error_win = error_arr[:, lo:hi].mean(axis=1)

        if method == "paired":
            diff = correct_win - error_win
            sd = diff.std(ddof=1)
            d_values[k] = diff.mean() / sd if sd > 0 else np.nan
        else:
            pooled = np.concatenate([correct_win, error_win])
            sd = pooled.std(ddof=1)
            d_values[k] = (correct_win.mean() - error_win.mean()) / sd if sd > 0 else np.nan

    return {"window_centers": window_centers, "window_width": window_width,
            "method": method, "d_values": d_values}


def plot_sliding_window_effect_size(sliding_d_results, area_label=None, title=None,
                                     color="tab:purple", show_reference_lines=True):
    """
    Plot the sliding-window Cohen's d curve over time. Magnitude only -- no
    significance shading.
    """
    window_centers = sliding_d_results["window_centers"]
    window_width = sliding_d_results["window_width"]
    method = sliding_d_results["method"]
    d_values = sliding_d_results["d_values"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(window_centers, d_values, color=color, linewidth=2)
    ax.axhline(0, color="black", linestyle="-", linewidth=1)

    if show_reference_lines:
        for val, lab in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
            ax.axhline(val, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
            ax.axhline(-val, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
            ax.text(window_centers[-1], val, f" {lab}", fontsize=7, va="center", color="gray")

    ax.set_xlabel("Frame (sliding window center)")
    ax.set_ylabel(f"Cohen's d ({method})")

    default_title = f"Sliding-window effect size (width={window_width} frames, {method})"
    if area_label:
        default_title += f" -- {area_label}"
    ax.set_title(title or default_title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, ax


def cluster_permutation_test(correct_list, error_list, centers, n_permutations=2000,
                              cluster_alpha=0.05, seed=42, label="metric"):
    """
    Cluster-based permutation test across the FULL time course (Maris & Oostenveld
    style). Finds contiguous runs of timepoints where correct vs. error diverge,
    without hand-picking a window, and tests whether the strongest such run is
    bigger than chance using session-level sign-flip permutation.
    """
    from scipy.stats import t as tdist

    rng = np.random.default_rng(seed)
    centers = np.asarray(centers)
    correct_arr = np.vstack(correct_list)
    error_arr = np.vstack(error_list)
    diff = correct_arr - error_arr
    n_sessions, n_windows = diff.shape

    def _paired_tstat(d):
        mean = d.mean(axis=0)
        sd = d.std(axis=0, ddof=1)
        se = sd / np.sqrt(d.shape[0])
        with np.errstate(divide='ignore', invalid='ignore'):
            t_vals = np.where(se > 0, mean / se, 0.0)
        return t_vals

    df = n_sessions - 1
    t_thresh = tdist.ppf(1 - cluster_alpha / 2, df)

    def _find_clusters(t_vals):
        clusters = []
        for mask in (t_vals > t_thresh, t_vals < -t_thresh):
            in_cluster = False
            start = None
            for i, m in enumerate(mask):
                if m and not in_cluster:
                    start, in_cluster = i, True
                if not m and in_cluster:
                    clusters.append((start, i - 1, np.sum(np.abs(t_vals[start:i]))))
                    in_cluster = False
            if in_cluster:
                clusters.append((start, len(mask) - 1, np.sum(np.abs(t_vals[start:len(mask)]))))
        return clusters

    t_obs = _paired_tstat(diff)
    obs_clusters = _find_clusters(t_obs)

    max_null_mass = np.zeros(n_permutations)
    for p in range(n_permutations):
        signs = rng.choice([-1, 1], size=n_sessions)
        diff_perm = diff * signs[:, None]
        t_perm = _paired_tstat(diff_perm)
        perm_clusters = _find_clusters(t_perm)
        max_null_mass[p] = max((c[2] for c in perm_clusters), default=0.0)

    results = []
    for start, end, mass in obs_clusters:
        p_val = (max_null_mass >= mass).mean()
        results.append({
            "start_frame": centers[start], "end_frame": centers[end],
            "n_frames": end - start + 1, "mass": mass, "p_value": p_val,
        })

    print(f"--- Cluster-based permutation test ({label}), {n_permutations} permutations, "
          f"cluster-forming t-threshold={t_thresh:.2f} ---")
    if not results:
        print("No candidate clusters found (nothing crossed the cluster-forming threshold).")
    else:
        for r in sorted(results, key=lambda x: x["p_value"]):
            print(f"  Frames {r['start_frame']}-{r['end_frame']} ({r['n_frames']} frames): "
                  f"mass={r['mass']:.2f}, p={r['p_value']:.4f}")

    return {"clusters": results, "null_max_mass": max_null_mass, "t_obs": t_obs,
            "centers": centers, "t_thresh": t_thresh}