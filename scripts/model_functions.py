import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


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

'''
def split_data(X, y, test_size=0.3):
    """
    Separates data into a Training Set (to learn) and a Test Set (to validate).
    """
    # stratify=y ensures we keep the exact same ratio of Face/Shuffle in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42, 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test



def svm_kfold_nested(
    X, y,
    n_splits=5,
    C_grid=(0.01, 0.1, 1),
    kernel="linear",
    seed=0
):
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    # Use the same CV strategy for both inner and outer loops
    cv_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # 1) Define the Inner CV (Hyperparameter Tuning)
    base_model = SVC(kernel=kernel)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid={"C": list(C_grid)},
        cv=cv_inner,
        scoring="accuracy",
        n_jobs=-1
    )

    # 2) Run the Outer CV (Performance Estimation)
    # This automatically runs grid_search.fit() on each training fold
    # and evaluates on each held-out fold.
    nested_scores = cross_val_score(grid_search, X, y, cv=cv_outer)

    # 3) Fit final model on ALL data to get the Best C for deployment
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_C = grid_search.best_params_["C"]

    results = {
        "best_C": float(best_C),
        "cv_fold_accuracy": nested_scores,
        "cv_mean_accuracy": float(nested_scores.mean()),
        "cv_std_accuracy": float(nested_scores.std(ddof=1)),
        "inner_cv_best_score": float(grid_search.best_score_),
    }

    return results, best_model
'''




def leave_one_pair_out_splits(y_trials, n_frames):
    """
    Yields (train_idx, test_idx) where test contains all frames of:
    - one trial from class 0
    - one trial from class 1
    Total splits: n0*n1 (30*30 = 900).
    """
    y_trials = np.asarray(y_trials).astype(int)
    trials0 = np.where(y_trials == 0)[0]
    trials1 = np.where(y_trials == 1)[0]

    n_trials = len(y_trials)
    all_idx = np.arange(n_trials * n_frames)

    for t0 in trials0:
        test0 = np.arange(t0 * n_frames, (t0 + 1) * n_frames)
        for t1 in trials1:
            test1 = np.arange(t1 * n_frames, (t1 + 1) * n_frames)
            test_idx = np.concatenate([test0, test1])

            mask = np.ones(all_idx.size, dtype=bool)
            mask[test_idx] = False
            train_idx = all_idx[mask]

            yield train_idx, test_idx




################ model functions ################

def _get_scores_for_auc(estimator, X_test):
    """Return continuous scores for AUC (best-effort)."""
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X_test))
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X_test))[:, 1]  # prob(class=1)
    return None


def _safe_auc(y_true, y_scores):
    """AUC is undefined if only one class exists in y_true."""
    if y_scores is None:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_scores))
    except ValueError:
        return np.nan



def make_linear_svm(C=0.001):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(C=C, dual=True, max_iter=10000))
    ])


def get_splits(splitter, X, y, groups_arr):
    if hasattr(splitter, "split"):
        if groups_arr is None:
            return list(splitter.split(X, y))
        return list(splitter.split(X, y, groups_arr))
    return list(splitter)  # already a generator/list of (train_idx, test_idx)



def check_full_coverage(splits, n_samples):
    counts = np.zeros(n_samples, dtype=int)
    for _, test_idx in splits:
        counts[np.asarray(test_idx)] += 1
    return np.all(counts == 1)



def extract_linear_weights(est):
    """Return (w_scaled, w_original_units) or (None, None) if not available."""
    try:
        scaler = est.named_steps.get("scaler", None)
        clf = est.named_steps.get("clf", None)
        if clf is None or not hasattr(clf, "coef_"):
            return None, None

        w_scaled = np.asarray(clf.coef_).ravel()

        if scaler is not None and hasattr(scaler, "scale_"):
            w = w_scaled / (np.asarray(scaler.scale_).ravel() + 1e-12)
        else:
            w = w_scaled.copy()

        return w_scaled, w
    except Exception:
        return None, None




def fit_eval_one_fold(fold_idx, X, y, train_idx, test_idx, make_estimator):
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)

    est = make_estimator()
    est.fit(X[train_idx], y[train_idx])

    y_pred = est.predict(X[test_idx])
    acc = float(accuracy_score(y[test_idx], y_pred))

    scores = _get_scores_for_auc(est, X[test_idx])
    auc = _safe_auc(y[test_idx], scores)

    w_scaled, w = extract_linear_weights(est)

    return {
        "fold": int(fold_idx),
        "test_idx": test_idx,
        "y_pred": y_pred,
        "scores": scores,
        "acc": acc,
        "auc": auc,
        "n_test": int(len(test_idx)),
        "w_scaled": w_scaled,
        "w": w
    }



def summarize_folds(fold_results):
    accs = np.array([f["acc"] for f in fold_results], dtype=float)
    aucs = np.array([f["auc"] for f in fold_results], dtype=float)

    out = {
        "folds": [{"fold": f["fold"], "acc": f["acc"], "auc": f["auc"], "n_test": f["n_test"]}
                for f in fold_results],
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs, ddof=1)) if np.sum(np.isfinite(aucs)) > 1 else 0.0,
    }

    W = [f["w"] for f in fold_results if f["w"] is not None]
    if len(W) > 0:
        W = np.stack(W, axis=0)
        out["w_mean"] = W.mean(axis=0)
        out["w_std"] = W.std(axis=0, ddof=1) if W.shape[0] > 1 else np.zeros(W.shape[1], dtype=float)
    else:
        out["w_mean"] = None
        out["w_std"] = None

    out["fold_weights"] = [{"fold": f["fold"], "w": f["w"], "w_scaled": f["w_scaled"]}
                        for f in fold_results]
    return out




def add_oof_outputs(results, fold_results, y):
    oof_pred = np.empty(y.shape[0], dtype=int)
    oof_scores = np.empty(y.shape[0], dtype=float)

    for f in fold_results:
        oof_pred[f["test_idx"]] = f["y_pred"]
        if f["scores"] is None:
            oof_scores[f["test_idx"]] = np.nan
        else:
            oof_scores[f["test_idx"]] = np.asarray(f["scores"], dtype=float)

    results["oof_pred"] = oof_pred
    results["oof_scores"] = oof_scores
    results["confusion_matrix"] = confusion_matrix(y, oof_pred)
    return results

def run_cv(X, y, groups=None,
        splitter=None,
        make_estimator=None,
        expect_full_coverage=True,
        n_jobs=1,
        verbose=True,
        plot_splits=False,
        split_title="CV splits"):
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    groups_arr = None if groups is None else np.asarray(groups)

    if splitter is None:
        raise ValueError("splitter is required.")

    if make_estimator is None:
        make_estimator = lambda: make_linear_svm(C=0.001)

    splits = get_splits(splitter, X, y, groups_arr)

    if expect_full_coverage and not check_full_coverage(splits, y.shape[0]):
        raise RuntimeError(
            "Not full-coverage: some samples are tested 0 or >1 times. "
            "Set expect_full_coverage=False for this splitter."
        )

    if n_jobs is None or n_jobs == 1:
        fold_results = [fit_eval_one_fold(i, X, y, tr, te, make_estimator)
                        for i, (tr, te) in enumerate(splits)]
    else:
        fold_results = Parallel(n_jobs=n_jobs)(
            delayed(fit_eval_one_fold)(i, X, y, tr, te, make_estimator)
            for i, (tr, te) in enumerate(splits)
        )
    if verbose:
        for f in fold_results:
            auc_txt = f"{f['auc']:.4f}" if np.isfinite(f["auc"]) else "nan"
            print(f"Fold {f['fold']}: acc={f['acc']:.4f}, AUC={auc_txt}, n_test={f['n_test']}")

    results = summarize_folds(fold_results)
    if expect_full_coverage:
        results = add_oof_outputs(results, fold_results, y)
    return results




def select_best_C(X, y, splitter, C_grid=np.logspace(-4, 2, 7), groups=None, metric="acc",# "acc" or "auc"
                tie_break="smaller_C",        # "smaller_C" or "larger_C"
                expect_full_coverage=False,   # inner CV is often not full-coverage (e.g., repeated CV)
                n_jobs=1,
                verbose=True):
    """
    Select C by running run_cv for each C in C_grid and choosing the best mean score.

    Returns:
        best_C: float
        table: list of dicts with per-C results
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    groups_arr = None if groups is None else np.asarray(groups)

    if C_grid is None:
        C_grid = np.logspace(-4, 2, 7)
    rows = []
    for C in C_grid:
        res = run_cv(
            X, y,
            groups=groups_arr,
            splitter=splitter,
            make_estimator=lambda C=C: make_linear_svm(C),
            expect_full_coverage=expect_full_coverage,
            n_jobs=n_jobs,
            verbose=False
        )
        row = {
            "C": float(C),
            "acc_mean": float(res["acc_mean"]),
            "acc_std": float(res["acc_std"]),
            "auc_mean": float(res["auc_mean"]),
            "auc_std": float(res["auc_std"]),
            "n_folds": int(len(res["folds"]))
        }
        rows.append(row)
    # choose score
    key = "acc_mean" if metric == "acc" else "auc_mean"
    scores = np.array([r[key] for r in rows], dtype=float)

    best_idx = int(np.nanargmax(scores))
    best_score = scores[best_idx]

    # tie-break within tolerance (SEM-based)
    # candidates whose mean is within 1 SEM of the best
    std_key = "acc_std" if metric == "acc" else "auc_std"
    n_folds = np.array([r["n_folds"] for r in rows], dtype=float)
    sem = np.array([r[std_key] / np.sqrt(max(n-1, 1)) for r, n in zip(rows, n_folds)], dtype=float)
    tol = float(np.nanmin(sem)) if np.isfinite(np.nanmin(sem)) else 0.0

    candidates = [i for i, s in enumerate(scores) if np.isfinite(s) and s >= best_score - tol]
    if tie_break == "smaller_C":
        best_idx = min(candidates, key=lambda i: rows[i]["C"])
    elif tie_break == "larger_C":
        best_idx = max(candidates, key=lambda i: rows[i]["C"])
    else:
        best_idx = candidates[0]
    best_C = rows[best_idx]["C"]
    if verbose:
        print(f"Selected C={best_C} using metric={metric} (tie_break={tie_break}).")
        for r in rows:
            print(f"C={r['C']:.4g} | acc={r['acc_mean']:.4f}±{r['acc_std']:.4f} | "
                f"auc={r['auc_mean']:.4f}±{r['auc_std']:.4f}")
    return best_C, rows


