import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def flatten_data(data_3d):
    """
    Reshapes 3D brain data (Pixels, Frames, Trials) into 2D (Trials, Features).
    """
    # 1. Ensure 'Trials' is the first dimension
    # If input is (Pixels, Frames, Trials), move Trials to the front
    if data_3d.shape[-1] < data_3d.shape[0]: 
        data_3d = np.moveaxis(data_3d, -1, 0) 
    n_trials, n_pixels, n_frames = data_3d.shape
    # 2. Flatten Space and Time into one long vector per trial
    # Result shape: (60, 5000)
    X_flat = data_3d.reshape(n_trials, n_pixels * n_frames)
    return X_flat


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




def svm_kfold(
    X, y,
    n_splits=5,
    C_grid=(0.01, 0.1, 1),
    kernel="linear",
    seed=0
):
    """
    SVM with regularization (C) + Stratified K-Fold CV.
    Assumes X is ALREADY z-scored / standardized (so no StandardScaler here).

    Inputs:
        X: (n_trials, n_features)
        y: (n_trials,)
    Returns:
        results: dict with fold accuracies, mean/std, best C
        best_model: fitted SVC on ALL data using best C
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # 1) Pick best C using CV (inner selection)
    base_model = SVC(kernel=kernel)
    grid = GridSearchCV(
        estimator=base_model,
        param_grid={"C": list(C_grid)},
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X, y)
    best_C = grid.best_params_["C"]

    # 2) Estimate performance with CV using that best C
    fold_acc = []
    for train_idx, test_idx in cv.split(X, y):
        model = SVC(kernel=kernel, C=best_C)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        fold_acc.append(accuracy_score(y[test_idx], y_pred))

    fold_acc = np.array(fold_acc, dtype=float)

    # 3) Fit final model on all data
    best_model = SVC(kernel=kernel, C=best_C)
    best_model.fit(X, y)

    results = {
        "best_C": float(best_C),
        "cv_fold_accuracy": fold_acc,
        "cv_mean_accuracy": float(fold_acc.mean()),
        "cv_std_accuracy": float(fold_acc.std(ddof=1)) if len(fold_acc) > 1 else 0.0,
        "grid_best_score_innerCV": float(grid.best_score_),
    }

    return results, best_model
