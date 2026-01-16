# model_evaluation.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def plot_accuracy_kfold_bars(cv_fold_acc, chance=0.5, title="Decoding accuracy (CV)"):
    """
    Bar plot of each fold accuracy + mean ± SEM.
    - Shows individual folds as bars (not one big bar)
    - Adds chance line
    - Uses a tighter y-range (0.4–1) for readability
    - Annotates mean value
    """
    acc = np.asarray(cv_fold_acc, dtype=float)
    k = acc.size
    mean = acc.mean()
    sem = acc.std(ddof=1) / np.sqrt(k) if k > 1 else 0.0

    x = np.arange(1, k + 1)

    plt.figure()
    plt.bar(x, acc)
    plt.axhline(chance, linestyle="--")
    plt.axhline(mean, linestyle=":")
    plt.errorbar([k + 0.8], [mean], yerr=[sem], fmt="o", capsize=6)

    # mean annotation
    plt.text(k + 0.8, mean + 0.02, f"mean={mean:.2f}", ha="center", va="bottom")

    plt.xticks(list(x) + [k + 0.8], [f"Fold {i}" for i in x] + ["Mean"])
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0.4, 1.0)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred,
                          class_names=("Non-face", "Face"),
                          title="Confusion matrix (normalized)"):
    """
    Clean normalized confusion matrix.
    - Normalized per true class
    - Fixed color scale (0–1) so it stays consistent across runs
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    cm = cm / cm.sum(axis=1, keepdims=True)

    plt.figure()
    plt.imshow(cm, vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_scores, title="ROC Curve"):
    """
    Plot ROC curve given true labels and predicted scores.
    """
    from sklearn.metrics import roc_curve, auc

    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.show()
    
def permutation_test_linear_svm_fast(
    X, y,
    n_perm=300,
    n_splits=5,
    C=0.01,
    seed=0,
    early_stop_alpha=0.05,     # set None to disable early stop
    min_perm_before_stop=100
):
    """
    Faster permutation test:
    - uses LinearSVC (faster than SVC(kernel='linear'))
    - precomputes CV splits once
    - optional early stopping

    Returns:
        shuffled_acc: (n_perm,) mean CV accuracy per permutation
        p_value: one-sided p-value for real_acc (computed outside or inside if provided)
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(X, y))  # reuse folds

    shuffled_acc = []
    for p in range(n_perm):
        y_shuff = rng.permutation(y)

        fold_acc = []
        for train_idx, test_idx in splits:
            clf = LinearSVC(C=C, dual=True, max_iter=5000)  # dual=True works well for high-D
            clf.fit(X[train_idx], y_shuff[train_idx])
            y_pred = clf.predict(X[test_idx])
            fold_acc.append(accuracy_score(y_shuff[test_idx], y_pred))

        shuffled_acc.append(float(np.mean(fold_acc)))

    return np.array(shuffled_acc, dtype=float)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def plot_under_overfit_curve_svm(
    X, y,
    C_values=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 100),
    n_splits=5,
    seed=0,
    title="Underfitting vs Overfitting (SVM)"
):
    """
    Plots:
      - Training error (1-acc) vs model complexity log10(C)
      - CV error (1-acc) vs model complexity log10(C)  [proxy for test error]
    Marks the best fit (min CV error).
    Assumes X already z-scored.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    train_err = []
    cv_err = []

    for C in C_values:
        # Training error
        clf = SVC(kernel="linear", C=C)
        clf.fit(X, y)
        y_pred_train = clf.predict(X)
        train_acc = accuracy_score(y, y_pred_train)
        train_err.append(1.0 - train_acc)

        # CV error (proxy for test error)
        fold_acc = []
        for tr, te in cv.split(X, y):
            clf_cv = SVC(kernel="linear", C=C)
            clf_cv.fit(X[tr], y[tr])
            y_pred = clf_cv.predict(X[te])
            fold_acc.append(accuracy_score(y[te], y_pred))
        cv_err.append(1.0 - float(np.mean(fold_acc)))

    train_err = np.array(train_err, dtype=float)
    cv_err = np.array(cv_err, dtype=float)

    # Complexity axis
    x = np.log10(np.array(C_values, dtype=float))

    # Best fit = minimum CV error
    best_idx = int(np.argmin(cv_err))
    best_x = x[best_idx]
    best_y = cv_err[best_idx]

    # ---- Plot ----
    plt.figure()
    plt.plot(x, cv_err, label="Test Error (CV)")
    plt.plot(x, train_err, label="Training Error")

    # Vertical line at best fit
    plt.axvline(best_x, linestyle="--")
    plt.text(best_x, best_y, "  Best Fit", va="bottom")

    plt.xlabel("Model complexity  (log10(C))")
    plt.ylabel("Error  (1 - accuracy)")
    plt.title(title)
    plt.ylim(0, 1)

    # Under/overfitting labels
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    plt.text(xmin, ymax * 0.95, "Underfitting  \u2190", ha="left", va="top")
    plt.text(xmax, ymax * 0.95, "\u2192  Overfitting", ha="right", va="top")

    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "C_values": C_values,
        "x_log10C": x,
        "train_error": train_err,
        "cv_error": cv_err,
        "best_C": float(C_values[best_idx]),
        "best_cv_error": float(best_y),
    }
