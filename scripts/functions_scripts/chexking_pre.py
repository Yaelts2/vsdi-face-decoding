import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from sklearn.svm import LinearSVC

# Use your actual data - shape is (pixels, frames, trials)
X_face = np.load("data/processed/condsXn/condsXn1_110209a.npy")  # (10000, 256, 28)
X_non  = np.load("data/processed/condsXn/condsXn5_110209a.npy")  # (10000, 256, 28)

# Average across frames to get one vector per trial
X_face_avg = X_face.mean(axis=1).T  # (28, 10000)
X_non_avg  = X_non.mean(axis=1).T   # (28, 10000)

X = np.concatenate([X_face_avg, X_non_avg], axis=0)  # (56, 10000)
y = np.array([1]*28 + [0]*28)  # (56,)

print("X shape:", X.shape)
print("y shape:", y.shape)

clf = LinearSVC(C=0.0001, dual=False, max_iter=100000, random_state=42)
clf.fit(X, y)

print("coef sum:", clf.coef_.sum())
print("coef mean:", clf.coef_.mean())
print("coef[0:5]:", clf.coef_[0, :5])