import numpy as np
from sklearn.model_selection import GroupKFold

groups = np.repeat(np.arange(56), 5)
y = np.array([1]*140 + [0]*140)
X = np.zeros((280, 10))

gkf = GroupKFold(n_splits=5)

# get the actual split indices
splits = list(gkf.split(X, y, groups))
for i, (tr, te) in enumerate(splits):
    print(f"Fold {i}: te[:5]={te[:5]}, tr[:5]={tr[:5]}")
    print(f"  te sum={te.sum()}, tr sum={tr.sum()}")