from pathlib import Path
import sys
import scipy.io as sio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_mat"

mat_name = sys.argv[1]
path = RAW / mat_name

print("Opening:", path)
md = sio.loadmat(path)

print("\nAll variables in the file:")
for k in sorted(md.keys()):
    if k.startswith("__"):
        continue
    v = md[k]
    shape = getattr(v, "shape", None)
    print(f"{k:25s} shape={shape}")

print("\nVariables that look like condition matrices (10000 x 256 x N):")
for k in sorted(md.keys()):
    if k.startswith("__"):
        continue
    v = md[k]
    if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[0] == 10000 and v.shape[1] == 256:
        print(f"{k:25s} shape={v.shape}")
