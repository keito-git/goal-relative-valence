"""Validity of the SST-2 diff-of-means direction at every layer: split-half AUC on held-out SST-2 sentences."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
R = Path(os.environ.get("GOALVAL_ROOT", "."))
out = {}
for tag in sys.argv[1:]:
    A = R / "acts" / tag
    X = np.load(A / "sst2_raw.npy", mmap_mode="r"); y = np.load(A / "sst2_labels.npy")
    rng = np.random.default_rng(20261006); idx = rng.permutation(len(y)); a, b = idx[: len(y) // 2], idx[len(y) // 2:]
    aucs = []
    for L in range(X.shape[1]):
        Z = np.asarray(X[:, L], dtype=np.float32)
        w = Z[a][y[a] == 1].mean(0) - Z[a][y[a] == 0].mean(0)
        s = (Z[b] - Z[a].mean(0)) @ w
        aucs.append(float(roc_auc_score(y[b], s)) if np.isfinite(s).all() and s.std() > 0 else float("nan"))
    out[tag] = aucs
    n = len(aucs) - 1
    print(f"{tag:16s} mid(0.25-0.5)={np.nanmean(aucs[n//4:n//2]):.3f} late(0.75-1)={np.nanmean(aucs[3*n//4:]):.3f} final={aucs[-1]:.3f}")
json.dump(out, open(R / "results" / "sst2_splithalf_auc.json", "w"))
