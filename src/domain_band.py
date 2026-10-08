"""Post hoc: late-minus-mid conflict-cell AUC per confirmation domain (SST-2 diff-of-means, self)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
R = Path(os.environ.get("GOALVAL_ROOT", ".")); rows = [json.loads(l) for l in open(R / "data/goal_stimuli.jsonl")]
pos = sys.argv[1]; out = {}
doms = ["lottery", "race", "court", "hiring"]
for tag in sys.argv[2:]:
    A = R / "acts" / tag
    G = np.load(A / f"goal_{pos}.npy", mmap_mode="r"); S = np.load(A / f"sst2_{pos}.npy", mmap_mode="r"); ys = np.load(A / "sst2_labels.npy")
    n = G.shape[1] - 1
    bands = {"mid": [L for L in range(n) if 0.2 <= L / n <= 0.45], "late": [L for L in range(n) if 0.6 <= L / n <= 0.9]}
    res = {}
    for d in doms:
        idx = np.array([i for i, r in enumerate(rows) if r["domain"] == d and r["perspective"] == "self"])
        o = np.array([rows[i]["outcome"] for i in idx]); s = np.array([rows[i]["surface"] for i in idx]); c = s != o
        v = {}
        for bn, Ls in bands.items():
            a = []
            for L in Ls:
                x = np.asarray(S[:, L], np.float32); w = x[ys == 1].mean(0) - x[ys == 0].mean(0)
                p = (np.asarray(G[idx, L], np.float32) - x.mean(0)) @ w
                a.append(roc_auc_score(o[c], p[c]))
            v[bn] = float(np.mean(a))
        res[d] = v
    out[tag] = res
    print(tag, pos, " ".join(f"{d}:{res[d]['mid']:.2f}->{res[d]['late']:.2f}" for d in doms), flush=True)
json.dump(out, open(R / "results" / f"domain_band_{pos}.json", "w"), indent=1)
