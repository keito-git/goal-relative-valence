"""Band-averaged conflict AUC and appraisal share with item-level bootstrap CIs (confirmation domains, SST-2 diff-of-means).

Bands by relative depth: mid [0.20, 0.45], late [0.60, 0.90] (final layer excluded). Self and other perspectives.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
R = Path(os.environ.get("GOALVAL_ROOT", ".")); B = 1000
rows = [json.loads(l) for l in open(R / "data/goal_stimuli.jsonl")]
surf = np.array([r["surface"] for r in rows]); outc = np.array([r["outcome"] for r in rows])

def strat_d(s, f, st):
    ds = []
    for v in (0, 1):
        m = st == v; a, b = s[m & (f == 1)], s[m & (f == 0)]
        ds.append((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    return np.mean(ds)

out = {}
pos = sys.argv[1]
for tag in sys.argv[2:]:
    A = R / "acts" / tag
    G = np.load(A / f"goal_{pos}.npy", mmap_mode="r"); S = np.load(A / f"sst2_{pos}.npy", mmap_mode="r"); ys = np.load(A / "sst2_labels.npy")
    n = G.shape[1] - 1
    bands = {"mid": [L for L in range(n) if 0.2 <= L / n <= 0.45], "late": [L for L in range(n) if 0.6 <= L / n <= 0.9]}
    res = {}
    for persp in ("self", "other"):
        m = np.array([r["split"] == "confirmation" and r["perspective"] == persp for r in rows]); idx = np.where(m)[0]
        P = {}
        for L in sorted(set(sum(bands.values(), []))):
            s = np.asarray(S[:, L + 0], np.float32)
            w = s[ys == 1].mean(0) - s[ys == 0].mean(0)
            P[L] = (np.asarray(G[idx, L], np.float32) - s.mean(0)) @ w
        o, sf = outc[idx], surf[idx]
        rng = np.random.default_rng(20261006)
        def stats(sel):
            out_ = {}
            for bn, Ls in bands.items():
                auc, rr = [], []
                for L in Ls:
                    p = P[L][sel]; oo, ss = o[sel], sf[sel]; c = ss != oo
                    auc.append(roc_auc_score(oo[c], p[c]))
                    do, dsu = strat_d(p, oo, ss), strat_d(p, ss, oo); rr.append(do / (abs(do) + abs(dsu)))
                out_[bn] = (np.mean(auc), np.mean(rr))
            return out_
        point = stats(np.arange(len(idx)))
        boots = [stats(rng.integers(0, len(idx), len(idx))) for _ in range(B)]
        res[persp] = {bn: {"conflict_auc": point[bn][0], "r": point[bn][1],
                           "conflict_auc_ci": list(np.percentile([b[bn][0] for b in boots], [2.5, 97.5])),
                           "r_ci": list(np.percentile([b[bn][1] for b in boots], [2.5, 97.5]))} for bn in bands}
        res[persp]["late_minus_mid_auc_ci"] = list(np.percentile([b["late"][0] - b["mid"][0] for b in boots], [2.5, 97.5]))
    out[tag] = res
    s_ = res["self"]
    print(f"{tag:16s} {pos} self mid AUC={s_['mid']['conflict_auc']:.2f} [{s_['mid']['conflict_auc_ci'][0]:.2f},{s_['mid']['conflict_auc_ci'][1]:.2f}] "
          f"late AUC={s_['late']['conflict_auc']:.2f} [{s_['late']['conflict_auc_ci'][0]:.2f},{s_['late']['conflict_auc_ci'][1]:.2f}] "
          f"diff CI [{s_['late_minus_mid_auc_ci'][0]:.2f},{s_['late_minus_mid_auc_ci'][1]:.2f}] | late r={s_['late']['r']:.2f} [{s_['late']['r_ci'][0]:.2f},{s_['late']['r_ci'][1]:.2f}]", flush=True)
json.dump(out, open(R / "results" / f"band_boot_{pos}.json", "w"))
